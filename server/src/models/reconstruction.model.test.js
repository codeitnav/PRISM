import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

import Ajv2020 from 'ajv/dist/2020.js'
import addFormats from 'ajv-formats'

import { Reconstruction } from './reconstruction.model.js'

// Validates the shared example fixture against both the raw JSON Schema
// (docs/schema/*.schema.json) and the Mongoose model that mirrors it. A
// failure here after a schema edit means the Mongoose mirror is out of sync
// and needs to be updated to match.

const __dirname = path.dirname(fileURLToPath(import.meta.url))
// Locally (repo checkout) docs/ sits three levels up from this file. In Docker,
// only server/ is mounted as the app - docs/ is mounted separately
// (docker-compose.yml) and SCHEMA_DIR is set to point at it directly.
const REPO_ROOT = path.resolve(__dirname, '../../../')
const SCHEMA_DIR = process.env.SCHEMA_DIR || path.join(REPO_ROOT, 'docs', 'schema')
const EXAMPLES_DIR = path.join(SCHEMA_DIR, 'examples')

const SCHEMA_FILES = [
  'structured-fields.schema.json',
  'candidate.schema.json',
  'confidence.schema.json',
  'baselines.schema.json',
  'timings.schema.json',
  'graph.schema.json',
  'error.schema.json',
  'reconstruct-request.schema.json',
  'reconstruction.schema.json',
]

function loadSchema(filename) {
  return JSON.parse(readFileSync(path.join(SCHEMA_DIR, filename), 'utf-8'))
}

function loadExample(filename) {
  return JSON.parse(readFileSync(path.join(EXAMPLES_DIR, filename), 'utf-8'))
}

function buildAjv() {
  // strictRequired is disabled: our if/then pattern (reconstruct-request.schema.json)
  // requires 'text' conditionally, referencing a property declared in the parent
  // schema rather than repeated inside the `then` branch - valid JSON Schema, but
  // ajv's strict mode flags it as if it were a typo.
  const ajv = new Ajv2020({ strict: true, strictRequired: false })
  addFormats(ajv)
  for (const file of SCHEMA_FILES) {
    ajv.addSchema(loadSchema(file))
  }
  return ajv
}

test('reconstruction example matches JSON Schema', () => {
  const ajv = buildAjv()
  const validate = ajv.getSchema('prism://schema/reconstruction.schema.json')
  const example = loadExample('reconstruction.example.json')

  const valid = validate(example)
  assert.equal(valid, true, JSON.stringify(validate.errors))
})

test('reconstruction example matches Mongoose model', () => {
  const example = loadExample('reconstruction.example.json')
  const { id: _id, ...rest } = example
  const doc = new Reconstruction(rest)

  const err = doc.validateSync()
  assert.equal(err, undefined, err?.message)
})

test('reconstruct-request example matches JSON Schema', () => {
  const ajv = buildAjv()
  const validate = ajv.getSchema('prism://schema/reconstruct-request.schema.json')
  const example = loadExample('reconstruct-request.example.json')

  const valid = validate(example)
  assert.equal(valid, true, JSON.stringify(validate.errors))
})

test('reconstruct-request schema rejects text modality with no text', () => {
  const ajv = buildAjv()
  const validate = ajv.getSchema('prism://schema/reconstruct-request.schema.json')

  const valid = validate({ modality: 'text' })
  assert.equal(valid, false)
})
