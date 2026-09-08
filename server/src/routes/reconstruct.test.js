import { test, before, after } from 'node:test'
import assert from 'node:assert/strict'
import mongoose from 'mongoose'
import request from 'supertest'

import { createApp } from '../app.js'
import { env } from '../config/env.js'

const IMAGE_PATH = '/data/diffusiondb/images/000002.png'

before(async () => {
  await mongoose.connect(env.mongoUri)
})

after(async () => {
  await mongoose.disconnect()
})

test('POST /api/reconstruct returns a persisted reconstruction retrievable via GET', async () => {
  const app = createApp()

  const postRes = await request(app)
    .post('/api/reconstruct')
    .field('modality', 'image')
    .field('top_k', '5')
    .attach('image', IMAGE_PATH)

  assert.equal(postRes.status, 200, JSON.stringify(postRes.body))
  assert.equal(postRes.body.modality, 'image')
  assert.equal(postRes.body.status, 'completed')
  assert.ok(postRes.body.id)
  assert.ok(Array.isArray(postRes.body.candidates))
  assert.equal(postRes.body.candidates.length, 5)
  assert.ok(postRes.body.timings.total_ms >= 0)

  const getRes = await request(app).get(`/api/reconstruct/${postRes.body.id}`)
  assert.equal(getRes.status, 200)
  assert.equal(getRes.body.id, postRes.body.id)
  assert.equal(getRes.body.candidates.length, 5)
})

test('POST /api/reconstruct rejects a request with no modality', async () => {
  const app = createApp()
  const res = await request(app).post('/api/reconstruct').attach('image', IMAGE_PATH)
  assert.equal(res.status, 400)
  assert.equal(res.body.error, 'validation_error')
})

test('GET /api/reconstruct/:id returns 404 for an unknown id', async () => {
  const app = createApp()
  const res = await request(app).get(`/api/reconstruct/${new mongoose.Types.ObjectId()}`)
  assert.equal(res.status, 404)
  assert.equal(res.body.error, 'not_found')
})
