import { test } from 'node:test'
import assert from 'node:assert/strict'
import request from 'supertest'
import { createApp } from './app.js'

test('GET /health returns 200 and ok status', async () => {
  const app = createApp()
  const res = await request(app).get('/health')
  assert.equal(res.status, 200)
  assert.equal(res.body.status, 'ok')
  assert.equal(res.body.service, 'server')
})

test('GET /unknown-route returns 404', async () => {
  const app = createApp()
  const res = await request(app).get('/unknown-route')
  assert.equal(res.status, 404)
})
