// Backend orchestration for POST /api/reconstruct + GET /api/reconstruct/:id.
// server never contains ML logic - this route uploads/stores the file, calls
// the ml service's /internal/retrieve, /internal/caption, and
// /internal/decompose in sequence, persists the result, and returns it.
import crypto from 'node:crypto'
import fs from 'node:fs'
import { readFile } from 'node:fs/promises'
import path from 'node:path'

import { Router } from 'express'
import multer from 'multer'

import { env } from '../config/env.js'
import { Reconstruction } from '../models/reconstruction.model.js'

const router = Router()

const UPLOAD_DIR = '/data/uploads'
fs.mkdirSync(UPLOAD_DIR, { recursive: true })

const upload = multer({
  storage: multer.diskStorage({
    destination: UPLOAD_DIR,
    filename: (_req, file, cb) => {
      const ext = path.extname(file.originalname || '')
      cb(null, `${crypto.randomUUID()}${ext}`)
    },
  }),
  limits: { fileSize: 20 * 1024 * 1024 }, // 20MB
})

const ML_TIMEOUT_MS = 60_000

function errorPayload(error, message) {
  return { error, message, details: null }
}

// Captioning and decomposition are an enhancement on top of retrieval, not a
// hard requirement - a failure here degrades the response (falls back to the
// top-1 candidate's fields, status 'degraded') rather than failing the whole
// request, since retrieval already succeeded by the time these run.
async function callMlServiceOrNull(endpoint, init, requestId, stepName) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), ML_TIMEOUT_MS)
  const t0 = Date.now()
  try {
    const res = await fetch(`${env.mlServiceUrl}${endpoint}`, { ...init, signal: controller.signal })
    if (!res.ok) {
      throw new Error(`${endpoint} responded ${res.status}`)
    }
    return { data: await res.json(), ms: Date.now() - t0 }
  } catch (err) {
    console.error(`[${requestId}] ${stepName} failed, degrading:`, err.message)
    return null
  } finally {
    clearTimeout(timeout)
  }
}

router.post('/api/reconstruct', upload.single('image'), async (req, res) => {
  const requestId = crypto.randomUUID()
  const { modality } = req.body
  const topK = req.body.top_k ? Number(req.body.top_k) : 10
  console.log(`[${requestId}] POST /api/reconstruct modality=${modality}`)

  if (modality !== 'image' && modality !== 'text') {
    return res.status(400).json(errorPayload('validation_error', "modality must be 'image' or 'text'"))
  }
  if (modality === 'text') {
    // Text pipeline is a later, reduced-depth task (Task 7.2) - not wired up yet.
    return res.status(400).json(errorPayload('validation_error', 'text modality is not implemented yet'))
  }
  if (!req.file) {
    return res.status(400).json(errorPayload('validation_error', 'image file is required when modality is "image"'))
  }

  const t0 = Date.now()
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), ML_TIMEOUT_MS)

  try {
    const fileBuffer = await readFile(req.file.path)
    const form = new FormData()
    form.append('image', new Blob([fileBuffer], { type: req.file.mimetype }), req.file.originalname || 'upload')

    const mlRes = await fetch(`${env.mlServiceUrl}/internal/retrieve?top_k=${topK}`, {
      method: 'POST',
      body: form,
      signal: controller.signal,
    })
    if (!mlRes.ok) {
      throw new Error(`ml service responded ${mlRes.status}`)
    }
    const candidates = await mlRes.json()
    const retrievalMs = Date.now() - t0

    let status = 'completed'
    let structuredFields = candidates[0]?.structured_fields ?? null
    let captioningMs = null
    let decompositionMs = null

    const captionForm = new FormData()
    captionForm.append('image', new Blob([fileBuffer], { type: req.file.mimetype }), req.file.originalname || 'upload')
    const captionResult = await callMlServiceOrNull(
      '/internal/caption',
      { method: 'POST', body: captionForm },
      requestId,
      'captioning',
    )

    if (captionResult) {
      captioningMs = captionResult.ms
      const decomposeResult = await callMlServiceOrNull(
        '/internal/decompose',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            caption: captionResult.data.caption,
            retrieved_prompts: candidates.slice(0, 5).map((c) => c.prompt),
            pez_prompt: null,
          }),
        },
        requestId,
        'decomposition',
      )
      if (decomposeResult) {
        structuredFields = decomposeResult.data
        decompositionMs = decomposeResult.ms
      } else {
        status = 'degraded'
      }
    } else {
      status = 'degraded'
    }

    const doc = await Reconstruction.create({
      modality: 'image',
      status,
      input: {
        filename: req.file.originalname || null,
        content_type: req.file.mimetype || null,
        text: null,
        storage_path: req.file.path,
      },
      structured_fields: structuredFields,
      candidates,
      confidence: null,
      baselines: {},
      graph: null,
      timings: {
        // /internal/retrieve doesn't currently expose an embedding-only
        // split, so embedding_ms is reported as 0 rather than a fabricated
        // number - retrieval_ms is the honest full-call latency.
        embedding_ms: 0,
        retrieval_ms: retrievalMs,
        captioning_ms: captioningMs,
        decomposition_ms: decompositionMs,
        total_ms: Date.now() - t0,
      },
      error: null,
    })

    console.log(`[${requestId}] persisted reconstruction ${doc.id} in ${Date.now() - t0}ms`)
    return res.status(200).json(doc.toJSON())
  } catch (err) {
    console.error(`[${requestId}] reconstruct failed:`, err.message)
    if (err.name === 'AbortError') {
      return res.status(504).json(errorPayload('ml_service_unavailable', 'ML service did not respond within 60s'))
    }
    return res.status(502).json(errorPayload('ml_service_unavailable', err.message))
  } finally {
    clearTimeout(timeout)
  }
})

router.get('/api/reconstruct/:id', async (req, res) => {
  const doc = await Reconstruction.findById(req.params.id).catch(() => null)
  if (!doc) {
    return res.status(404).json(errorPayload('not_found', 'reconstruction not found'))
  }
  return res.status(200).json(doc.toJSON())
})

export default router
