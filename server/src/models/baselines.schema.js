// Mirrors docs/schema/baselines.schema.json.
import { Schema } from 'mongoose'

const pezBaselineSchema = new Schema(
  {
    prompt: { type: String, required: true, minlength: 1 },
    clip_score: { type: Number, min: -1, max: 1, default: null },
    latency_ms: { type: Number, min: 0, default: null },
  },
  { _id: false },
)

const clipTagBaselineSchema = new Schema(
  {
    prompt: { type: String, required: true, minlength: 1 },
    tags: { type: [String], default: [] },
    latency_ms: { type: Number, min: 0, default: null },
  },
  { _id: false },
)

export const baselinesSchema = new Schema(
  {
    pez: { type: pezBaselineSchema, default: null },
    clip_tag: { type: clipTagBaselineSchema, default: null },
  },
  { _id: false },
)
