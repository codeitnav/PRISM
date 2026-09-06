// Mirrors docs/schema/timings.schema.json.
import { Schema } from 'mongoose'

export const timingsSchema = new Schema(
  {
    embedding_ms: { type: Number, required: true, min: 0 },
    retrieval_ms: { type: Number, required: true, min: 0 },
    captioning_ms: { type: Number, min: 0, default: null },
    decomposition_ms: { type: Number, min: 0, default: null },
    pez_ms: { type: Number, min: 0, default: null },
    regeneration_ms: { type: Number, min: 0, default: null },
    total_ms: { type: Number, required: true, min: 0 },
  },
  { _id: false },
)
