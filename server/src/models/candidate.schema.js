// Mirrors docs/schema/candidate.schema.json.
import { Schema } from 'mongoose'
import { structuredFieldsSchema } from './structuredFields.schema.js'

export const candidateSchema = new Schema(
  {
    id: { type: String, required: true, minlength: 1 },
    prompt: { type: String, required: true, minlength: 1 },
    source: { type: String, required: true, minlength: 1 },
    similarity: { type: Number, required: true, min: 0, max: 1 },
    structured_fields: { type: structuredFieldsSchema, default: null },
    thumbnail_url: { type: String, default: null },
  },
  { _id: false },
)
