// Mirrors docs/schema/structured-fields.schema.json.
import { Schema } from 'mongoose'

export const structuredFieldsSchema = new Schema(
  {
    subject: { type: String, required: true, minlength: 1 },
    style: { type: String, default: null },
    medium: { type: String, default: null },
    lighting: { type: String, default: null },
    modifiers: { type: [String], default: [] },
    tone: { type: String, default: null },
    negative_constraints: { type: [String], default: [] },
  },
  { _id: false },
)
