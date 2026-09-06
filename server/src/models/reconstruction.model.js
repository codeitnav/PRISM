// Mirrors docs/schema/reconstruction.schema.json.
import { Schema, model } from 'mongoose'
import { baselinesSchema } from './baselines.schema.js'
import { candidateSchema } from './candidate.schema.js'
import { confidenceSchema } from './confidence.schema.js'
import { graphSchema } from './graph.schema.js'
import { structuredFieldsSchema } from './structuredFields.schema.js'
import { timingsSchema } from './timings.schema.js'

const reconstructionInputSchema = new Schema(
  {
    filename: { type: String, default: null },
    content_type: { type: String, default: null },
    text: { type: String, default: null },
    storage_path: { type: String, required: true, minlength: 1 },
  },
  { _id: false },
)

const reconstructionSchema = new Schema(
  {
    modality: { type: String, required: true, enum: ['image', 'text'] },
    status: {
      type: String,
      required: true,
      enum: ['pending', 'processing', 'completed', 'failed', 'degraded'],
      default: 'pending',
    },
    input: { type: reconstructionInputSchema, required: true },
    structured_fields: { type: structuredFieldsSchema, default: null },
    candidates: { type: [candidateSchema], default: [] },
    confidence: { type: confidenceSchema, default: null },
    baselines: { type: baselinesSchema, required: true, default: () => ({}) },
    graph: { type: graphSchema, default: null },
    timings: { type: timingsSchema, default: null },
    error: { type: String, default: null },
  },
  {
    collection: 'reconstructions',
    timestamps: { createdAt: 'created_at', updatedAt: 'updated_at' },
    toJSON: {
      virtuals: true,
      transform(_doc, ret) {
        ret.id = ret._id.toString()
        delete ret._id
        delete ret.__v
        return ret
      },
    },
  },
)

export const Reconstruction = model('Reconstruction', reconstructionSchema)
