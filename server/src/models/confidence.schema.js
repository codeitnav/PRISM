// Mirrors docs/schema/confidence.schema.json.
import { Schema } from 'mongoose'

const confidenceComponentsSchema = new Schema(
  {
    cos_sim: { type: Number, min: 0, max: 1, default: null },
    retrieval_margin: { type: Number, required: true, min: 0, max: 1 },
    component_agreement: { type: Number, required: true, min: 0, max: 1 },
  },
  { _id: false },
)

const confidenceWeightsSchema = new Schema(
  {
    alpha: { type: Number, required: true },
    beta: { type: Number, required: true },
    gamma: { type: Number, required: true },
  },
  { _id: false },
)

export const confidenceSchema = new Schema(
  {
    score: { type: Number, required: true, min: 0, max: 1 },
    components: { type: confidenceComponentsSchema, required: true },
    weights: { type: confidenceWeightsSchema, default: null },
  },
  { _id: false },
)
