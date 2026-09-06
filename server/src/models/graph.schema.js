// Mirrors docs/schema/graph.schema.json.
import { Schema } from 'mongoose'

const graphNodeSchema = new Schema(
  {
    id: { type: String, required: true, minlength: 1 },
    type: { type: String, required: true, enum: ['input', 'candidate'] },
    label: { type: String, required: true, minlength: 1 },
    prompt: { type: String, default: null },
    thumbnail_url: { type: String, default: null },
    style_cluster: { type: String, default: null },
  },
  { _id: false },
)

const graphEdgeSchema = new Schema(
  {
    source: { type: String, required: true, minlength: 1 },
    target: { type: String, required: true, minlength: 1 },
    weight: { type: Number, required: true, min: 0, max: 1 },
  },
  { _id: false },
)

export const graphSchema = new Schema(
  {
    nodes: { type: [graphNodeSchema], default: [] },
    edges: { type: [graphEdgeSchema], default: [] },
  },
  { _id: false },
)
