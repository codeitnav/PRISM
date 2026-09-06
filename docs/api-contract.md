# API Contract

**Status: frozen for Phase 3 (Task 0.2).** Changing any schema here requires updating the JSON Schema file, the Pydantic model (`ml/app/schemas/`), the Mongoose schema (`server/src/models/`), and the shared example under `docs/schema/examples/` together, then re-running the contract tests (see bottom of this doc). Treat these four as one unit — never edit one without the others.

This is the interface between `client`, `server`, and `ml`. `server` never contains ML logic; it validates/orchestrates and forwards to `ml`, and both `server` and `ml` read/write the same `Reconstruction` shape so it can be persisted once in MongoDB and served to the client unchanged.

## Source of truth

The canonical schemas are JSON Schema (draft 2020-12) files under [`docs/schema/`](schema/):

| File | Defines |
|---|---|
| `structured-fields.schema.json` | `StructuredFields` |
| `candidate.schema.json` | `Candidate` |
| `confidence.schema.json` | `Confidence` |
| `baselines.schema.json` | `Baselines` |
| `timings.schema.json` | `Timings` |
| `graph.schema.json` | `Graph`, `GraphNode`, `GraphEdge` |
| `error.schema.json` | `Error` |
| `reconstruct-request.schema.json` | `ReconstructRequest` (the non-file form fields) |
| `reconstruction.schema.json` | `Reconstruction` (the top-level object; composes all of the above except `Error`/`ReconstructRequest`) |

A worked example of a complete `Reconstruction` lives at [`docs/schema/examples/reconstruction.example.json`](schema/examples/reconstruction.example.json) — read that alongside this doc, it's easier to follow than the schema alone.

**Derived, not hand-drifted:** `ml/app/schemas/*.py` (Pydantic v2) and `server/src/models/*.js` (Mongoose) are maintained as field-for-field mirrors of these JSON Schema files — same field names, same required/optional-ness, same enums. A contract test on each side (`ml/tests/test_contract.py`, `server/src/models/reconstruction.model.test.js`) loads the shared example and validates it against both the raw JSON Schema and the language-specific model, so the three can't silently drift apart.

## Endpoints

### `POST /api/reconstruct`

Multipart form (not JSON — the payload includes a binary image file). Request shape: [`reconstruct-request.schema.json`](schema/reconstruct-request.schema.json).

| Field | Type | Required | Notes |
|---|---|---|---|
| `modality` | `"image" \| "text"` | yes | Selects the pipeline. |
| `image` | file (multipart part) | required if `modality=image` | Not expressible in JSON Schema since it's a binary part, not a JSON field — validated by the server directly (content-type allowlist, size limit). |
| `text` | string | required if `modality=text` | The AI-generated text to reconstruct a prompt from. |
| `top_k` | integer, default `10` | no | Number of candidates to retrieve. |
| `include_regeneration` | boolean, default `false` | no | Enables the slow reconstruction-consistency step (Task 6.1). Off by default. |

**Response — `200 OK`:** a full [`Reconstruction`](schema/reconstruction.schema.json) object with `status: "completed"` (or `"degraded"`, see below).

**Response — `4xx/5xx`:** an [`Error`](schema/error.schema.json) object.

> Phase 3 implements this synchronously (the response is the finished reconstruction). `status`/`pending`/`processing` are reserved in the schema now so the endpoint can become async later (e.g. return `202` + poll `GET /api/reconstruct/:id`) without a breaking schema change.

### `GET /api/reconstruct/:id`

Returns the persisted [`Reconstruction`](schema/reconstruction.schema.json) by id, or a `404` [`Error`](schema/error.schema.json) if not found.

## The `Reconstruction` object

Full schema: [`reconstruction.schema.json`](schema/reconstruction.schema.json). Summary:

```
Reconstruction
├── id, modality, status, created_at, updated_at
├── input                 { filename, content_type, text, storage_path }
├── structured_fields      StructuredFields | null   ← the core contribution (Task 5.x)
├── candidates[]           Candidate[]                ← FAISS retrieval results (Task 2.x)
├── confidence             { score, components: { cos_sim, retrieval_margin, component_agreement }, weights } | null
├── baselines               { pez, clip_tag }          ← Task 4.x, shown for comparison
├── graph                  { nodes[], edges[] } | null ← drives the frontend similarity graph (Task 3.3)
├── timings                per-stage latency ms | null
└── error                  string | null
```

**`status` values:**
- `pending` / `processing` — reserved for a future async flow, unused by the current synchronous endpoint.
- `completed` — full pipeline ran, including the fine-tuned decomposition model.
- `degraded` — the decomposer's output failed schema validation and the response fell back to the retrieval top-1 candidate's fields (Task 5.4). Still usable, must be visibly flagged in the UI (Task 8.4).
- `failed` — see `error` for details; other fields may be partially populated.

**On confidence:** this is a calibrated score, not a claim that the reconstructed prompt is the one actually used — see the project's framing (ill-posed inference, plausible reconstruction). `components` and `weights` are exposed so the UI can show the breakdown, not just a bare number (Task 8.4).

**`baselines.pez` / `baselines.clip_tag` can each be `null`** — they're computed for evaluation/comparison and may be skipped in latency-sensitive contexts; `structured_fields` (the actual product output) does not depend on either being present.

## The `Graph` payload

Full schema: [`graph.schema.json`](schema/graph.schema.json). Embedded in `Reconstruction.graph` — there is no separate graph endpoint. `nodes[0]` by convention is always `{id: "input", type: "input", ...}`; every other node is `type: "candidate"` and its `id` matches a `Candidate.id`. Edge `weight` is similarity in `[0,1]`; the frontend maps this to force-directed layout distance as `1 - weight` (Task 3.3) — the payload carries no layout/position data, that's computed client-side.

## Error format

All non-2xx responses from both `server` and `ml` return [`error.schema.json`](schema/error.schema.json):

```json
{ "error": "validation_error", "message": "modality must be 'image' or 'text'", "details": null }
```

`error` is a stable machine-readable code (`validation_error`, `not_found`, `ml_service_unavailable`, `internal_error`, ...); `message` is for logs/debugging, not guaranteed stable wording; `details` is optional structured context (e.g. field-level errors).

## Internal ML service routes

Not part of the public contract (no client ever calls these directly — `server` does), but they return fragments of the same schema so `server` can assemble the final `Reconstruction`:

- `POST /internal/retrieve` → `Candidate[]` (Task 2.4)
- other internal routes (captioning, decomposition, PEZ, confidence) are added as their tasks land; each returns a fragment already shaped to slot into `Reconstruction` with no reshaping in `server`.

## Contract tests

Both sides validate the same fixture (`docs/schema/examples/reconstruction.example.json`) against the raw JSON Schema *and* their own generated/mirrored model:

```bash
# ml
docker compose run --rm ml pytest tests/test_contract.py

# server
docker compose run --rm server npm test -- --test-name-pattern=contract
```

If you change a field: edit the `.schema.json` file first, update the matching example if the shape changed, then update both `ml/app/schemas/` and `server/src/models/` to match, then re-run both of the above before merging.
