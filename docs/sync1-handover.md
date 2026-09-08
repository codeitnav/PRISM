# Sync Point 1 Handover — Retrieval Handoff

**From:** Kajal (Person A — Platform/Data/Retrieval)
**To:** Navya (Person B — Modeling/Intelligence)
**Date:** 2026-09-08
**Status:** All of Person A's pre-Sync-1 tasks (1.1, 1.2, 2.1–2.4) are complete and tested.

---

## 1. What's done

| Task | Output |
|---|---|
| 1.1 — DiffusionDB ingestion | 700 curated image-prompt pairs: `data/diffusiondb/images/*.png` + `data/diffusiondb/pairs.parquet` |
| 1.2 — Held-out splits | `data/splits/{train,val,test}.json` — 560 / 70 / 70, prompt-disjoint (verified zero cross-split near-duplicates above cosine 0.95) |
| 2.1 — Embedding service | `ml/app/embed.py`: `embed_images()`, `embed_texts()`, both cached to disk by content hash |
| 2.2 — FAISS index | `data/faiss/index.faiss` + `id_map.json`, built from the **560 train-split images only** |
| 2.3 — Mongo seeding | `reference_prompts` collection, 560 docs |
| 2.4 — Retrieval endpoint | `POST /internal/retrieve` on the `ml` service, tested end-to-end |

Full test suite: **12/12 passing** (`docker compose run --rm ml pytest`).

---

## 2. Scope deviations you need to know about

This dev machine is CPU-only (no GPU) with limited disk space, so a few things were scoped down from the original brief. Documented here so it's not a surprise later:

- **Dataset size:** 700 pairs, not the brief's ~200–500K. Used DiffusionDB's official small predefined split (`2m_random_5k`, capped at 700 kept pairs) instead of manually curating from the full 2M/14M shards.
- **Image embedding model:** **CLIP ViT-B/32** (512-dim), not the brief's ViT-L/14. ViT-L/14 measured ~40s/image on this CPU — impractical. ViT-B/32 measured ~6.3s/image cold, ~0.3s/image batched/warm. Same OpenAI CLIP family, so PEZ (Task 4.1, still using CLIP) shares the same embedding space.
- **Text embedding model:** `all-MiniLM-L6-v2` (384-dim), not E5-large. The brief allows "E5-large or BGE," so this is within spec, just the lightest reasonable option (~80MB vs ~1.3GB).

**If your work (1.3, 5.1, 5.2) ever calls CLIP directly or assumes a specific embedding dimension, use ViT-B/32 / 512-dim to match what's already indexed** — don't re-embed with ViT-L/14, the two wouldn't be comparable.

---

## 3. The retrieval endpoint — how to call it

```
POST http://localhost:8000/internal/retrieve
Content-Type: multipart/form-data

Fields:
  image   (file, required)  — the AI-generated image to query
  top_k   (int, optional, default 10)
```

**Response:** JSON array of `Candidate` objects (matches `docs/schema/candidate.schema.json` exactly):

```json
[
  {
    "id": "000002",
    "prompt": "still from studio ghibli movie My Neighbor Totoro, Hayao Miy...",
    "source": "diffusiondb",
    "similarity": 0.9999,
    "structured_fields": null,
    "thumbnail_url": null
  },
  ...
]
```

Note: `retrieval_margin` (top1 − top2 similarity) is **not** returned by this endpoint — the frozen contract only specifies `Candidate[]`. It's trivially derivable from `candidates[0].similarity - candidates[1].similarity` whenever you need it for Task 6.2 (confidence scoring).

---

## 4. Where your Task 1.3 output goes

Your weak-labeling output (structured fields per prompt) should be written back into the **same Mongo documents**, not a new collection:

- Collection: `reference_prompts`
- Key: `_id` = the dataset id string (e.g. `"000002"`) — same id used in `data/faiss/id_map.json` and in `data/diffusiondb/pairs.parquet`'s `id` column
- Field to fill in: `structured_fields`, currently `null` on every document, shaped as:
  ```json
  {
    "subject": "...",        // required
    "style": "...",          // optional
    "medium": "...",         // optional
    "lighting": "...",       // optional
    "modifiers": ["..."],    // list, defaults to []
    "tone": "...",           // optional
    "negative_constraints": ["..."]  // list, defaults to []
  }
  ```
  (This mirrors `docs/schema/structured-fields.schema.json` / `ml/app/schemas/structured_fields.py` exactly — use those as the source of truth if this drifts.)
- Use `update_one({"_id": dataset_id}, {"$set": {"structured_fields": {...}}})` per document — don't `delete_many`/reinsert, that would wipe the `faiss_id` field my seeding script populated.

Your raw prompts to label are readable from `data/diffusiondb/pairs.parquet` (columns: `id, image_path, prompt, seed, cfg, sampler`) — all 700, not just the 560 train-split ones, in case you want labels for val/test too.

---

## 5. What's NOT ready yet (blocks your Task 5.2 partially)

Your **Task 5.2** (decomposition SFT dataset construction) depends on both my **2.4** (✅ done) and my **4.1 — PEZ baseline** (❌ not started). I'm doing 3.1 (backend orchestration) next, then 4.1/4.2 (baselines). I'll flag you again once 4.1 lands.

Your **Task 1.3** (weak labels) and **5.1** (captioning) have no dependency on 4.1 — you're clear to start those now.

---

## 6. One practical thing to sort out together

`data/` (images, parquet, splits, FAISS index) is **gitignored** — not committed to the repo (see `data/README.md`). If you're on a different machine, you'll need either:
- a copy of my `data/` folder directly, or
- to re-run `make ingest` / `docker compose run --rm ml python scripts/ingest_diffusiondb.py` yourself (deterministic — same HF dataset config, will produce the same 700 pairs), then `make split` / `build-index` / `seed` in that order.

Confirm which approach works for your setup before you start 1.3.
