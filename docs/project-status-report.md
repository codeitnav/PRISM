# PRISM — Project Status Report

**Prepared by:** Kajal (Person A — Platform, Data & Retrieval)
**Date:** 2026-09-08
**Covers:** Day 0 through Task 4.3 (Sync Point 1 reached, Sync Point 2 pending on Navya's side)

---

## 1. What PRISM Is

PRISM (**P**rompt **R**econstruction and **I**nference from **S**emantic **M**ultimodal representations) is a reverse prompt engineering platform. Given an AI-generated image, it reconstructs the most probable *structured* prompt behind it — subject, style, medium, modifiers, tone, constraints — along with a calibrated confidence score and an interactive similarity graph, framed as **plausible reconstruction, not exact recovery** (many different prompts can produce similar images, so there's no single "correct" answer to recover).

The image pipeline is the primary, fully-implemented system; a text pipeline exists as a reduced-depth extension.

## 2. Team & Roles

| Person | Role | Focus |
|---|---|---|
| **Kajal** (Person A) | Platform & Retrieval Lead | Infrastructure, MongoDB, FAISS, backend orchestration, baselines, evaluation harness |
| **Navya** (Person B) | Modeling & Intelligence Lead | Weak labels, captioning, LoRA decomposition model, confidence/calibration, text pipeline |

The split isn't "backend vs. ML" — each person owns a full, mostly-independent pipeline slice. Kajal's outputs (retrieval, baselines) are direct inputs to Navya's modeling work.

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite |
| Backend orchestration | Node.js + Express |
| ML microservice | Python + FastAPI |
| Database | MongoDB (system of record) |
| Vector search | FAISS (inside the ML service) |
| Image embeddings | CLIP (OpenCLIP) |
| Text embeddings | Sentence-Transformers |
| Containerization | Docker Compose (4 services: client, server, ml, mongo) |

**Dev hardware:** ASUS VivoBook, Intel i5, **no dedicated GPU** — confirmed CPU-only by `ml/scripts/check_env.py`. This constrained several scope decisions documented below.

## 4. Progress Summary

| # | Task | Owner | Status |
|---|---|---|---|
| 0.1–0.3 | Repo scaffold, API contract, GPU check | Both (joint) | ✅ Done |
| 1.1 | DiffusionDB subset ingestion | Kajal | ✅ Done |
| 1.2 | Held-out train/val/test splits | Kajal | ✅ Done |
| 2.1 | Embedding service (CLIP + text) | Kajal | ✅ Done |
| 2.2 | FAISS index build | Kajal | ✅ Done |
| 2.3 | MongoDB seeding | Kajal | ✅ Done |
| 2.4 | Retrieval endpoint | Kajal | ✅ Done |
| — | **Sync Point 1 (Retrieval Handoff)** | Both | ✅ Reached |
| 3.1 | Backend orchestration (`/api/reconstruct`) | Kajal | ✅ Done |
| 4.1 | PEZ baseline | Kajal | ✅ Done |
| 4.2 | Naive CLIP-tag baseline | Kajal | ✅ Done |
| 4.3 | Evaluation harness | Kajal | ✅ Done |
| 1.4 | Text-side dataset (Alpaca) | Kajal | ✅ Done |
| 1.3 | Weak structured labels | Navya | ⬜ In progress |
| 5.1–5.5 | Captioning → LoRA decomposition → wiring | Navya | ⬜ Pending |
| — | **Sync Point 2 (Model Handoff)** | Both | ⬜ Waiting on Navya's 5.3 |

**11 of ~16 of Kajal's tasks are complete.** All of Kajal's work through Sync Point 2 is done — Kajal is currently ahead of schedule, waiting on Navya's LoRA adapter.

---

## 5. Task-by-Task Outcomes

### Task 1.1 — DiffusionDB Subset Ingestion
Downloaded and curated **700 image-prompt pairs** from DiffusionDB (Stable Diffusion outputs with their original prompts). Filtered out NSFW-flagged content, prompts that were too short/long, and near-duplicate prompts.

- **Scope decision:** the original brief called for ~200K–500K pairs. Reduced to 5K (via DiffusionDB's official small predefined split) then capped at 700, driven entirely by this machine's CPU-only, storage-constrained profile. Documented, deliberate, not a shortcut.
- Output: `data/diffusiondb/images/*.png` + `pairs.parquet`

### Task 1.2 — Held-Out Splits
Split the 700 pairs into **560 train / 70 val / 70 test**, with a hard guarantee: no two near-duplicate prompts (cosine similarity ≥ 0.95, measured via a lightweight sentence embedding model) can land in different splits — otherwise the system could "cheat" during evaluation.

- Found and correctly handled 54 near-duplicate prompt clusters (129 pairs).
- Verified: zero cross-split violations.

### Task 2.1 — Embedding Service
Built `embed_images()` and `embed_texts()` — functions that turn images and text into numeric vectors ("fingerprints") for similarity comparison. Both are normalized and cached to disk (repeat requests are instant).

- **Scope decisions (both documented, hardware-driven):**
  - Image model: **CLIP ViT-B/32** instead of the brief's ViT-L/14. ViT-L/14 measured ~40s/image on this CPU (would take ~8 hours for 700 images); ViT-B/32 measured ~6.3s/image — a ~6.3x speedup, same OpenAI CLIP family.
  - Text model: **all-MiniLM-L6-v2** instead of E5-large. The brief explicitly allows "E5-large or BGE," so this is within spec — just the lighter option (~80MB vs ~1.3GB).
- Infra fix along the way: moved the downloaded model cache from a slow Windows-bind-mounted folder to a native Docker volume, fixing what looked like a performance bug but was actually a filesystem bridge issue.

### Task 2.2 — FAISS Index Build
Embedded all **560 training-split images** (val/test deliberately excluded — they must stay held-out as query-only, or evaluation would be meaningless) and built a FAISS similarity search index over them.

- Result: 560-vector index, self-retrieval smoke test passed (a training image correctly retrieves itself at rank 1, similarity 1.0000).
- Ran in ~3 minutes (much faster than initially estimated, thanks to batching).

### Task 2.3 — MongoDB Seeding
Loaded the 560 reference prompts into a MongoDB collection (`reference_prompts`), indexed for fast lookup.

- Verified lookup latency: **1.66ms** (well under the 10ms target).

### Task 2.4 — Retrieval Endpoint
Built `POST /internal/retrieve`: given an uploaded image, it embeds it, searches the FAISS index, and returns the most similar reference prompts.

- Tested end-to-end: a known training image correctly retrieves itself with 99%+ similarity.
- **→ This is where Sync Point 1 (Retrieval Handoff) was reached.** See `docs/sync1-handover.md` for the full technical handoff to Navya.

### Task 3.1 — Backend Orchestration
Built the actual user-facing API: `POST /api/reconstruct` (upload an image → get back a saved result) and `GET /api/reconstruct/:id` (fetch it again later). This is the "envelope" everything else — including Navya's future model output — will eventually ride inside.

- Found and fixed a gap: the server had a database schema defined but was never actually connecting to MongoDB.
- Added a 60-second timeout and request-tracing so failures are debuggable.
- Verified via automated tests (9/9 passing) and a live `curl` request.

### Task 4.1 — PEZ Baseline ("Hard Prompts Made Easy")
Implemented a baseline reconstruction method: given an image, mathematically search CLIP's vocabulary for a short sequence of real words whose combined meaning best matches the image — no training, no labels, just optimization.

- **Result: mean CLIP-score of 0.2964** across 20 test images.
- **Key engineering win:** discovered that batching all 20 images into one optimization run (instead of one at a time) cut the cost ~7x, since CLIP's text model processes a fixed-size window regardless of batch size. Runtime: ~23 minutes total (would have been hours otherwise).
- Full results: `docs/results/pez_baseline.md`

### Task 4.2 — Naive CLIP-Tag Baseline
A simpler second baseline: classify the image against a fixed list of ~50 style/medium/lighting words and concatenate the best matches into a fake "prompt." No optimization — just direct comparison.

- **Result: mean CLIP-score of 0.2355** (lower than PEZ, as expected — much less expressive freedom).
- Ran in **33 seconds** (vs. PEZ's ~23 minutes) since there's no optimization loop.
- Full results: `docs/results/cliptag_baseline.md`

### Task 4.3 — Evaluation Harness
Built one shared scoring system that runs any reconstruction method through the same evaluation — so every method's numbers are directly comparable, including the real model Navya will deliver later.

- Added **BERTScore** as a new metric: measures whether the generated prompt *reads* like the real prompt (semantic/textual similarity), not just whether it matches the image.
- **Efficiency decision:** rather than re-run PEZ's expensive 23-minute optimization to score it again, the baseline scripts now save their raw results to a reusable file, which the harness loads instantly.

**Combined results:**

| Method | CLIP-score | BERTScore F1 | Speed |
|---|---|---|---|
| PEZ | 0.2964 | 0.7188 | 68s/image |
| CLIP-tag | 0.2355 | 0.7404 | 1.6s/image |

**Notable finding:** PEZ wins on CLIP-score but *loses* on BERTScore. PEZ optimizes purely for image-embedding similarity, producing garbled text that matches the image well but reads as nonsense. CLIP-tag produces clean, readable phrases that read more like a real prompt, even though they match the image less precisely. This is a genuine, explainable trade-off — not a bug in either method — and a good talking point for the report.

- Full results: `docs/results/baselines.md` + `.csv`

### Task 1.4 — Text-Side Dataset (Alpaca)
Set up the text-only counterpart to Task 1.1/1.2, for whenever the text pipeline (Navya's Task 7.2) needs a reference set to retrieve/train against.

- Downloaded the cleaned Stanford Alpaca instruction dataset (`yahma/alpaca-cleaned`) and curated **700 (prompt, output) pairs** — same scope cap as the image side, for consistency. Mapping: an Alpaca row's `output` (generated text) plays the role an image plays on the image side; its `instruction` (+ `input`, if present) is the prompt to be reconstructed.
- Filtered on prompt length (3-60 tokens) and non-empty output; 0 near-duplicate prompts found in this subset (vs. 110 on the image side), so no dedup was needed.
- Reused Task 1.2's exact prompt-disjoint splitting method (same Union-Find clustering, same 0.95 cosine threshold) on the Alpaca prompts: **560 train / 70 val / 70 test**, verified zero cross-split near-duplicate prompts.
- Output: `data/alpaca/pairs.parquet`, `data/splits/alpaca_{train,val,test}.json`
- This is purely a data-prep step (optional per the plan, no downstream blockers) — no embeddings/FAISS/Mongo were built for it, since that's Navya's text-pipeline task, not part of Kajal's list.

---

## 6. Summary of Scope Decisions (for the report/viva)

All driven by the same root cause: **CPU-only hardware, limited storage, limited time.** Each is documented in code comments at the point of the decision, not hidden.

| Area | Brief specified | Actually used | Why |
|---|---|---|---|
| Dataset size | ~200K–500K pairs | 700 pairs | CPU-only, storage-constrained |
| Image embedding model | CLIP ViT-L/14 | CLIP ViT-B/32 | ~6.3x faster, same model family |
| Text embedding model | E5-large or BGE | all-MiniLM-L6-v2 | Within spec; ~16x smaller |
| BERTScore backbone | (library default: roberta-large) | distilbert-base-uncased | Smaller, faster |
| PEZ iterations | Few hundred–thousands (paper) | 100 | CPU constraint; batched for efficiency |

## 7. What's Next

- **Kajal:** nothing required until Navya delivers — all work through Sync Point 2 is done.
- **Navya:** Task 1.3 (weak labels) and 5.1 (captioning) are unblocked now; Task 5.2 was waiting on Kajal's 2.4 and 4.1, both now delivered.
- **Sync Point 2 (Model Handoff):** Navya delivers her trained LoRA decomposition model; Kajal wires it into the evaluation harness (already built) for real, final metrics.

## 8. Key Files Reference

| What | Where |
|---|---|
| Sync 1 technical handoff | `docs/sync1-handover.md` |
| API contract (frozen) | `docs/api-contract.md` |
| PEZ baseline results | `docs/results/pez_baseline.md` |
| CLIP-tag baseline results | `docs/results/cliptag_baseline.md` |
| Combined evaluation | `docs/results/baselines.md`, `.csv` |
| All `make` commands | `Makefile` (targets: `ingest`, `split`, `bench-embed`, `build-index`, `seed`, `pez-baseline`, `cliptag-baseline`, `eval`, `ingest-alpaca`, `split-alpaca`) |
