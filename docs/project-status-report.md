# PRISM — Project Status Report

**Prepared by:** Kajal (Person A — Platform, Data & Retrieval)
**Date:** 2026-09-08
**Covers:** Day 0 through Task 4.3 (Kajal) and Tasks 1.3, 5.1, 5.2, 5.3 (Navya) — Sync Point 1 reached, Sync Point 2 pending on Navya's 5.3
**Last updated:** 2026-09-11 by Navya, adding Task 5.3

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
| 1.3 | Weak structured labels | Navya | ✅ Done |
| 5.1 | Captioning stage (BLIP) | Navya | ✅ Done |
| 5.2 | Decomposition SFT dataset construction | Navya | ✅ Done |
| 5.3 | LoRA fine-tune (SmolLM2-135M) | Navya | ✅ Done (Sync Point 2 deliverable) |
| — | **Sync Point 2 (Model Handoff)** | Both | ✅ Adapter ready: `ml/models/decomposer-lora` |
| 5.4 | Constrained decoding + fallback | Kajal *(picked up from Navya, by agreement)* | ✅ Done |
| 5.5 | Wire decomposition into pipeline | Kajal *(picked up from Navya, by agreement)* | ✅ Done |

**13 of Kajal's tasks are complete**, including Tasks 5.4 and 5.5, which were originally on Navya's side of the split but picked up by Kajal by mutual agreement once Sync Point 2 landed, so the pipeline wiring wouldn't sit idle waiting. Remaining work (6.1–6.4 confidence/calibration, 7.2 text pipeline, 8.3 failure analysis) is still Navya's.

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

**Combined results (updated after Sync Point 2 to add the LoRA decomposer):**

| Method | CLIP-score | BERTScore F1 | Speed |
|---|---|---|---|
| PEZ | 0.2964 | 0.7188 | 68s/image |
| CLIP-tag | 0.2355 | 0.7404 | 1.6s/image |
| LoRA decomposer | 0.1689 | 0.7199 | 3.3s/image* |

*\*Representative inference-time figure once the base model is cached (measured by Navya); the harness run on Kajal's machine measured higher because it included the one-time base-model download.*

**Notable finding:** PEZ wins on CLIP-score but *loses* on BERTScore. PEZ optimizes purely for image-embedding similarity, producing garbled text that matches the image well but reads as nonsense. CLIP-tag produces clean, readable phrases that read more like a real prompt, even though they match the image less precisely. This is a genuine, explainable trade-off — not a bug in either method — and a good talking point for the report.

**Decomposer, added after Sync Point 2:** its structured JSON output was flattened into a single prompt string so it could be scored on the same footing. It currently scores lowest on CLIP-score — consistent with the low component-level F1 (0.1248) Navya already measured — but its BERTScore sits between PEZ and CLIP-tag, since its output reads as a coherent sentence even when the specific details are wrong. Expected for a first fine-tune on 560 rows and a 135M-parameter model, not a harness issue. Full write-up: `docs/results/baselines.md`.

- Full results: `docs/results/baselines.md` + `.csv`

### Task 1.4 — Text-Side Dataset (Alpaca)
Set up the text-only counterpart to Task 1.1/1.2, for whenever the text pipeline (Navya's Task 7.2) needs a reference set to retrieve/train against.

- Downloaded the cleaned Stanford Alpaca instruction dataset (`yahma/alpaca-cleaned`) and curated **700 (prompt, output) pairs** — same scope cap as the image side, for consistency. Mapping: an Alpaca row's `output` (generated text) plays the role an image plays on the image side; its `instruction` (+ `input`, if present) is the prompt to be reconstructed.
- Filtered on prompt length (3-60 tokens) and non-empty output; 0 near-duplicate clusters found in this subset via the same cosine-0.95 clustering Task 1.2 uses (vs. 54 clusters / 129 pairs on the image side), so no dedup was needed. *(Correction, 2026-09-11: this line previously compared against 110, which is Task 1.1's ingestion-time exact-string-dedup count, not Task 1.2's clustering count — the wrong metric for an apples-to-apples comparison. Flagged by Navya in §9.)*
- Reused Task 1.2's exact prompt-disjoint splitting method (same Union-Find clustering, same 0.95 cosine threshold) on the Alpaca prompts: **560 train / 70 val / 70 test**, verified zero cross-split near-duplicate prompts.
- Output: `data/alpaca/pairs.parquet`, `data/splits/alpaca_{train,val,test}.json`
- This is purely a data-prep step (optional per the plan, no downstream blockers) — no embeddings/FAISS/Mongo were built for it, since that's Navya's text-pipeline task, not part of Kajal's list.

### Task 1.3 — Weak Structured Labels *(Navya)*
Turned every raw DiffusionDB prompt into the frozen `StructuredFields` schema
(subject / style / medium / lighting / modifiers / tone / negative_constraints) with a rule +
lexicon pass — the supervision targets for the decomposition model (5.2/5.3) and the ground
truth for component-wise scoring in the eval harness.

- Output: `data/diffusiondb/pairs_labeled.parquet` (all 700 rows, train+val+test), plus
  `structured_fields` written into all **560** `reference_prompts` documents via
  `update_one($set)` — **0 nulls remaining, `faiss_id` intact on all 560** (verified), so
  retrieval is unaffected.
- **Audited field-level precision (100 random samples, seed 1337): overall 0.742.**
  Per field: `medium` 1.00, `style` 0.93, `lighting` 0.93, `subject` 0.87, `tone` 0.75,
  `modifiers` 0.26. Full analysis: `docs/label-quality.md`.
- **Corpus-frequency mining paid off twice.** It exposed that these DiffusionDB configs
  store prompts *detokenized* — `3 d render`, `4 k`, `5 0 mm`, `art station` — which was
  silently costing ~230 lexicon hits across 700 prompts. Matching now normalizes the
  spacing, lifting `medium` coverage 37.9% → 50.3% (style, lighting, tone also up). It also
  surfaced ~40 genuine new vocabulary terms, now promoted into the lexicons.
- **The `modifiers` field is a known-weak junk drawer (precision 0.26)** — it is the sink
  for anything the lexicons don't recognise, so it absorbs scene detail that belongs in
  `subject` plus second-place values the single-slot schema has nowhere to put. *Action for
  5.2: do not train the decomposer to reproduce `modifiers` verbatim* — either weight the
  component loss towards the typed fields or filter `modifiers` to lexicon-matched entries
  for the SFT target.
- **`negative_constraints` is effectively empty — 4 of 700 rows (0.6%).** DiffusionDB's
  `2m_random_*` prompts predate widespread negative-prompt use. This component should be
  **excluded from the component-F1 headline** in 4.3/8.1 rather than reported as solved —
  raising at Sync Point 3.
- **Scope deviation:** the roadmap pairs the rule pass with 5–10K LLM-labeled seed prompts.
  That is larger than the entire 700-prompt corpus, and this environment has no LLM API
  credentials. The override hook is built and wired (`llm_seed_labels.jsonl` →
  per-prompt override, provenance tracked in a `label_provenance` column) but unused, so
  **all precision numbers are rule-pass-only**. The audit is also an assistant judgment
  pass rather than independent human annotation; all 314 per-field verdicts are stored in
  `data/diffusiondb/label_audit.jsonl` for re-checking.
- Tests: 23 added (`ml/tests/test_weak_label.py`); **ml suite now 38/38, server 9/9**.

### Task 5.1 — Captioning Stage *(Navya)*
Added a captioning stage: `app.caption.caption_images(list[bytes]) -> list[str]`, a
`POST /internal/caption` endpoint, and a runner that captions dataset splits. The caption is
one of three inputs the decomposition model is conditioned on in Task 5.2 — alongside
Kajal's retrieved candidates and her PEZ output — and it supplies what neither of those can:
a literal, grounded description of what is actually *in* the image, independent of whatever
the nearest training prompt happened to say.

- Captions cached to disk by image content hash (same pattern as `app.embed`), with the
  **model name in the cache key** so switching checkpoints can't serve stale captions.
  Deterministic beam search (`num_beams=3`), not sampling — 5.2 pairs each caption with a
  structured target, so captions must be reproducible across runs.
- **Done-condition met:** all **70 test-split** images captioned and persisted to
  `data/captions/captions.parquet`. 0 empty captions; mean length 13.8 words;
  **8.33 s/image** on this CPU. Train+val (630 more) captioned as well, since 5.2 needs the
  train split.
- **Scope deviation — model choice, measured not assumed.** The roadmap asks for BLIP-2, or
  LLaVA-1.5-7B "if VRAM allows". Reading each checkpoint's own config: `blip2-opt-2.7b` is
  ~3.74B params (OPT-2.7b + EVA ViT-g + Q-Former) = **~15 GB fp32 / ~7.5 GB bf16**, and
  LLaVA-1.5-7B ~28 GB / ~14 GB — against **~6.3 GB of RAM actually available** on this
  CPU-only box. Neither fits, even at bf16. Default is therefore **BLIP-1 large** (~470M,
  ~1.9 GB): the direct predecessor, same Salesforce BLIP captioning lineage. This is a
  *configuration* limit, not a code limit — `CAPTION_MODEL` selects any BLIP/BLIP-2
  checkpoint and the architecture is chosen from its config, so on a GPU machine
  `CAPTION_MODEL=Salesforce/blip2-opt-2.7b make caption` runs the roadmap's intended model
  with no code change.
- **Measured artifact worth knowing about for 5.2:** BLIP opens **55 of 70 captions (79%)**
  with a contentless phrase — `there is/are` (69%), `this is` (6%), `an image/picture of`
  (4%). As an input feature that is pure noise, so `strip_caption_boilerplate()` removes it
  (verified: 0/70 residual). It deliberately **preserves** `photo of`, `painting of`,
  `screenshot of`, `3d rendering of` — those name the medium, which is real visual evidence
  and one of the fields being reconstructed. The stripper is *not* applied inside
  `caption_images()`: the cache and parquet keep the raw model output as a faithful record,
  and callers opt in.
- Tests: 29 added (`ml/tests/test_caption.py`) — cache keying, batching, order preservation
  under partial cache hits, architecture dispatch, and the stripper's signal/noise boundary,
  plus one test that loads the real checkpoint end to end, and the endpoint contract (422 for bad input vs 503 for a genuinely unavailable model - the backend treats 5xx as transient). **ml suite now 67/67.**
- Results: `docs/results/captioning.md`

### Task 5.2 — Decomposition SFT Dataset *(Navya)*
Joined all four upstream artifacts into the supervised fine-tuning set Task 5.3 trains on:
`input = {caption, top-5 retrieved prompts, PEZ prompt}`, `target = the weak-labeled
structured JSON of the true prompt`.

- Output: **`data/decomp_sft.jsonl`** — **700 rows** (560 train / 70 val / 70 test),
  **700/700 schema-valid**. The held-out eval split travels *with* the dataset (every row
  carries its own `split`) rather than in a separate file that can drift out of sync.
- **Strict JSON targets enforced at the artifact level, not just in memory.**
  `app.sft.validate_row` re-checks every serialized row: `target_text` must parse, carry
  exactly the schema's keys in canonical order, satisfy `StructuredFields`, agree with the
  structured `target`, and be byte-identical to the canonical rendering. The builder exits
  non-zero on any failure, so a bad dataset cannot reach disk silently.
- **Found and fixed a leakage trap.** The FAISS index *is* the 560 train images, so every
  train row retrieved **itself** at rank 1 with similarity ~1.0 — and that prompt is exactly
  what its target was derived from. Left in, the dataset would have taught the model to copy
  `retrieved_prompts[0]`, ignore the caption and PEZ entirely, and then collapse at
  inference, where an unseen image is never in the index. Self-matches are dropped and the
  invariant is re-checked per row: the guard fired on **560 of 560 train rows and 0 val/test
  rows** — exactly where leakage exists and nowhere else.
- **The subtler half of that call.** My first version of the guard rejected any row whose
  true prompt appeared anywhere in `input_text`. It fired on row `000028` — but the culprit
  was image `000032`, a *different* image whose prompt contains `000028`'s verbatim. That is
  legitimate retrieval success which also happens at inference (the train split contains
  near-duplicate prompt clusters), and filtering it would make training inputs
  systematically weaker than production. So the guard now checks **identity, not string
  containment**, and containment is recorded instead as a statistic:
  **16/700 rows (2.3%)** — a useful ceiling on what a copy-only strategy could achieve when
  interpreting 5.3's component F1. That the task requires synthesis rather than copying for
  97.7% of rows is itself a result worth reporting.
- **Acted on the 1.3 audit.** Targets keep only lexicon-recognised modifiers
  (**1459 of 3063 entries, 47.6%**), since `modifiers` measured 0.26 precision and is the
  sink for unclassified segments. `modifiers_raw` is retained per row so Task 8.2 can ablate
  the choice. Captions use 5.1's `strip_caption_boilerplate()` — this is that documented
  opt-in point — affecting **576/700 rows (82%)**, with `caption_raw` also retained.
- **PEZ coverage 100%**, but it took a corrected measurement to get there. Kajal's PEZ output
  lives in gitignored `data/`, so it did not exist on this machine. My first throughput
  measurement said ~63 s/image (**~9.8 h** for the train split) — taken while captioning was
  saturating ~9 cores. Uncontended it is **7.8 s/image at batch 40**, and the full 700-image
  run took **121 min**, yielding mean CLIP-score **0.3000** against Kajal's 20-image
  benchmark of 0.2964 — confirming the larger batch is equivalent (the loss is
  batch-meaned, but Adam's update is invariant to a constant gradient scale and the
  per-image soft tokens are independent). `scripts/run_pez_for_sft.py` is resumable and
  writes to its own file, leaving her Task 4.1 artifact and the eval harness's assertions
  untouched.
- **Retrieval quality, for Task 6.2's benefit:** top-1 similarity after self-exclusion
  averages **0.8034** (min 0.4730, max 0.9955), but the **top1−top2 margin averages only
  0.0311** — candidates cluster very tightly, which matters for how much weight the
  retrieval-margin term can carry in the confidence formula.
- Tests: 37 added (`ml/tests/test_sft.py`), including adversarial cases for both leakage
  modes. **ml suite now 104/104, server 9/9.**
- Results: `docs/results/decomp_sft.md`

### Task 5.3 — LoRA Fine-Tune *(Navya)* — Sync Point 2 deliverable
Fine-tuned a decomposition model with PEFT/LoRA on the 560-row SFT set from Task 5.2.

- **Adapter:** `ml/models/decomposer-lora` (3.6 MB, committed to the repo rather than left
  in gitignored `data/`, so Kajal can pick it up directly for the eval harness).
- **Both done-conditions pass:**

  | Requirement | Result | Status |
  |---|---|---|
  | Schema-valid JSON on >98% of val inputs | **100.0%** (70/70) | PASS |
  | Beats zero-shot control on component F1 | **0.1248** vs 0.0000 | PASS |

- **Base model — the roadmap's 7B is impossible here, and the first substitute was worse
  than impossible.** bitsandbytes 4-bit is CUDA-only, so 7B needs ~14 GB at bf16 against
  ~9 GB free. I first chose `flan-t5-base`, which was wrong for a reason worth recording:
  **T5 cannot emit JSON at all** — `{` and `}` are absent from its SentencePiece vocabulary
  (both map to `<unk>`), because T5's C4 preprocessing stripped curly braces. Every SFT
  target was silently tokenized with `<unk>` where braces belong. A 30-minute overfit run
  confirmed 0% schema validity as a *hard ceiling*, not undertraining. There is now a
  regression test asserting the base tokenizer round-trips `{`/`}`.
- **Final base: `HuggingFaceTB/SmolLM2-135M-Instruct`**, LoRA r=16 on `q_proj`/`v_proj` —
  **0.92M trainable / 135.4M (0.68%)**. Lighter *and* faster than the flan-t5-base it
  replaced (4.3 vs 7.2 s/example), byte-level BPE so braces round-trip, instruction-tuned,
  and decoder-only (which is what `outlines` targets, de-risking 5.4).
- **Training:** 5 epochs, batch 4 × grad-accum 4, lr 1e-3 linear decay, **135 min on CPU**.
  Val loss **2.18 → 1.32**, still falling at epoch 5 and never turning up, so the run was
  budget-limited rather than overfitting. Best-val checkpointing throughout.
- **The honest reading, which the report states rather than stopping at "PASS":** the model
  learned the **output contract**, not the **mapping**. 100% schema validity is real and
  useful. But headline macro F1 is 0.1248, and the per-field pattern shows majority-class
  collapse: `style` F1 0.000 with 87% correct abstention, `tone` F1 0.000 with 93%
  abstention. `style` is absent from 76% of targets and `tone` from 87%, so predicting
  `null` is a strong strategy that earns nothing in F1. Best field is `lighting` (0.286 —
  small closed vocabulary); `subject` is 0.196.
- **Both controls score 0.0 F1, and that is a limitation, not a triumph.** I added a
  few-shot control (same weights, no training, shown two format demonstrations)
  specifically to separate *format learning* from *task learning*. It scored 0.0% valid —
  worse than the strict zero-shot control's 2.9%, because a longer prompt gives a 135M
  model more to lose track of. So at this scale the two cannot be separated: the fine-tune
  is what makes the model able to attempt the task at all. That belongs in the write-up as
  a stated limitation of the comparison.
- **Component F1 metric implemented** (`ml/app/component_eval.py`) — the piece Kajal's Task
  4.3 harness explicitly left pending on my Task 1.3 labels. Standalone and importable, so
  4.3/8.1 can call it directly. Token-level F1 for string fields (exact match is too
  brittle for this corpus), set overlap for list fields, and explicit null handling so
  "correctly saying nothing" is tracked separately from precision.
- **Recommended next steps before 8.1**, cheapest first: re-weight the loss away from
  `null` (addresses the clearest failure mode without more data or compute); step up to
  SmolLM2-360M (~6 h — both controls at 0.0 suggest capacity really is binding); raise LoRA
  rank above r=16.
- Tests: 36 added (18 `test_component_eval.py`, 18 `test_decompose.py`). **ml suite now
  140/140.**
- Results: `docs/results/train_decomposer.md`, `docs/results/decomposer_eval.md`

### Task 5.4 — Constrained Decoding + Fallback *(picked up by Kajal)*
Guarantees the serving pipeline never receives an unparseable decomposition, without touching how `decompose_batch` is scored for the eval harness.

- **Scope decision:** Task 5.3's own write-up flagged `outlines` (grammar-constrained decoding) as the intended approach. Given this machine's tight memory budget (Docker's WSL2 VM is capped at 3.7GB out of 7.6GB total RAM — discovered directly while testing this task, see below), adding a new dependency that wraps the model with its own generation machinery was judged a real risk for a modest gain: `decompose_batch` already measures 100% schema validity on val. Implemented instead as a **bounded retry + deterministic fallback** — the standard, dependency-free version of the same guarantee.
- `decompose_reliably()` (`ml/app/decompose.py`) re-prompts only the rows that failed to parse (not the whole batch), up to one retry, then falls back to a schema-valid result built from the evidence block's caption line, so a row is never dropped.
- `decompose_batch` itself is untouched, since Task 5.3's own eval measures its *raw* schema-validity rate as a done-condition metric — adding a safety net there would hide what that metric is designed to catch.
- Tests: `ml/tests/test_decompose_reliable.py` (4 tests, stubbed — no model needed to verify the retry/fallback control flow).

### Task 5.5 — Wire Decomposition Into the Pipeline *(picked up by Kajal)*
Replaced the Task 3.1 placeholder (top-1 candidate's fields) with the real model.

- New `POST /internal/decompose` (`ml/app/routes/decompose.py`): takes a caption + retrieved prompts + optional PEZ prompt, renders the same evidence-block format Task 5.2 trains on, and returns guaranteed-valid `StructuredFields` via `decompose_reliably`.
- `server/src/routes/reconstruct.js` now calls `/internal/retrieve` → `/internal/caption` → `/internal/decompose` in sequence per request, and persists real `captioning_ms`/`decomposition_ms` (previously always `null`).
- **Graceful degradation, not a hard dependency:** if captioning or decomposition fails or times out, the response still returns 200 with the retrieval-only placeholder fields, `status: "degraded"` (an enum value the frozen schema already reserved for exactly this) instead of failing the whole request — retrieval already succeeded by that point.
- **Found and fixed along the way:** the trained adapter's default load path (`data/models/decomposer-lora`, gitignored) didn't match where it's actually committed (`ml/models/decomposer-lora`) — anyone who hadn't manually copied it would hit a silent `FileNotFoundError`. Fixed by setting `DECOMPOSER_ADAPTER` in `docker-compose.yml` to the committed path, the single place this should be configured.
- **Also found:** Docker Desktop's WSL2 memory limit on this machine defaults to 3.7GB (half of the laptop's 7.6GB total) — running BLIP-large + the decomposer's base model together came close to that ceiling and caused one transient multi-minute stall during testing. Not hit again after that; noted here since it's a real hardware ceiling for the report, not a code bug, and worth knowing if `/api/reconstruct` seems unexpectedly slow later.
- Verified end-to-end: full request (retrieval + caption + decompose) completes in **~31s**, `status: "completed"`, `structured_fields.subject` populated from the real model. Server suite 9/9, ml suite 146/146.

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
| Weak-label LLM seed set | 5–10K LLM-labeled prompts | none (hook built, unused) | Seed set would exceed the 700-prompt corpus; no LLM credentials in this env |
| Captioning model | BLIP-2, or LLaVA-1.5-7B if VRAM allows | BLIP-1 large (~470M) | BLIP-2 needs ~7.5GB bf16 / ~15GB fp32 vs ~6.3GB available, CPU-only; env-swappable, no code change needed on a GPU box |
| Decomposer base model | Mistral-7B-Instruct / Llama-3-8B, 4-bit | SmolLM2-135M-Instruct + LoRA r=16 | bitsandbytes 4-bit is CUDA-only, 7B needs ~14GB bf16 vs ~9GB free; flan-t5-base tried first but T5 cannot emit `{`/`}` at all |
| Constrained decoding (5.4) | `outlines` grammar-constrained generation | Bounded retry + deterministic fallback | Docker's WSL2 VM is capped at 3.7GB on this 7.6GB-RAM laptop; `decompose_batch` already measures 100% schema validity on val, so a new dependency was judged not worth the memory/complexity risk for the remaining gain |

## 7. What's Next

- **Kajal:** Tasks 5.4 and 5.5 are now done too (picked up from Navya's side by agreement, see above) — the real decomposer is live in `/api/reconstruct`, replacing the Task 3.1 placeholder. Nothing required until Navya's next delivery, except replying to §9's data-provenance question (done, see the reply there) and, for 8.1, remembering that component F1 is currently **0.1248** and `negative_constraints` should stay out of the headline macro at low support.
- **Navya:** Tasks 1.3, 5.1, 5.2, 5.3 done, and 5.4/5.5 no longer blocking anything on your side. **Next up: 6.1–6.4 (confidence/calibration)** — the biggest remaining piece — then 7.2 (text pipeline) and 8.3 (failure analysis).
- **Sync Point 2 (Model Handoff):** reached — adapter delivered, scored through the eval harness, and wired into the live pipeline.

## 8. Key Files Reference

| What | Where |
|---|---|
| Sync 1 technical handoff | `docs/sync1-handover.md` |
| Weak-label quality + audit | `docs/label-quality.md` |
| Captioning results + model rationale | `docs/results/captioning.md` |
| Decomposition SFT dataset | `docs/results/decomp_sft.md` |
| Decomposer training curve | `docs/results/train_decomposer.md` |
| Decomposer evaluation + limitations | `docs/results/decomposer_eval.md` |
| Trained LoRA adapter (Sync Point 2) | `ml/models/decomposer-lora/` |
| API contract (frozen) | `docs/api-contract.md` |
| PEZ baseline results | `docs/results/pez_baseline.md` |
| CLIP-tag baseline results | `docs/results/cliptag_baseline.md` |
| Combined evaluation | `docs/results/baselines.md`, `.csv` |
| All `make` commands | `Makefile` (targets: `ingest`, `split`, `bench-embed`, `build-index`, `seed`, `pez-baseline`, `cliptag-baseline`, `eval`, `ingest-alpaca`, `split-alpaca`, `weak-label`, `audit-labels`, `caption`, `decomp-sft`, `pez-sft`, `train-decomposer`, `eval-decomposer`) |

---

## 9. Open Item — Data Provenance Between Machines *(added by Navya, 2026-09-08)*

`data/` is gitignored, so per §6 of `docs/sync1-handover.md` I rebuilt it on my machine by
re-running Kajal's pipeline rather than copying her folder: `make ingest` → `split` →
`build-index` → `seed`. That reproduced the headline shape exactly — **700 pairs,
560/70/70 splits, prompt-disjointness verified at cosine 0.95, self-retrieval smoke test
passing at rank 1** — and everything downstream of it is internally consistent
(`pairs.parquet` ids ↔ `id_map.json` ↔ the 560 Mongo `_id`s, all verified).

One number did not match, and it's worth a moment of Kajal's time:

| Metric | Kajal's report (§5, Task 1.2) | My re-run |
|---|---|---|
| Near-duplicate prompt clusters | 54 (129 pairs) | **62 (155 pairs)** |

The split script is deterministic given the same input, so a different cluster count
suggests the **ingested 700 pairs may not be byte-identical between our two machines** —
plausibly a different `poloclub/diffusiondb` revision, or a filter tweak between her run
and the version now in the repo. (The status report is also internally inconsistent here:
Task 1.2 says 54 clusters / 129 pairs, while the Task 1.4 note cites "110 on the image
side.")

**Why it matters:** nothing is broken today, because each machine's index, Mongo
collection and labels are built from the same local corpus. The risk is only at handoff —
if we ever exchange artifacts keyed by dataset id (my `pairs_labeled.parquet`, her
`index.faiss` / PEZ result cache), an id could resolve to a *different* prompt on the other
side, and it would fail silently rather than loudly.

**Suggested resolution, cheapest first:**
1. Compare a checksum of the prompt column — e.g. `sha256` over the `id,prompt` pairs
   sorted by id — and confirm we're on the same corpus.
2. If they differ, pick one machine's `data/` as canonical for all shared artifacts and
   pin the HF dataset revision in `scripts/ingest_diffusiondb.py`.
3. Either way, this belongs in `docs/REPRODUCE.md` for Task 8.5, since the reproducibility
   pack claims a clean clone reproduces the headline table from scratch.

**Reply *(Kajal, 2026-09-11):*** Ran suggestion #1 on my machine's `data/diffusiondb/pairs.parquet`
(the same file the original 700-pair Task 1.1 numbers came from):

```
rows: 700
sha256: ee200e2144937ce67cd043e00882d0dd8c2864b2aa5974e27c80335a175f002c
```

Can you run the same command on your machine (`docker compose run --rm ml python -c "..."`, same
snippet: sort rows by id, hash `id,prompt` per row) and paste your hash here? If they differ, that
confirms the corpora genuinely diverged (most likely candidate: `poloclub/diffusiondb`'s
`2m_random_5k` config wasn't pinned to a revision in `scripts/ingest_diffusiondb.py`, so a
re-upload or a `datasets` library version difference between our two installs could change
row order/content even for a nominally-fixed predefined split). Also fixed the "110" mixup this
made me notice — see the corrected line under Task 1.4 above. Agree with your #2/#3: once we
confirm a mismatch, I'll pin a `revision=` on the `load_dataset` call and we pick one machine's
`data/` as canonical before Task 8.5.
