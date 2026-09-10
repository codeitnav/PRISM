# Decomposition SFT Dataset - Task 5.2

- **Rows:** 700  (test: 70, train: 560, val: 70)
- **Retrieved prompts per row:** top-5, mean actually present 5.00
- **Rows with no retrieved neighbours:** 0
- **Self-match dropped (leakage guard):** 560 rows, of 560 train rows (the FAISS index is the train split, so every train row retrieves itself at rank 1; val/test rows should not)
- **PEZ coverage:** 700/700 rows (100.0%) from ['pez_sft_train_val_test.json']
- **Input length (chars):** min 584, max 2398, mean 1394
- **Top-1 retrieval similarity (after self-exclusion):** min 0.4730, mean 0.8034, max 0.9955
- **Retrieval margin (top1-top2):** mean 0.0311
- **Captions with boilerplate stripped:** 576/700 (82%)
- **True prompt verbatim inside a retrieved neighbour:** 16/700 (2.3%) - not leakage, but an upper bound on what copying alone could achieve (see below)
- **Lexicon vocabulary:** 412 curated phrases
- **Schema validation:** 700/700 rows pass `app.sft.validate_row`

## What each row looks like

`data/decomp_sft.jsonl`, one JSON object per line:

| Key | Purpose |
|---|---|
| `input_text` | the rendered prompt Task 5.3 trains on |
| `target_text` | the strict, compact, canonical-key-order JSON the model must emit |
| `input` | structured mirror (`caption`, `caption_raw`, `retrieved_ids`, `retrieved_prompts`, `retrieved_similarities`, `pez_prompt`) so 5.3/8.2 can re-render or ablate an input without rebuilding |
| `target` | structured mirror of `target_text` |
| `true_prompt` | **reference only, never an input** - lets the eval harness score against the real prompt and makes failures debuggable |
| `modifiers_raw` | unfiltered weak-label modifiers, for the 8.2 ablation |
| `self_match_dropped`, `true_prompt_in_retrieved` | provenance flags, see below |

The held-out eval split travels *with* the dataset: every row carries its own
`split` field rather than living in a separate file that can drift out of sync.

### Why the PEZ line is always present

The input renders `PEZ baseline: (unavailable)` rather than omitting the line when a
row has no PEZ output. Keeping the format fixed means a PEZ-less row is not a
structurally different prompt - the model learns that this evidence can be absent,
which is also the live serving case whenever PEZ is skipped for latency. Coverage is
currently 100%, but the dataset was first built at 0% (PEZ lives in gitignored
`data/`) and that path stays supported.

## Target field coverage

| Field | Rows with a value | Share |
|---|---|---|
| `subject` | 700 | 100.0% |
| `style` | 241 | 34.4% |
| `medium` | 352 | 50.3% |
| `lighting` | 194 | 27.7% |
| `tone` | 131 | 18.7% |
| `modifiers` | 447 | 63.9% |
| `negative_constraints` | 4 | 0.6% |

## Leakage guard vs. legitimate retrieval

The FAISS index is built from the 560 train images, so a train row retrieves **itself** at rank 1 with similarity ~1.0 - and that prompt is exactly what its target was derived from. Training on that would teach the model to copy `retrieved_prompts[0]` and ignore the caption entirely, and it would collapse at inference, where an unseen image is never in the index. Every row's own id is therefore dropped from its candidates, and `app.sft.validate_row` re-checks the invariant per row after serialization.

A *different* image whose prompt is near-identical to - or even verbatim contains - this one's is a separate matter, and is deliberately **kept**. That is legitimate retrieval success which also occurs at inference (the train split contains near-duplicate prompt clusters). Filtering it would make the training inputs systematically weaker than production. It affects 16/700 rows (2.3%), recorded per row as `true_prompt_in_retrieved` - useful as a ceiling on the copy-only strategy when interpreting Task 5.3's component F1.

## Modifier filtering

Weak labels produced **3063** modifier entries across 700 rows; **1459** survive lexicon filtering (**47.6%** kept)

The Task 1.3 audit measured `modifiers` at 0.26 field-level precision (`docs/label-quality.md`) - it is the sink for every segment the lexicons did not classify, so it absorbs scene detail belonging in `subject` alongside genuine quality modifiers. Training on it verbatim would teach the model to imitate a junk drawer, so targets keep only lexicon-recognised modifiers. The unfiltered list survives in each row as `modifiers_raw`, so Task 8.2 can ablate this choice.
