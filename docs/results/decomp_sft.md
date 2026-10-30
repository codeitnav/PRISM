# Decomposition SFT Dataset - Task 5.2

- **Rows:** 3220  (test: 322, train: 2576, val: 322)
- **Retrieved prompts per row:** top-5, mean actually present 5.00
- **Rows with no retrieved neighbours:** 0
- **Self-match dropped (leakage guard):** 2576 rows, of 2576 train rows (the FAISS index is the train split, so every train row retrieves itself at rank 1; val/test rows should not)
- **PEZ coverage:** 3220/3220 rows (100.0%) from ['pez_baseline.json', 'pez_sft_train_val_test.json']
- **Input length (chars):** min 675, max 2570, mean 1400
- **Top-1 retrieval similarity (after self-exclusion):** min 0.4594, mean 0.8252, max 0.9978
- **Retrieval margin (top1-top2):** mean 0.0243
- **Captions with boilerplate stripped:** 2574/3220 (80%)
- **True prompt verbatim inside a retrieved neighbour:** 54/3220 (1.7%) - not leakage, but an upper bound on what copying alone could achieve (see below)
- **Lexicon vocabulary:** 412 curated phrases
- **Schema validation:** 3220/3220 rows pass `app.sft.validate_row`

## Target field coverage

| Field | Rows with a value | Share |
|---|---|---|
| `subject` | 3220 | 100.0% |
| `style` | 1403 | 43.6% |
| `medium` | 1341 | 41.6% |
| `lighting` | 600 | 18.6% |
| `tone` | 624 | 19.4% |
| `modifiers` | 1935 | 60.1% |
| `negative_constraints` | 18 | 0.6% |

## Leakage guard vs. legitimate retrieval

The FAISS index is built from the 560 train images, so a train row retrieves **itself** at rank 1 with similarity ~1.0 - and that prompt is exactly what its target was derived from. Training on that would teach the model to copy `retrieved_prompts[0]` and ignore the caption entirely, and it would collapse at inference, where an unseen image is never in the index. Every row's own id is therefore dropped from its candidates, and `app.sft.validate_row` re-checks the invariant per row after serialization.

A *different* image whose prompt is near-identical to - or even verbatim contains - this one's is a separate matter, and is deliberately **kept**. That is legitimate retrieval success which also occurs at inference (the train split contains near-duplicate prompt clusters). Filtering it would make the training inputs systematically weaker than production. It affects 54/3220 rows (1.7%), recorded per row as `true_prompt_in_retrieved` - useful as a ceiling on the copy-only strategy when interpreting Task 5.3's component F1.

## Modifier filtering

Weak labels produced **14675** modifier entries across 3220 rows; **5935** survive lexicon filtering (**40.4%** kept)

The Task 1.3 audit measured `modifiers` at 0.26 field-level precision (`docs/label-quality.md`) - it is the sink for every segment the lexicons did not classify, so it absorbs scene detail belonging in `subject` alongside genuine quality modifiers. Training on it verbatim would teach the model to imitate a junk drawer, so targets keep only lexicon-recognised modifiers. The unfiltered list survives in each row as `modifiers_raw`, so Task 8.2 can ablate this choice.
