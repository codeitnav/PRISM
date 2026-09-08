# Label Quality — Task 1.3 Weak Structured Labels

**Owner:** Navya (Person B — Modeling & Intelligence)
**Date:** 2026-09-08
**Artifacts:** `ml/app/weak_label.py`, `ml/scripts/run_weak_label.py`, `ml/scripts/audit_weak_labels.py`
**Data:** `data/diffusiondb/pairs_labeled.parquet` (700 rows), `data/diffusiondb/label_audit.jsonl` (100 audited samples)

Task 1.3's done-condition is: `pairs_labeled.parquet` exists, and a manual audit of 100
random samples reports field-level precision, logged here.

---

## 1. What the labeler does

`app.weak_label.weak_label(prompt) -> StructuredFields` runs a rule + lexicon pass:

1. **Segment.** Split on the separators DiffusionDB prompts actually use — commas,
   semicolons, sentence periods, and the `|` / `::` weighting syntax that leaks in from
   Midjourney and AUTOMATIC1111 users. Periods are *not* split after a single-character
   token, which keeps decimal weights (`f 1. 8`) and artist initials
   (`j. c. leyendecker`) intact.
2. **Clean.** Strip attention-weight syntax (`(masterpiece:1.4)`), brackets, `--flags`,
   and credit boilerplate (`by …`, `in the style of …`).
3. **Classify.** Match each segment against five curated lexicons (style, medium,
   lighting, tone, modifiers), longest phrase first.
4. **Assign, in two passes.** Pass 1 lets short, lexicon-dominated segments claim the
   single-slot fields, because an explicitly-tagged segment is a stronger signal than a
   phrase merely mentioned inside the subject. Pass 2 backfills still-empty slots from
   phrases found inside the long content-bearing segments, so
   *"hyperrealistic photograph of an astronaut"* still yields `style=hyperrealistic`.
5. **Negatives.** Pulled from explicit negative syntax — a leading `no `/`without `, or
   `--neg` / `negative prompt:` markers.

### Corpus-frequency mining fed the lexicons

The roadmap asks for vocabularies "mined from corpus frequency." `mine_lexicon_candidates()`
surfaces the most frequent segments no lexicon classifies; running it on the ingested 700
prompts drove two concrete fixes:

- **Detokenized spacing.** These DiffusionDB configs store prompts detokenized, inserting
  spaces inside short tokens: `3 d render`, `4 k`, `8 k`, `5 0 mm`, `art station`. This was
  the single largest source of missed hits (~230 across 700 prompts). Matching now runs
  against a re-joined form via `normalize_for_match()`; the stored label keeps the original
  spelling. Effect on coverage: **medium 37.9% → 50.3%**, style 31.4% → 34.7%,
  lighting 25.1% → 27.7%, tone 15.3% → 19.0%.
- **New vocabulary.** Genuinely frequent terms were promoted into the lexicons
  (`dynamic lighting`, `radiant light`, `hdr`, `illustration`, `oil pastels`,
  `textured canvas`, `pixiv`, `pbr`, `fine details`, `perfect symmetry`, …), each marked in
  the source.

After both passes the residual unclassified segments are dominated by **artist names**
(`artgerm`, `wlop`, `ilya kuvshinov`, `greg rutkowski`, …) and genuinely scene-specific
content (`polygonal wooden walls`) — i.e. content, not missing vocabulary. That is the
healthy stopping point for a lexicon pass.

---

## 2. Coverage and precision (the audited numbers)

Field-level **precision** is measured over fields the labeler actually populated:
`correct / (correct + wrong)`. A field left null is a recall miss, not a precision error,
so it is reported separately as **coverage** — the share of the 100 samples where the
labeler emitted any value.

| Field | Coverage (of 100) | Judged | Correct | Precision |
|---|---|---|---|---|
| `subject` | 100 | 100 | 87 | **0.87** |
| `style` | 40 | 40 | 37 | **0.93** |
| `medium` | 50 | 50 | 50 | **1.00** |
| `lighting` | 27 | 27 | 25 | **0.93** |
| `tone` | 16 | 16 | 12 | **0.75** |
| `modifiers` | 80 | 80 | 21 | **0.26** |
| `negative_constraints` | 1 | 1 | 1 | **1.00** |

**Overall field-level precision: 0.742** (233 correct of 314 judged field values).

Whole-corpus coverage (all 700 rows, not just the audited 100): style 34.4%, medium 50.3%,
lighting 27.7%, tone 18.7%, modifiers 76.3%, negative_constraints 0.6%.

### How the audit was conducted

- Sample: 100 rows drawn from `pairs_labeled.parquet` with a fixed seed (1337), so the
  sample is reproducible — `make audit-labels`.
- Judged by Claude Code acting as the annotator, reading each raw prompt against the
  emitted fields. This is an assistant judgment pass, **not** an independent human
  annotation, and it should be treated as such. Every per-field verdict is recorded in
  `data/diffusiondb/label_audit.jsonl`, so any of the 314 judgments can be re-checked or
  overridden and the table regenerated with
  `docker compose run --rm ml python -m scripts.audit_weak_labels --score`.
- Rubric used:
  - `subject` — correct if it names the primary subject without swallowing style/medium/
    quality tags and without losing the real subject to an artist credit.
  - `style` / `medium` / `lighting` / `tone` — correct if the value genuinely belongs to
    that category and appears in the prompt.
  - `modifiers` — correct only if *every* entry is a legitimate free-form quality/detail
    modifier (artist and platform credits allowed, since the frozen schema has no field
    for them); wrong if any entry is scene content that belongs in `subject`, or a value
    that belongs in a still-empty typed slot.
  - `negative_constraints` — correct if the entries are genuinely negative.

---

## 3. What the numbers mean

**The typed single-slot fields are strong.** `medium` 1.00, `style` 0.93, `lighting` 0.93
— when the labeler commits to one of these, it is almost always right. That is what makes
these usable as SFT targets for Task 5.3. The trade-off is recall, not precision: `medium`
fires on only half the corpus, `tone` on under a fifth.

**`modifiers` at 0.26 is the real weakness, and it is structural.** `modifiers` is the
default sink for any segment the lexicons don't recognise, so it absorbs three different
kinds of content:

1. scene detail that belongs in `subject` (`lava streams`, `glass ceilings`,
   `wearing a tanktop`),
2. values that belong in a typed slot already filled by an earlier segment — the
   single-slot schema means a second style tag (`film noir` after `dark fantasy`) has
   nowhere to go,
3. legitimate quality modifiers and artist credits.

Only (3) is correct by the rubric. **Implication for Task 5.2:** do not train the
decomposer to reproduce `modifiers` verbatim — it would learn to imitate a junk drawer.
Options, in the order I'd try them: weight the component loss towards the typed fields;
or filter `modifiers` to lexicon-matched entries only for the SFT target, keeping the
unmatched remainder out of the supervision signal.

**`subject` at 0.87** — the 13 failures are almost all one mode: the prompt opens with an
artist credit or a bare medium phrase, so the first segment isn't the subject
(`a paint of dan mumford`, `alyssa monks`, `a gorgeous painting by barlowe …`). A
"skip leading credit/medium-only segments when choosing the subject" rule would recover
most of these; logged as a follow-up rather than fixed now, since 0.87 is already
serviceable and the decomposer sees the caption and retrieved prompts too.

**`tone` at 0.75** is the weakest typed field. Failures are colour/camera terms leaking
into a mood slot (`warm colors`, `dramatic angle`) and adjectives that describe an object
rather than the mood (`dark` from *"dark night sky"*, `cold` from *"a cold, dried out
lake"*).

**`negative_constraints` is effectively empty — 4 of 700 rows (0.6%).** DiffusionDB's
`2m_random_*` prompts predate widespread negative-prompt use and don't carry a separate
negative field. Downstream consequences: the decomposer will learn to emit `[]` here
almost always, and this component should be **excluded from the component-F1 headline**
in Task 4.3 / 8.1 rather than counted as a solved field. Worth raising at Sync Point 3.

---

## 4. Scope deviations from the roadmap

- **No LLM-labeled seed set.** The roadmap pairs the rule pass with 5–10K LLM-labeled
  prompts for ambiguous cases. Two things make that inapplicable as written: the ingested
  corpus is 700 prompts total (see the scope note in `scripts/ingest_diffusiondb.py`), so a
  5–10K seed set is larger than the dataset; and this environment has no LLM API
  credentials configured. The hook is built rather than faked —
  `load_llm_seed_labels()` reads `data/diffusiondb/llm_seed_labels.jsonl`
  (`{"id": …, "structured_fields": {…}}` per line) and those labels override the rule
  output per prompt, with provenance recorded in the `label_provenance` column. Current
  run: 700/700 `rule`, 0 `llm_seed`. **All precision numbers above are rule-pass-only.**
- **Audit is an assistant judgment pass, not independent human annotation** — see §2.

---

## 5. Reproducing

```bash
make weak-label     # -> data/diffusiondb/pairs_labeled.parquet + Mongo structured_fields
make audit-labels   # -> data/diffusiondb/label_audit.jsonl (100-sample worksheet)
# fill in verdicts, then:
docker compose run --rm ml python -m scripts.audit_weak_labels --score
cp data/results/label-quality.md docs/label-quality.md   # table only; this file adds the analysis
```

`ml/tests/test_weak_label.py` covers the labeler (23 tests): segment cleaning, negative
extraction, Midjourney separators, the detokenization normalizer, period-splitting with
its decimal and initials guards, two-pass slot precedence, schema validity, and the
LLM-seed override path.
