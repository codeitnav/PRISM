# Decomposer Evaluation - Task 5.3

- **Base model:** `Qwen/Qwen2.5-VL-3B-Instruct` + LoRA adapter (`decomposer-lora`)
- **Split:** val (322 rows, prompt-disjoint from train)

## Done-condition

| Requirement | Result | Status |
|---|---|---|
| Schema-valid JSON on >98% of val inputs | 322/322 = **100.0%** | PASS |
| Beats zero-shot control on component F1 | **0.2089** vs 0.0617 | PASS |

## Comparison

| Variant | Schema-valid | Headline macro F1 | Full macro F1 | s/row |
|---|---|---|---|---|
| LoRA fine-tuned | 100.0% | **0.2089** | 0.1791 | 0.77 |
| Zero-shot control | 99.4% | **0.0617** | 0.0529 | 0.72 |
| Few-shot control (format shown, task untrained) | 100.0% | **0.0846** | 0.0725 | 0.78 |

### On the controls

The few-shot control (same weights, no training, shown two format demonstrations) separates **format learning** from **task learning**. The zero-shot control emits schema-valid JSON on 99.4% of rows (F1 0.0617) and the few-shot control on 100.0% (F1 0.0846). Any F1 the fine-tune earns above the few-shot score is attributable to learning the task rather than to learning to emit JSON: **0.2089** for the fine-tune versus **0.0846** few-shot. The per-field breakdown below shows how well each field was learned.

## Per-field breakdown

### LoRA fine-tuned

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.263 | 0.229 | **0.245** | 0.022 | 0.000 | 322 |
| `style` | 0.275 | 0.188 | **0.224** | 0.164 | 0.840 | 134 |
| `medium` | 0.303 | 0.256 | **0.277** | 0.221 | 0.743 | 131 |
| `lighting` | 0.200 | 0.120 | **0.150** | 0.103 | 0.913 | 58 |
| `tone` | 0.163 | 0.125 | **0.142** | 0.118 | 0.919 | 51 |
| `modifiers` | 0.230 | 0.203 | **0.216** | 0.111 | 0.649 | 171 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 2 |

**Headline macro F1** (excluding `negative_constraints`): **0.2089**
Full macro F1 (all 7 fields): 0.1791
Rows scored: 322

### Zero-shot control

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.157 | 0.087 | **0.112** | 0.000 | 0.000 | 322 |
| `style` | 0.049 | 0.119 | **0.069** | 0.104 | 0.032 | 134 |
| `medium` | 0.091 | 0.160 | **0.116** | 0.107 | 0.026 | 131 |
| `lighting` | 0.021 | 0.102 | **0.035** | 0.052 | 0.023 | 58 |
| `tone` | 0.007 | 0.047 | **0.013** | 0.039 | 0.018 | 51 |
| `modifiers` | 0.039 | 0.018 | **0.025** | 0.000 | 0.629 | 171 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 2 |

**Headline macro F1** (excluding `negative_constraints`): **0.0617**
Full macro F1 (all 7 fields): 0.0529
Rows scored: 322

### Few-shot control (format shown, task untrained)

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.159 | 0.164 | **0.161** | 0.000 | 0.000 | 322 |
| `style` | 0.047 | 0.089 | **0.062** | 0.067 | 0.287 | 134 |
| `medium` | 0.100 | 0.183 | **0.129** | 0.107 | 0.115 | 131 |
| `lighting` | 0.039 | 0.176 | **0.063** | 0.086 | 0.170 | 58 |
| `tone` | 0.014 | 0.062 | **0.023** | 0.059 | 0.170 | 51 |
| `modifiers` | 0.073 | 0.065 | **0.069** | 0.000 | 0.225 | 171 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 2 |

**Headline macro F1** (excluding `negative_constraints`): **0.0846**
Full macro F1 (all 7 fields): 0.0725
Rows scored: 322

> Scores are agreement with the **weak labels** from Task 1.3, not with ground truth. See `docs/label-quality.md` for those labels' measured per-field precision (`modifiers` in particular is noisy at 0.26, which is why the SFT targets keep only lexicon-recognised entries). `negative_constraints` is populated on 4 of 700 rows, so it is excluded from the headline macro and reported separately.

## Sample predictions

**`000002`** — true prompt: `cyberpunk glasses, cyberpunk, futuristic`

- predicted: `{"subject": "portrait of synthwave 8 0 s handsome confident max headroom wearing vr glasses", "style": "synthwave", "medium": "digital art", "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"futuristic","style":"cyberpunk glasses","medium":null,"lighting":null,"modifiers":["cyberpunk"],"tone":null,"negative_constraints":[]}`

**`000009`** — true prompt: `tin toys wooden cyborg by alphonse mucha`

- predicted: `{"subject": "a humanoid creature on other planets that appear intelligent is talking about life", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"tin toys wooden cyborg by alphonse mucha","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000025`** — true prompt: `made in abyss cutscene`

- predicted: `{"subject": "bruce campbell in made in abyss", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"made in abyss cutscene","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000026`** — true prompt: `mr. and mrs. dursley of number four, privet drive, were proud to say that they were perfectly normal, thank you very muc`

- predicted: `{"subject": "oliver hardy and stan laurel in game of thrones", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"mr","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000027`** — true prompt: `bruce campbell is looking like 😑, papyrus`

- predicted: `{"subject": "Nicolas Cage as a Mafia boss from 1960s", "style": "concept art", "medium": null, "lighting": null, "modifiers": ["highly detailed", "artstation", "smooth", "sharp focus", "illustration"], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"bruce campbell is looking like 😑","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000030`** — true prompt: `constructivism art table made of rectangle tetris, octane render, trending on artstation`

- predicted: `{"subject": "a modern table leg made of rectangle tetris", "style": null, "medium": "octane render", "lighting": null, "modifiers": ["trending on artstation"], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"constructivism art table made of rectangle tetris","style":null,"medium":"octane render","lighting":null,"modifiers":["trending on artstation"],"tone":null,"negative_constraints":[]}`

**`000040`** — true prompt: `$ ki mask slump god`

- predicted: `{"subject": "kanye west wearing a mask", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"$ ki mask slump god","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000043`** — true prompt: `the slump god, rapper, character portrait, standing at attention in military garb, 1 8 th century, american romanticism,`

- predicted: `{"subject": "portrait of george washington wearing a military uniform", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"the slump god","style":"american romanticism","medium":"oil on canvas","lighting":null,"modifiers":["character portrait","oil on canvas by william sidney mount","trending on artstation"],"tone":null,"negative_constraints":[]}`
