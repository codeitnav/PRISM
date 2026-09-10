# Decomposer Evaluation - Task 5.3

- **Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct` + LoRA adapter (`decomposer-lora`)
- **Split:** val (70 rows, prompt-disjoint from train)

## Done-condition

| Requirement | Result | Status |
|---|---|---|
| Schema-valid JSON on >98% of val inputs | 70/70 = **100.0%** | PASS |
| Beats zero-shot control on component F1 | **0.1248** vs 0.0000 | PASS |

## Comparison

| Variant | Schema-valid | Headline macro F1 | Full macro F1 | s/row |
|---|---|---|---|---|
| LoRA fine-tuned | 100.0% | **0.1248** | 0.1070 | 3.34 |
| Zero-shot control | 2.9% | **0.0000** | 0.0000 | 8.88 |
| Few-shot control (format shown, task untrained) | 0.0% | **0.0000** | 0.0000 | 11.02 |

### On the controls - and why neither is informative here

Both controls score **0.0 F1**, and that is a limitation of the comparison rather than a triumph. The strict zero-shot control emits schema-valid JSON on only 2.9% of rows; the few-shot control, shown two format demonstrations, manages 0.0% - *worse*, because the longer prompt gives a 135M model more to lose track of.

The few-shot control was added specifically to separate **format learning** from **task learning**: if an untrained model shown the format could score above zero, then the fine-tune's gain would be attributable to the task rather than to learning to emit JSON. It cannot. So at this model scale the two are inseparable, and the honest reading of **0.1248** vs **0.0000** is: *the fine-tune is what makes the model able to attempt the task at all*, not that it has learned the mapping well. The absolute F1 below is what says how well it learned it.

## Per-field breakdown

### LoRA fine-tuned

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.205 | 0.187 | **0.196** | 0.000 | 0.000 | 70 |
| `style` | 0.000 | 0.000 | **0.000** | 0.000 | 0.868 | 17 |
| `medium` | 0.182 | 0.164 | **0.172** | 0.172 | 0.780 | 29 |
| `lighting` | 0.229 | 0.381 | **0.286** | 0.273 | 0.831 | 11 |
| `tone` | 0.000 | 0.000 | **0.000** | 0.000 | 0.934 | 9 |
| `modifiers` | 0.102 | 0.089 | **0.095** | 0.081 | 0.818 | 37 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 0 |

**Headline macro F1** (excluding `negative_constraints`): **0.1248**
Full macro F1 (all 7 fields): 0.1070
Rows scored: 70

### Zero-shot control

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 70 |
| `style` | 0.000 | 0.000 | **0.000** | 0.000 | 0.962 | 17 |
| `medium` | 0.000 | 0.000 | **0.000** | 0.000 | 0.951 | 29 |
| `lighting` | 0.000 | 0.000 | **0.000** | 0.000 | 0.966 | 11 |
| `tone` | 0.000 | 0.000 | **0.000** | 0.000 | 0.967 | 9 |
| `modifiers` | 0.000 | 0.000 | **0.000** | 0.000 | 0.939 | 37 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 0.986 | 0 |

**Headline macro F1** (excluding `negative_constraints`): **0.0000**
Full macro F1 (all 7 fields): 0.0000
Rows scored: 70

### Few-shot control (format shown, task untrained)

| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |
|---|---|---|---|---|---|---|
| `subject` | 0.000 | 0.000 | **0.000** | 0.000 | 0.000 | 70 |
| `style` | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 17 |
| `medium` | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 29 |
| `lighting` | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 11 |
| `tone` | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 9 |
| `modifiers` | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 37 |
| `negative_constraints` *(low support)* | 0.000 | 0.000 | **0.000** | 0.000 | 1.000 | 0 |

**Headline macro F1** (excluding `negative_constraints`): **0.0000**
Full macro F1 (all 7 fields): 0.0000
Rows scored: 70

> Scores are agreement with the **weak labels** from Task 1.3, not with ground truth. See `docs/label-quality.md` for those labels' measured per-field precision (`modifiers` in particular is noisy at 0.26, which is why the SFT targets keep only lexicon-recognised entries). `negative_constraints` is populated on 4 of 700 rows, so it is excluded from the headline macro and reported separately.

## Interpreting these numbers honestly

Both stated done-conditions pass, but the absolute quality is weak and the report should
say so plainly rather than stopping at "PASS".

**What the model clearly learned: the output contract.** 100% schema validity on 70
held-out rows, up from 2.9% untrained. Every row parses, carries exactly the seven schema
keys, and satisfies `StructuredFields`. That is a real and useful result - it is what makes
Task 5.4's constrained decoding a safety net rather than a necessity, and what lets Task
5.5 wire this into the pipeline at all.

**What it largely did not learn: the mapping.** Headline macro F1 is 0.1248. The per-field
pattern shows exactly where it went:

| Field | F1 | Correct abstention | Reading |
|---|---|---|---|
| `lighting` | 0.286 | 0.831 | Best field - small closed vocabulary ("cinematic lighting", "volumetric light") |
| `subject` | 0.196 | — | Present on every row, so no abstention to fall back on; ~20% token overlap |
| `medium` | 0.172 | 0.780 | Exact-match 0.172, so when it commits it is sometimes exactly right |
| `modifiers` | 0.095 | 0.818 | Hardest - a long open-ended list |
| `style` | 0.000 | 0.868 | Never correct when present, but abstains correctly 87% of the time |
| `tone` | 0.000 | 0.934 | Same pattern, more extreme |

The high abstention rates next to near-zero F1 are the tell: the model learned the
**majority class**. `style` is absent from 76% of targets, `tone` from 87%, so predicting
`null` is a strong strategy that costs nothing in abstention terms and earns nothing in F1.
It learned the prior and the format, not the evidence-to-field mapping.

**Three reasons that is unsurprising, in order of likely impact:**

1. **560 training rows.** For fields whose vocabulary is open-ended and culturally specific
   ("octane render", "trending on artstation", artist names), a few hundred examples is very
   little signal per vocabulary item.
2. **135M parameters.** That both controls score 0.0 F1 - including the one *shown* the
   format - is direct evidence this model has no usable prior for the task. It has to learn
   everything from those 560 rows.
3. **A noisy target.** Scores are agreement with the Task 1.3 weak labels, whose own
   measured field-level precision is `subject` 0.87, `style` 0.93, `medium` 1.00,
   `lighting` 0.93, `tone` 0.75, `modifiers` 0.26 (`docs/label-quality.md`). Perfect
   agreement with a noisy target is neither achievable nor desirable, so there is a ceiling
   well below 1.0 - though nowhere near as low as 0.12.

**What this means for the project.** The honest headline is: *the fine-tune reliably
produces well-formed structured output, and does so 2.7x faster than the untrained model
(3.34 vs 8.88 s/row), but does not yet reconstruct fields accurately enough to claim it
beats PEZ on content.* The Task 8.1 benchmark should report this number as it stands, and
the Task 8.3 failure analysis has an obvious first question: how much of the gap is
majority-class collapse versus genuine inability.

**Highest-value next steps, cheapest first:**

- **Re-weight the loss away from `null`.** The majority-class collapse is the single
  clearest failure mode, and it is addressable without more data or a bigger model.
- **Step up to SmolLM2-360M** (~3x training cost, ~6 h). The controls scoring 0.0 suggest
  capacity is a real binding constraint, not just data.
- **Raise LoRA rank** from r=16. 0.92M trainable params on a 7-field structured task is
  frugal; val loss was still falling at epoch 5 (1.3185, never turned up), so the run was
  capacity- or budget-limited rather than overfitting.

## Sample predictions

**`000000`** — true prompt: `zen landcape brush ink`

- predicted: `{"subject": "zen infinity ink", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"zen landcape brush ink","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000006`** — true prompt: `6 5 mm first - person view of the back of some adult college students at their desks facing the front of the class with `

- predicted: `{"subject": "a futuristic fine lasers tracing", "style": "art", "medium": "oil on canvas", "lighting": null, "modifiers": ["art of a women in modern fashion", "color ink painting", "candy chip color", "octane hyperrealism photorealistic airbrush collage painting", "dark monochrome", "trending on artnet"], "tone": "dark", "negative_constraints": []}`
- target:    `{"subject":"6 5 mm first - person view of the back of some adult college students at their desks facing the front of the class with big heads takato yamamoto","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000012`** — true prompt: `a three year old boy with ginger hair wearing denim overalls chasing butterflies. clean elegant painting, beautiful deta`

- predicted: `{"subject": "realistic painting maine coon cat wearing witch hat sleeping near hearth in bakery dontuts cakes", "style": null, "medium": "cinematic minecraft", "lighting": null, "modifiers": ["highly detailed", "elegant", "fine details", "masterpiece", "trending on artstation"], "tone": "realistic", "negative_constraints": []}`
- target:    `{"subject":"a three year old boy with ginger hair wearing denim overalls chasing butterflies","style":null,"medium":null,"lighting":null,"modifiers":["clean elegant painting","beautiful detailed face"],"tone":null,"negative_constraints":[]}`

**`000013`** — true prompt: `symmetry ( badass ) cyberborg mark zuckerberg ultra detailed, intricate, anime, dynamic lighting, digital art, digital p`

- predicted: `{"subject": "a robot with glowing eyes", "style": "cyberpunk", "medium": "vector behance hd jesper ejsing", "lighting": "digital art", "modifiers": ["vintage retro scifi", "trending on artstation", "symmetry!!!"], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"symmetry badass cyberborg mark zuckerberg ultra detailed","style":"anime","medium":"digital art","lighting":"dynamic lighting","modifiers":["intricate","digital painting","art station","sharp focus","illustration"],"tone":null,"negative_constraints":[]}`

**`000029`** — true prompt: `a hypertorus shaped everything bagel`

- predicted: `{"subject": "dslr food photograph of pizza with gumball and peanut butter toppings", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"a hypertorus shaped everything bagel","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000031`** — true prompt: `queen of hearts suggestive full color portrait tarot card`

- predicted: `{"subject": "nice girl with a crown on her head", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"queen of hearts suggestive full color portrait tarot card","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000033`** — true prompt: `multidimensional fractal - c 2 0. 0`

- predicted: `{"subject": "zen infinity ink", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"multidimensional fractal - c 2 0. 0","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`

**`000036`** — true prompt: `barack obama having a boxing match against darth vader`

- predicted: `{"subject": "film still of darth vader in pulp fiction", "style": null, "medium": null, "lighting": null, "modifiers": [], "tone": null, "negative_constraints": []}`
- target:    `{"subject":"barack obama having a boxing match against darth vader","style":null,"medium":null,"lighting":null,"modifiers":[],"tone":null,"negative_constraints":[]}`
