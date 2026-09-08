"""Task 4.3 - evaluation harness.

Scores any "reconstructor" callable - anything shaped like
app.baselines.pez.reconstruct_batch or app.baselines.clip_tag.classify_batch
(takes a (B, D) batch of image embeddings, returns an object with `.items`
(each having `.prompt` and `.cosine_similarity`) and `.wall_clock_seconds`) -
against the same test-split images, and reports:
  - CLIP-score (cosine similarity to the original image - already computed
    by each reconstructor, re-reported here for one unified table)
  - BERTScore precision/recall/F1 against the ground-truth original prompt
  - latency (seconds/image)

This is also the harness Task 5.x's real decomposition model should be
scored through later - anything with the same `.items`/`.wall_clock_seconds`
shape works as a "reconstructor" here, live or precomputed.

Component-wise precision/recall/F1 against weak structured labels is NOT
computed here - Task 1.3 (Navya's weak-labeling pass) hasn't landed yet.
Extend this harness once it does, rather than building a second one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import bert_score

# distilbert-base-uncased, not the library's much larger roberta-large
# default - this machine is CPU-only and storage-constrained (same reasoning
# as the model choices in app.embed). bert_score has known-good layer
# defaults baked in for this model, so no extra tuning needed.
BERTSCORE_MODEL = "distilbert-base-uncased"


@dataclass
class EvalRow:
    reconstructor: str
    image_id: str
    original_prompt: str
    generated_prompt: str
    clip_score: float
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    latency_seconds: float


def run_eval(
    reconstructors: dict[str, Callable],
    image_ids: list[str],
    original_prompts: list[str],
    target_embeddings,
) -> list[EvalRow]:
    rows: list[EvalRow] = []

    for name, reconstructor in reconstructors.items():
        result = reconstructor(target_embeddings)
        latency_per_item = result.wall_clock_seconds / len(result.items)
        generated_prompts = [item.prompt for item in result.items]

        precision, recall, f1 = bert_score.score(
            generated_prompts, original_prompts, model_type=BERTSCORE_MODEL, verbose=False
        )

        for i, item in enumerate(result.items):
            rows.append(
                EvalRow(
                    reconstructor=name,
                    image_id=image_ids[i],
                    original_prompt=original_prompts[i],
                    generated_prompt=item.prompt,
                    clip_score=item.cosine_similarity,
                    bertscore_precision=float(precision[i]),
                    bertscore_recall=float(recall[i]),
                    bertscore_f1=float(f1[i]),
                    latency_seconds=latency_per_item,
                )
            )

    return rows
