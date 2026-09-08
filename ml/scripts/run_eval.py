#!/usr/bin/env python3
"""Task 4.3 - run the evaluation harness on both baselines.

Scores PEZ (Task 4.1) and the naive CLIP-tag baseline (Task 4.2) through
the same harness (app.eval.run_eval), reusing their already-computed
results (data/results/{pez,cliptag}_baseline.json) instead of re-running
PEZ's ~20+ minute optimization - see app.eval's module docstring for the
"reconstructor" interface this relies on.

Output:
    data/results/baselines.csv - one row per (reconstructor, image)
    data/results/baselines.md  - summary + per-image detail
    (copy the .md into docs/results/ afterward - docs/ is mounted read-only
    in the ml container)

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.run_eval
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from app.eval import run_eval

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
TEST_SPLIT_PATH = DATA_DIR / "splits" / "test.json"
RESULTS_DIR = DATA_DIR / "results"

NUM_TEST_IMAGES = 20


@dataclass
class _Item:
    prompt: str
    cosine_similarity: float


@dataclass
class _BatchResult:
    items: list[_Item]
    wall_clock_seconds: float


def _load_precomputed(path: Path):
    """Wraps an already-computed baseline's JSON as a "reconstructor"
    callable matching app.eval's expected interface - ignores whatever
    embeddings it's called with and just returns the cached results.
    """
    with open(path) as f:
        data = json.load(f)
    items_by_id = {item["id"]: _Item(item["prompt"], item["cosine_similarity"]) for item in data["items"]}
    wall_clock = data["wall_clock_seconds"]

    def _reconstructor(_target_embeddings):
        return _BatchResult(items=list(items_by_id.values()), wall_clock_seconds=wall_clock)

    return _reconstructor, items_by_id


def main() -> None:
    with open(TEST_SPLIT_PATH) as f:
        test_ids = json.load(f)[:NUM_TEST_IMAGES]
    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    rows = [rows_by_id[i] for i in test_ids]
    original_prompts = [r["prompt"] for r in rows]
    image_ids = [r["id"] for r in rows]

    pez_fn, pez_items = _load_precomputed(RESULTS_DIR / "pez_baseline.json")
    cliptag_fn, cliptag_items = _load_precomputed(RESULTS_DIR / "cliptag_baseline.json")
    assert list(pez_items) == image_ids, "pez_baseline.json image order doesn't match test split"
    assert list(cliptag_items) == image_ids, "cliptag_baseline.json image order doesn't match test split"

    print("Scoring PEZ and CLIP-tag through the same harness (computing BERTScore)...")
    eval_rows = run_eval(
        {"pez": pez_fn, "clip_tag": cliptag_fn},
        image_ids=image_ids,
        original_prompts=original_prompts,
        target_embeddings=None,  # unused - both reconstructors return precomputed results
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baselines.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(vars(eval_rows[0])))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(vars(row))
    print(f"Saved -> {csv_path}")

    by_name: dict[str, list] = {}
    for row in eval_rows:
        by_name.setdefault(row.reconstructor, []).append(row)

    md_path = RESULTS_DIR / "baselines.md"
    with open(md_path, "w") as f:
        f.write("# Baseline Evaluation (Task 4.3)\n\n")
        f.write(
            f"Both baselines (PEZ, Task 4.1; naive CLIP-tag, Task 4.2) scored through the same harness "
            f"(`ml/app/eval.py`) on the same {len(rows)} test-split images.\n\n"
        )
        f.write(
            "**Not included:** component-wise precision/recall/F1 against weak structured labels - "
            "Task 1.3 (weak labeling) hasn't landed yet. Extend this harness once it does.\n\n"
        )
        f.write("## Summary\n\n")
        f.write("| Reconstructor | Mean CLIP-score | Mean BERTScore F1 | Mean latency (s/image) |\n")
        f.write("|---|---|---|---|\n")
        for name, group in by_name.items():
            n = len(group)
            mean_clip = sum(r.clip_score for r in group) / n
            mean_bert_f1 = sum(r.bertscore_f1 for r in group) / n
            mean_latency = sum(r.latency_seconds for r in group) / n
            print(f"{name}: mean CLIP-score={mean_clip:.4f}, mean BERTScore F1={mean_bert_f1:.4f}")
            f.write(f"| {name} | {mean_clip:.4f} | {mean_bert_f1:.4f} | {mean_latency:.2f} |\n")

        f.write("\n## Per-image detail\n\n")
        f.write("| Reconstructor | Image ID | CLIP-score | BERTScore F1 | Generated prompt |\n")
        f.write("|---|---|---|---|---|\n")
        for row in eval_rows:
            prompt = row.generated_prompt.replace("|", "/")
            f.write(
                f"| {row.reconstructor} | {row.image_id} | {row.clip_score:.4f} | "
                f"{row.bertscore_f1:.4f} | {prompt} |\n"
            )

    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
