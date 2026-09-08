#!/usr/bin/env python3
"""Task 4.2 - naive CLIP-tag baseline evaluation.

Runs the CLIP-tag baseline (ml/app/baselines/clip_tag.py) on the same 20
test-split images used for the PEZ baseline (Task 4.1), so the two are
directly comparable, and logs the same CLIP-score metric (cosine similarity
between the generated prompt's text embedding and the original image).

Output:
    data/results/cliptag_baseline.md   - per-image results table
    data/results/cliptag_baseline.json - same data, machine-readable (so
                                         Task 4.3's eval harness can reuse
                                         these results instead of rerunning)
    (copy the .md into docs/results/ afterward - docs/ is mounted read-only
    in the ml container)

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.run_cliptag_baseline
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch

from app.baselines.clip_tag import VOCABULARY, classify_batch
from app.embed import embed_images

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
TEST_SPLIT_PATH = DATA_DIR / "splits" / "test.json"
RESULTS_MD_PATH = DATA_DIR / "results" / "cliptag_baseline.md"
RESULTS_JSON_PATH = DATA_DIR / "results" / "cliptag_baseline.json"

NUM_TEST_IMAGES = 20


def main() -> None:
    with open(TEST_SPLIT_PATH) as f:
        test_ids = json.load(f)[:NUM_TEST_IMAGES]

    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    rows = [rows_by_id[i] for i in test_ids]

    print(f"Embedding {len(rows)} test images (should hit cache from Task 4.1)...")
    image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in rows]
    target_embeddings = torch.from_numpy(embed_images(image_bytes))

    print(f"Classifying against {sum(len(v) for v in VOCABULARY.values())} tags across {len(VOCABULARY)} categories...")
    result = classify_batch(target_embeddings)

    print(f"\nDone in {result.wall_clock_seconds:.2f}s")
    print(f"Mean CLIP-score: {result.mean_cosine_similarity:.4f}\n")

    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MD_PATH, "w") as f:
        f.write("# Naive CLIP-tag Baseline Results (Task 4.2)\n\n")
        f.write(
            f"Ran on the same {len(rows)} test-split images as the PEZ baseline (Task 4.1), "
            f"classified against {sum(len(v) for v in VOCABULARY.values())} tags across "
            f"{', '.join(VOCABULARY)} categories.\n\n"
        )
        f.write(f"**Wall-clock time:** {result.wall_clock_seconds:.2f}s\n\n")
        f.write(f"**Mean CLIP-score (cosine similarity to original image):** {result.mean_cosine_similarity:.4f}\n\n")
        f.write("| Image ID | Original prompt (truncated) | CLIP-tag prompt | CLIP-score |\n")
        f.write("|---|---|---|---|\n")
        for row, item in zip(rows, result.items):
            original = row["prompt"].replace("|", "/")[:50]
            f.write(f"| {row['id']} | {original} | {item.prompt} | {item.cosine_similarity:.4f} |\n")
    print(f"Saved -> {RESULTS_MD_PATH}")

    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(
            {
                "wall_clock_seconds": result.wall_clock_seconds,
                "items": [
                    {"id": row["id"], "prompt": item.prompt, "cosine_similarity": item.cosine_similarity}
                    for row, item in zip(rows, result.items)
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved -> {RESULTS_JSON_PATH}")


if __name__ == "__main__":
    main()
