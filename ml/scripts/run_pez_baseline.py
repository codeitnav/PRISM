#!/usr/bin/env python3
"""Task 4.1 - PEZ baseline evaluation.

Runs the PEZ baseline (ml/app/baselines/pez.py) on the test split's images,
all in one batch, and logs a readable hard prompt + CLIP-score (cosine
similarity between the hard prompt's text embedding and the original
image's embedding - the same quantity PEZ optimizes) per image, plus the
mean across the set. This is the number every later reconstruction method
in this project has to beat.

Output:
    data/results/pez_baseline.md - per-image results table + mean CLIP-score
    (copy this into docs/results/ afterward - docs/ is mounted read-only in
    the ml container, see docker-compose.yml)

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.run_pez_baseline
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch

from app.baselines.pez import reconstruct_batch
from app.embed import embed_images

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
TEST_SPLIT_PATH = DATA_DIR / "splits" / "test.json"
RESULTS_PATH = DATA_DIR / "results" / "pez_baseline.md"

NUM_TEST_IMAGES = 20


def main() -> None:
    with open(TEST_SPLIT_PATH) as f:
        test_ids = json.load(f)[:NUM_TEST_IMAGES]

    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    rows = [rows_by_id[i] for i in test_ids]

    print(f"Embedding {len(rows)} test images...")
    image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in rows]
    target_embeddings = torch.from_numpy(embed_images(image_bytes))

    print(f"Running PEZ optimization on all {len(rows)} images as one batch...")
    result = reconstruct_batch(target_embeddings)

    print(f"\nDone in {result.wall_clock_seconds:.1f}s ({result.iterations} iterations, {result.num_tokens} soft tokens)")
    print(f"Mean CLIP-score: {result.mean_cosine_similarity:.4f}\n")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write("# PEZ Baseline Results (Task 4.1)\n\n")
        f.write(
            f"Ran on {len(rows)} test-split images, {result.iterations} optimization iterations, "
            f"{result.num_tokens} soft tokens, all images batched together in one run.\n\n"
        )
        f.write(f"**Wall-clock time:** {result.wall_clock_seconds:.1f}s\n\n")
        f.write(f"**Mean CLIP-score (cosine similarity to original image):** {result.mean_cosine_similarity:.4f}\n\n")
        f.write("| Image ID | Original prompt (truncated) | PEZ hard prompt | CLIP-score |\n")
        f.write("|---|---|---|---|\n")
        for row, item in zip(rows, result.items):
            original = row["prompt"].replace("|", "/")[:50]
            pez_prompt = item.prompt.replace("|", "/")
            f.write(f"| {row['id']} | {original} | {pez_prompt} | {item.cosine_similarity:.4f} |\n")

    print(f"Saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
