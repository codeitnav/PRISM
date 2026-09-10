#!/usr/bin/env python3
"""Generate PEZ prompts for arbitrary dataset splits.

Separate from scripts/run_pez_baseline.py, which covers exactly the images the
published baseline metrics are computed from and is asserted on by the
evaluation harness. This writes to its own file, which
scripts/build_decomp_sft.py picks up by glob alongside it.

Iteration count matches the benchmarked baseline by default. A PEZ prompt used
as model input must come from the same distribution the pipeline produces at
inference, so lowering --iterations to save time introduces train/serve skew.

PEZ is the slowest stage in the pipeline. Batching amortizes much of the cost,
but a full split still takes hours on CPU; progress is checkpointed after every
batch, so the run is safe to interrupt and resume.

Output:
    data/results/pez_sft_<splits>.json

Usage:
    python -m scripts.run_pez_for_sft --splits val test
    python -m scripts.run_pez_for_sft --splits train --batch-size 40
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from app.baselines.pez import ITERATIONS, NUM_SOFT_TOKENS, reconstruct_batch
from app.embed import embed_images

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="images optimized together; bounded by available memory",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    ids: list[str] = []
    seen: set[str] = set()
    for split in args.splits:
        path = SPLITS_DIR / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `make split` first.")
        with open(path) as f:
            for dataset_id in json.load(f):
                if dataset_id not in seen:
                    seen.add(dataset_id)
                    ids.append(dataset_id)
    ids.sort()
    if args.limit:
        ids = ids[: args.limit]

    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    rows = [rows_by_id[i] for i in ids]

    out_path = RESULTS_DIR / f"pez_sft_{'_'.join(args.splits)}.json"
    # Resume support: hours-long runs get interrupted, and re-optimizing
    # images already covered would be pure waste.
    done: dict[str, dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            for item in json.load(f).get("items", []):
                done[item["id"]] = item
        print(f"Resuming: {len(done)} of {len(rows)} already in {out_path.name}")

    todo = [r for r in rows if r["id"] not in done]
    print(
        f"PEZ for {len(todo)} images ({args.iterations} iterations, "
        f"batch size {args.batch_size})"
    )

    t0 = time.time()
    for start in range(0, len(todo), args.batch_size):
        chunk = todo[start : start + args.batch_size]
        image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in chunk]
        targets = torch.from_numpy(embed_images(image_bytes))
        result = reconstruct_batch(targets, iterations=args.iterations)

        for row, item in zip(chunk, result.items):
            done[row["id"]] = {
                "id": row["id"],
                "prompt": item.prompt,
                "cosine_similarity": item.cosine_similarity,
            }

        # Persist after every batch, so an interrupted run keeps its progress.
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "iterations": args.iterations,
                    "num_tokens": NUM_SOFT_TOKENS,
                    "wall_clock_seconds": time.time() - t0,
                    "items": [done[r["id"]] for r in rows if r["id"] in done],
                },
                f,
                indent=2,
            )

        covered = start + len(chunk)
        elapsed = time.time() - t0
        per_image = elapsed / covered
        remaining = (len(todo) - covered) * per_image
        print(
            f"  {covered}/{len(todo)} | {elapsed / 60:.1f} min elapsed | "
            f"{per_image:.1f}s/image | ~{remaining / 60:.0f} min left | "
            f"batch mean CLIP-score {result.mean_cosine_similarity:.4f}"
        )

    print(f"\nSaved {len(done)} PEZ prompts -> {out_path}")


if __name__ == "__main__":
    main()
