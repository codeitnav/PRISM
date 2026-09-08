#!/usr/bin/env python3
"""Task 1.1 - DiffusionDB subset ingestion (scoped to 700 pairs).

Downloads DiffusionDB's predefined "2m_random_5k" split from Hugging Face
(poloclub/diffusiondb), filters it, and writes a curated set of image-prompt
pairs to disk, stopping once MAX_PAIRS kept pairs have been collected.

Scope note: the original roadmap targets a ~200-500K curated subset built by
hand from the full 2M/14M shards. That assumes real bandwidth/storage/GPU
budget. This project runs on a CPU-only, storage-constrained dev machine, so
we draw from DiffusionDB's official small predefined config and cap the kept
set at 700 pairs. Documented as a deliberate, justified scope reduction, not
a shortcut.

Output:
    data/diffusiondb/images/<id>.png
    data/diffusiondb/pairs.parquet   (id, image_path, prompt, seed, cfg, sampler)

Usage (inside the ml container):
    docker compose run --rm ml python scripts/ingest_diffusiondb.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

HF_CONFIG = "2m_random_5k"
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "diffusiondb"
IMAGES_DIR = DATA_DIR / "images"
PAIRS_PATH = DATA_DIR / "pairs.parquet"

MIN_PROMPT_TOKENS = 3
MAX_PROMPT_TOKENS = 60
NSFW_THRESHOLD = 0.2  # drop a row if image_nsfw or prompt_nsfw >= this
MAX_PAIRS = 700  # stop early once we have this many kept pairs (scope: see project notes)


def normalize(prompt: str) -> str:
    """Collapse case/whitespace so near-identical prompts dedupe as equal."""
    return " ".join(prompt.lower().split())


def main() -> None:
    print(f"Downloading DiffusionDB config '{HF_CONFIG}' from Hugging Face...")
    ds = load_dataset("poloclub/diffusiondb", HF_CONFIG, split="train", trust_remote_code=True)
    print(f"Raw rows: {len(ds)}")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    seen_prompts: set[str] = set()
    kept_rows = []
    all_lengths = []
    dropped_nsfw = 0
    dropped_length = 0
    dropped_dupe = 0

    for row in ds:
        prompt = row["prompt"].strip()
        token_count = len(prompt.split())
        all_lengths.append(token_count)

        if float(row["image_nsfw"] or 0.0) >= NSFW_THRESHOLD or float(row["prompt_nsfw"] or 0.0) >= NSFW_THRESHOLD:
            dropped_nsfw += 1
            continue
        if not (MIN_PROMPT_TOKENS <= token_count <= MAX_PROMPT_TOKENS):
            dropped_length += 1
            continue

        key = normalize(prompt)
        if key in seen_prompts:
            dropped_dupe += 1
            continue
        seen_prompts.add(key)

        image_id = f"{len(kept_rows):06d}"
        row["image"].convert("RGB").save(IMAGES_DIR / f"{image_id}.png")

        kept_rows.append(
            {
                "id": image_id,
                "image_path": f"images/{image_id}.png",
                "prompt": prompt,
                "seed": row["seed"],
                "cfg": row["cfg"],
                "sampler": row["sampler"],
            }
        )

        if len(kept_rows) >= MAX_PAIRS:
            print(f"Reached MAX_PAIRS={MAX_PAIRS}, stopping early.")
            break

    pq.write_table(pa.Table.from_pylist(kept_rows), PAIRS_PATH)

    print("\n=== Stats report ===")
    print(f"Kept:    {len(kept_rows)} pairs")
    print(f"Dropped: {dropped_nsfw} NSFW, {dropped_length} bad length, {dropped_dupe} duplicate")
    print(
        f"Prompt length (tokens) - min: {min(all_lengths)}, max: {max(all_lengths)}, "
        f"mean: {sum(all_lengths) / len(all_lengths):.1f}"
    )
    print(f"Duplicate rate: {dropped_dupe / len(all_lengths) * 100:.1f}%")
    print(f"Saved -> {PAIRS_PATH}")


if __name__ == "__main__":
    main()
