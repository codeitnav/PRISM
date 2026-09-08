#!/usr/bin/env python3
"""Task 2.1 - throughput benchmark for the embedding service.

Times embed_images/embed_texts (cold, i.e. no cache hits) on a sample of the
already-ingested DiffusionDB pairs and logs items/sec for each, so we know
what throughput this CPU-only machine actually gets.

Usage (inside the ml container):
    docker compose run --rm ml python scripts/bench_embed.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from app.embed import embed_images, embed_texts

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"

SAMPLE_SIZE = 5  # kept small - this machine is CPU-only, no need for a large benchmark


def main() -> None:
    rows = pq.read_table(PAIRS_PATH).to_pylist()[:SAMPLE_SIZE]
    prompts = [r["prompt"] for r in rows]
    image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in rows]

    t0 = time.time()
    text_vecs = embed_texts(prompts)
    text_elapsed = time.time() - t0
    print(
        f"Text:  {len(prompts)} prompts in {text_elapsed:.2f}s "
        f"({len(prompts) / text_elapsed:.1f} items/sec), dim={text_vecs.shape[1]}"
    )

    t0 = time.time()
    image_vecs = embed_images(image_bytes)
    image_elapsed = time.time() - t0
    print(
        f"Image: {len(image_bytes)} images in {image_elapsed:.2f}s "
        f"({len(image_bytes) / image_elapsed:.1f} items/sec), dim={image_vecs.shape[1]}"
    )


if __name__ == "__main__":
    main()
