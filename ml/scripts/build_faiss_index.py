#!/usr/bin/env python3
"""Task 2.2 - FAISS index build.

Embeds every training-split reference image (CLIP ViT-B/32, via
app.embed.embed_images - see that module's scope note on the ViT-B/32 choice)
and builds a FAISS index over them for nearest-neighbor retrieval.

Only the TRAIN split is indexed - val/test images are held out as queries for
evaluation later and must never be retrievable candidates, or the system
could "retrieve" a test image against itself and evaluation would be
meaningless.

Output:
    data/faiss/index.faiss   - FAISS IndexFlatIP over normalized CLIP vectors
    data/faiss/id_map.json   - {faiss_int_id: dataset_id}, so a FAISS search
                               result resolves back to its row/prompt (and
                               will be how Task 2.3 keys the matching Mongo
                               document)

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.build_faiss_index
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

from app.embed import embed_images

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
TRAIN_SPLIT_PATH = DATA_DIR / "splits" / "train.json"
FAISS_DIR = DATA_DIR / "faiss"

BATCH_SIZE = 20


def main() -> None:
    with open(TRAIN_SPLIT_PATH) as f:
        train_ids = set(json.load(f))
    rows = [r for r in pq.read_table(PAIRS_PATH).to_pylist() if r["id"] in train_ids]
    print(f"Embedding {len(rows)} training images (batches of {BATCH_SIZE})...")

    all_vecs = []
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in batch]
        all_vecs.append(embed_images(image_bytes))
        done = min(i + BATCH_SIZE, len(rows))
        print(f"  {done}/{len(rows)} embedded ({time.time() - t0:.0f}s elapsed)")

    vectors = np.vstack(all_vecs).astype("float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # inner product == cosine similarity, vectors are L2-normalized
    index.add(vectors)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_DIR / "index.faiss"))

    id_map = {str(i): rows[i]["id"] for i in range(len(rows))}
    with open(FAISS_DIR / "id_map.json", "w") as f:
        json.dump(id_map, f, indent=2)

    print(f"\nBuilt FAISS index: {index.ntotal} vectors, dim={dim}")
    print(f"Saved -> {FAISS_DIR / 'index.faiss'}, {FAISS_DIR / 'id_map.json'}")

    # Smoke test: a known training image, re-embedded, should retrieve itself at rank 1.
    smoke_row = rows[0]
    smoke_bytes = (IMAGES_DIR / smoke_row["image_path"]).read_bytes()
    query_vec = embed_images([smoke_bytes]).astype("float32")
    scores, indices = index.search(query_vec, k=1)
    top_id = id_map[str(indices[0][0])]
    if top_id == smoke_row["id"]:
        print(
            f"SMOKE TEST PASSED: training image '{smoke_row['id']}' retrieved itself "
            f"at rank 1 (score={scores[0][0]:.4f})"
        )
    else:
        print(f"SMOKE TEST FAILED: expected '{smoke_row['id']}', got '{top_id}'")


if __name__ == "__main__":
    main()
