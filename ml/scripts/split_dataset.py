#!/usr/bin/env python3
"""Task 1.2 - Held-out train/val/test splits, prompt-disjoint.

Splits data/diffusiondb/pairs.parquet into train/val/test sets such that no
two near-duplicate prompts (cosine similarity >= SIMILARITY_THRESHOLD, via a
small sentence-embedding model) end up in different splits.

How: embed every prompt, union-find prompts into clusters wherever a pair's
similarity meets the threshold, then greedily assign whole clusters (not
individual prompts) to whichever split is furthest below its target size.
Because clustering is done over ALL pairs (not just same-split ones), any two
prompts landing in different splits are guaranteed to be below the threshold.

Output:
    data/splits/train.json  (list of ids)
    data/splits/val.json
    data/splits/test.json

Usage (inside the ml container):
    docker compose run --rm ml python scripts/split_dataset.py
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
SPLITS_DIR = DATA_DIR / "splits"

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# TEST_FRAC is whatever remains

SIMILARITY_THRESHOLD = 0.95
RANDOM_SEED = 42
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class UnionFind:
    """Tracks which prompts belong to the same near-duplicate cluster."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main() -> None:
    table = pq.read_table(PAIRS_PATH)
    ids = table.column("id").to_pylist()
    prompts = table.column("prompt").to_pylist()
    n = len(ids)
    print(f"Loaded {n} pairs from {PAIRS_PATH}")

    print(f"Embedding prompts with '{EMBED_MODEL}' to find near-duplicates...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(prompts, normalize_embeddings=True, show_progress_bar=False)
    sims = embeddings @ embeddings.T  # cosine similarity, since embeddings are normalized

    uf = UnionFind(n)
    duplicate_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= SIMILARITY_THRESHOLD:
                uf.union(i, j)
                duplicate_pairs += 1

    clusters = defaultdict(list)
    for idx in range(n):
        clusters[uf.find(idx)].append(idx)
    cluster_list = list(clusters.values())

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(cluster_list)
    cluster_list.sort(key=len, reverse=True)  # place big clusters first, they're hardest to fit

    n_train_target = round(n * TRAIN_FRAC)
    n_val_target = round(n * VAL_FRAC)
    n_test_target = n - n_train_target - n_val_target
    targets = {"train": n_train_target, "val": n_val_target, "test": n_test_target}
    counts = {"train": 0, "val": 0, "test": 0}
    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for cluster in cluster_list:
        split = min(targets, key=lambda s: counts[s] - targets[s])
        buckets[split].extend(cluster)
        counts[split] += len(cluster)

    # Sanity check: confirm the guarantee actually holds before writing anything.
    violations = 0
    split_of = {idx: split for split, members in buckets.items() for idx in members}
    for i in range(n):
        for j in range(i + 1, n):
            if split_of[i] != split_of[j] and sims[i, j] >= SIMILARITY_THRESHOLD:
                violations += 1

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, indices in buckets.items():
        split_ids = sorted(ids[idx] for idx in indices)
        with open(SPLITS_DIR / f"{split}.json", "w") as f:
            json.dump(split_ids, f, indent=2)

    print("\n=== Split sizes ===")
    for split in ("train", "val", "test"):
        print(f"{split}: {len(buckets[split])} (target ~{targets[split]})")
    multi_member_clusters = sum(1 for c in cluster_list if len(c) > 1)
    print(f"\nNear-duplicate clusters found: {multi_member_clusters} ({duplicate_pairs} pairs above cosine {SIMILARITY_THRESHOLD})")
    if violations == 0:
        print(f"VERIFIED: zero cross-split prompt neighbors above cosine {SIMILARITY_THRESHOLD}")
    else:
        print(f"FAIL: {violations} cross-split neighbor pair(s) remain above cosine {SIMILARITY_THRESHOLD}")


if __name__ == "__main__":
    main()
