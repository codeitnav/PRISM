#!/usr/bin/env python3
"""Task 2.3 - Mongo seeding.

Loads the training-split reference pairs (the same 560 images indexed by
Task 2.2's FAISS index) into MongoDB's `reference_prompts` collection, keyed
by the same dataset_id used in data/faiss/id_map.json - so a FAISS search
result resolves straight to a Mongo document.

Structured fields (subject/style/medium/...) aren't populated yet - that's
Task 1.3's weak-labeling output (Navya's side), added later by an update to
these same documents, not by this script.

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.seed_mongo
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
from pymongo import MongoClient, TEXT

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
ID_MAP_PATH = DATA_DIR / "faiss" / "id_map.json"

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongo:27017/prism")
COLLECTION_NAME = "reference_prompts"


def main() -> None:
    with open(ID_MAP_PATH) as f:
        id_map = json.load(f)  # {faiss_id: dataset_id}
    faiss_id_by_dataset_id = {dataset_id: int(faiss_id) for faiss_id, dataset_id in id_map.items()}

    rows = pq.read_table(PAIRS_PATH).to_pylist()
    train_rows = [r for r in rows if r["id"] in faiss_id_by_dataset_id]
    print(f"Seeding {len(train_rows)} reference prompts into '{COLLECTION_NAME}'...")

    client = MongoClient(MONGO_URI)
    collection = client.get_default_database()[COLLECTION_NAME]

    collection.delete_many({})  # idempotent: safe to re-run

    documents = [
        {
            "_id": r["id"],
            "prompt": r["prompt"],
            "seed": r["seed"],
            "cfg": r["cfg"],
            "sampler": r["sampler"],
            "source": "diffusiondb",
            "faiss_id": faiss_id_by_dataset_id[r["id"]],
            "structured_fields": None,  # populated later by Task 1.3's output
            "tags": [],
        }
        for r in train_rows
    ]
    collection.insert_many(documents)

    collection.create_index("faiss_id", unique=True)
    collection.create_index([("prompt", TEXT)])

    print(f"Inserted {collection.count_documents({})} documents.")

    # Confirm the roadmap's "<10ms faiss_id lookup" requirement. Warm the
    # connection up with one untimed query first - a fresh MongoClient
    # connects lazily on its first operation, and that one-time connect cost
    # would otherwise dominate the measurement.
    sample_faiss_id = documents[0]["faiss_id"]
    collection.find_one({"faiss_id": sample_faiss_id})
    t0 = time.time()
    doc = collection.find_one({"faiss_id": sample_faiss_id})
    elapsed_ms = (time.time() - t0) * 1000
    print(f"Lookup by faiss_id={sample_faiss_id}: {elapsed_ms:.2f}ms -> prompt: {doc['prompt'][:60]!r}")


if __name__ == "__main__":
    main()
