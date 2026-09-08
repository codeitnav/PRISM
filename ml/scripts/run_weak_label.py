#!/usr/bin/env python3
"""Task 1.3 - apply weak structured labels to the ingested corpus.

Reads every raw prompt from data/diffusiondb/pairs.parquet, runs the rule +
lexicon pass in app.weak_label, and produces two outputs:

1. data/diffusiondb/pairs_labeled.parquet - all 700 pairs (train/val/test,
   per the Sync-1 handover) with the structured fields flattened into
   columns, for the decomposition SFT dataset (Task 5.2) and component-wise
   eval (Task 4.3).
2. An in-place update of MongoDB's `reference_prompts` documents, filling
   the `structured_fields` field that scripts/seed_mongo.py left as null.
   Uses update_one($set) per document so seed_mongo's `faiss_id` survives -
   deleting and reinserting would wipe it and break retrieval.

Also prints the coverage stats and the top corpus-frequency segments that no
lexicon classifies, which is the feedback loop for extending the lexicons.

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.run_weak_label
    docker compose run --rm ml python -m scripts.run_weak_label --no-mongo
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pymongo import MongoClient

from app.weak_label import load_llm_seed_labels, mine_lexicon_candidates, weak_label_row

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
LABELED_PATH = DATA_DIR / "diffusiondb" / "pairs_labeled.parquet"
SEED_LABELS_PATH = DATA_DIR / "diffusiondb" / "llm_seed_labels.jsonl"

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongo:27017/prism")
COLLECTION_NAME = "reference_prompts"

SINGLE_SLOT_FIELDS = ["style", "medium", "lighting", "tone"]
LIST_FIELDS = ["modifiers", "negative_constraints"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-mongo",
        action="store_true",
        help="write the parquet only; skip the reference_prompts update",
    )
    args = parser.parse_args()

    rows = pq.read_table(PAIRS_PATH).to_pylist()
    seeds = load_llm_seed_labels(SEED_LABELS_PATH)
    print(f"Labeling {len(rows)} prompts ({len(seeds)} LLM seed labels available)...")

    labeled_rows = []
    provenance_counts: Counter[str] = Counter()
    coverage: Counter[str] = Counter()

    for row in rows:
        fields, provenance = weak_label_row(row["prompt"], row["id"], seeds)
        provenance_counts[provenance] += 1

        for name in SINGLE_SLOT_FIELDS:
            if getattr(fields, name):
                coverage[name] += 1
        for name in LIST_FIELDS:
            if getattr(fields, name):
                coverage[name] += 1

        labeled_rows.append(
            {
                "id": row["id"],
                "image_path": row["image_path"],
                "prompt": row["prompt"],
                "subject": fields.subject,
                "style": fields.style,
                "medium": fields.medium,
                "lighting": fields.lighting,
                "tone": fields.tone,
                "modifiers": fields.modifiers,
                "negative_constraints": fields.negative_constraints,
                "label_provenance": provenance,
            }
        )

    pq.write_table(pa.Table.from_pylist(labeled_rows), LABELED_PATH)
    print(f"Saved -> {LABELED_PATH}")

    total = len(labeled_rows)
    print("\n=== Field coverage (share of prompts with a non-empty value) ===")
    for name in SINGLE_SLOT_FIELDS + LIST_FIELDS:
        print(f"  {name:22s} {coverage[name]:4d}/{total}  ({coverage[name] / total * 100:5.1f}%)")
    print(f"\nProvenance: {dict(provenance_counts)}")

    print("\n=== Top unclassified segments (lexicon-extension candidates) ===")
    for segment, count in mine_lexicon_candidates(r["prompt"] for r in rows):
        print(f"  {count:4d}  {segment}")

    if args.no_mongo:
        print("\n--no-mongo: skipping reference_prompts update.")
        return

    client = MongoClient(MONGO_URI)
    collection = client.get_default_database()[COLLECTION_NAME]
    seeded_ids = {doc["_id"] for doc in collection.find({}, {"_id": 1})}
    if not seeded_ids:
        print(
            f"\nWARNING: '{COLLECTION_NAME}' is empty - run `make seed` first, "
            "then re-run this script to populate structured_fields."
        )
        return

    updated = 0
    for row in labeled_rows:
        if row["id"] not in seeded_ids:
            continue  # val/test rows aren't seeded; only the 560 train rows are
        collection.update_one(
            {"_id": row["id"]},
            {
                "$set": {
                    "structured_fields": {
                        "subject": row["subject"],
                        "style": row["style"],
                        "medium": row["medium"],
                        "lighting": row["lighting"],
                        "modifiers": row["modifiers"],
                        "tone": row["tone"],
                        "negative_constraints": row["negative_constraints"],
                    }
                }
            },
        )
        updated += 1

    remaining = collection.count_documents({"structured_fields": None})
    print(f"\nUpdated {updated} reference_prompts documents; {remaining} still null.")
    sample = collection.find_one({"structured_fields": {"$ne": None}})
    print(f"Sample _id={sample['_id']} faiss_id={sample.get('faiss_id')}")
    print(f"  structured_fields={sample['structured_fields']}")


if __name__ == "__main__":
    main()
