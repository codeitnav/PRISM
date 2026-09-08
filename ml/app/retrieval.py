"""Task 2.4 - retrieval: FAISS top-k search resolved to Mongo documents."""

from __future__ import annotations

import json
import os
from pathlib import Path

import faiss
import numpy as np
from pymongo import MongoClient

from app.config import settings

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
FAISS_DIR = DATA_DIR / "faiss"
INDEX_PATH = FAISS_DIR / "index.faiss"
ID_MAP_PATH = FAISS_DIR / "id_map.json"

_index = None
_id_map = None
_mongo_client = None


def _get_index():
    global _index, _id_map
    if _index is None:
        if not INDEX_PATH.exists() or not ID_MAP_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH} - run Task 2.2 (make build-index) first."
            )
        _index = faiss.read_index(str(INDEX_PATH))
        with open(ID_MAP_PATH) as f:
            _id_map = json.load(f)
    return _index, _id_map


def _get_collection():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.mongo_uri)
    return _mongo_client.get_default_database()["reference_prompts"]


def retrieve_candidates(query_vec: np.ndarray, top_k: int = 10) -> list[dict]:
    """Search the FAISS index and resolve hits to Mongo documents.

    query_vec: a single (D,) L2-normalized embedding (from app.embed).
    Returns dicts shaped like app.schemas.candidate.Candidate, ordered by
    descending similarity - retrieval_margin (top1 - top2) can be derived
    from candidates[0]['similarity'] - candidates[1]['similarity'] by whoever
    assembles the final Reconstruction object (Task 6.2), so it isn't
    duplicated here.
    """
    index, id_map = _get_index()
    collection = _get_collection()

    scores, indices = index.search(query_vec.reshape(1, -1).astype("float32"), top_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # FAISS pads with -1 if top_k exceeds the index size
        dataset_id = id_map[str(idx)]
        doc = collection.find_one({"_id": dataset_id})
        if doc is None:
            continue
        candidates.append(
            {
                "id": doc["_id"],
                "prompt": doc["prompt"],
                "source": doc["source"],
                "similarity": max(0.0, min(1.0, float(score))),
                "structured_fields": doc.get("structured_fields"),
                "thumbnail_url": None,
            }
        )
    return candidates
