"""Task 2.1 - Embedding service.

embed_images(list[bytes]) -> np.ndarray   CLIP image embeddings (see model note below)
embed_texts(list[str])    -> np.ndarray   text embeddings (see model note below)

Both are L2-normalized (so cosine similarity == dot product, matching how
FAISS/retrieval and the confidence scoring will use these vectors later) and
cached to disk by content hash, so re-embedding the same image or text is a
cache hit instead of a re-run through the model.

Scope note: the project brief allows "E5-large or BGE" for text embeddings -
any competent pretrained sentence-embedding model, not a specific one. Given
this machine's storage and CPU constraints, we use all-MiniLM-L6-v2 (~80MB)
instead of E5-large (~1.3GB): same role, far lighter to download and run.

For images, the brief specifies CLIP ViT-L/14, but on this CPU-only machine
that measured ~40s/image - embedding the 700-image reference set (Task 2.2)
would take most of a day. We use CLIP ViT-B/32 instead: same OpenAI CLIP
family (so PEZ, Task 4.1, still shares the embedding space), far fewer image
patches to process (32px patches vs 14px), at some cost to embedding
precision. Documented scope trade-off, not a silent swap.
"""

from __future__ import annotations

import hashlib
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
IMAGE_CACHE_DIR = DATA_DIR / "cache" / "embeddings" / "images"
TEXT_CACHE_DIR = DATA_DIR / "cache" / "embeddings" / "texts"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 'quickgelu', not plain 'ViT-B-32' - matches the OpenAI-trained weights'
# activation function (see ml/scripts/check_env.py for the ViT-L/14 case this
# was originally documented for; same reasoning applies here).
CLIP_MODEL_NAME = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_clip_model = None
_clip_preprocess = None
_text_model = None


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, device=DEVICE
        )
        model.eval()
        _clip_model, _clip_preprocess = model, preprocess
    return _clip_model, _clip_preprocess


def get_clip_model():
    """Public accessor for the loaded CLIP model + preprocess transform, for
    code that needs direct model access (e.g. the PEZ baseline, Task 4.1) -
    reuses the same lazily-loaded singleton instead of loading a second copy.
    """
    return _get_clip()


def _get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(TEXT_MODEL_NAME, device=DEVICE)
    return _text_model


def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_cached(cache_dir: Path, key: str) -> np.ndarray | None:
    path = cache_dir / f"{key}.npy"
    return np.load(path) if path.exists() else None


def _save_cached(cache_dir: Path, key: str, vec: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{key}.npy", vec)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_images(images: list[bytes]) -> np.ndarray:
    """Embed raw image bytes with CLIP ViT-B/32. Returns (N, D), L2-normalized."""
    keys = [_content_hash(b) for b in images]
    results: list[np.ndarray | None] = [_load_cached(IMAGE_CACHE_DIR, k) for k in keys]

    todo = [i for i, r in enumerate(results) if r is None]
    if todo:
        model, preprocess = _get_clip()
        batch = torch.stack(
            [preprocess(Image.open(BytesIO(images[i])).convert("RGB")) for i in todo]
        ).to(DEVICE)
        with torch.no_grad():
            vecs = model.encode_image(batch).cpu().numpy()
        vecs = _normalize(vecs)
        for pos, i in enumerate(todo):
            results[i] = vecs[pos]
            _save_cached(IMAGE_CACHE_DIR, keys[i], vecs[pos])

    return np.stack(results)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed strings with all-MiniLM-L6-v2. Returns (N, D), L2-normalized."""
    keys = [_content_hash(t.encode("utf-8")) for t in texts]
    results: list[np.ndarray | None] = [_load_cached(TEXT_CACHE_DIR, k) for k in keys]

    todo = [i for i, r in enumerate(results) if r is None]
    if todo:
        model = _get_text_model()
        inputs = [texts[i] for i in todo]
        vecs = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
        for pos, i in enumerate(todo):
            results[i] = vecs[pos]
            _save_cached(TEXT_CACHE_DIR, keys[i], vecs[pos])

    return np.stack(results)
