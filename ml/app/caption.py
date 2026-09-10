"""Image captioning.

caption_images(list[bytes]) -> list[str], cached on disk by content hash.

The caption supplies a literal, grounded description of the image, which
neither retrieval nor gradient-based prompt inversion provides: those describe
a different image, or optimize for embedding similarity rather than meaning.

Model selection is deliberate and constrained. BLIP-2 (~3.74B params, ~15 GB
fp32 / ~7.5 GB bf16) and LLaVA-1.5-7B do not fit the available memory on
CPU-only hardware, so the default is BLIP-1 large (~470M, ~1.9 GB) - the
direct predecessor, trained for the same task, at a real cost in caption
detail and compositionality.

The limit is configuration, not code: CAPTION_MODEL accepts any BLIP or
BLIP-2 checkpoint and the architecture is resolved from its config, so a
GPU host can run a larger model unchanged. The cache key includes the model
name, so switching checkpoints cannot serve stale captions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
CAPTION_CACHE_DIR = DATA_DIR / "cache" / "captions"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_MODEL = os.environ.get("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")

# Beam search rather than sampling: captions are paired with structured
# targets in a training set, so they must be reproducible across runs.
NUM_BEAMS = int(os.environ.get("CAPTION_NUM_BEAMS", "3"))
MAX_NEW_TOKENS = int(os.environ.get("CAPTION_MAX_NEW_TOKENS", "40"))

_model = None
_processor = None


def _model_class(model_name: str):
    """Resolve the model class from the checkpoint's own config."""
    model_type = AutoConfig.from_pretrained(model_name).model_type
    if model_type == "blip":
        from transformers import BlipForConditionalGeneration

        return BlipForConditionalGeneration
    if model_type == "blip-2":
        from transformers import Blip2ForConditionalGeneration

        return Blip2ForConditionalGeneration
    raise ValueError(
        f"CAPTION_MODEL={model_name!r} has model_type={model_type!r}; "
        "expected 'blip' or 'blip-2'."
    )


def _get_model():
    global _model, _processor
    if _model is None:
        processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
        model = _model_class(CAPTION_MODEL).from_pretrained(CAPTION_MODEL).to(DEVICE)
        model.eval()
        _model, _processor = model, processor
    return _model, _processor


def get_caption_model():
    """Return the lazily loaded model and processor."""
    return _get_model()


def _model_slug() -> str:
    return CAPTION_MODEL.replace("/", "__")


def _cache_path(image_bytes: bytes) -> Path:
    # The model name is part of the key so a caption produced by a different
    # checkpoint can never be served.
    key = hashlib.sha256(image_bytes + b"\x00" + CAPTION_MODEL.encode()).hexdigest()
    return CAPTION_CACHE_DIR / _model_slug() / f"{key}.json"


def _load_cached(image_bytes: bytes) -> str | None:
    path = _cache_path(image_bytes)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["caption"]


def _save_cached(image_bytes: bytes, caption: str) -> None:
    path = _cache_path(image_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"caption": caption, "model": CAPTION_MODEL}, f)


# BLIP opens the large majority of captions with a contentless declarative
# phrase ("there is a ...", "an image of ..."), which carries no information
# about the image and is noise as a downstream input feature.
#
# Not applied inside caption_images: the cache and persisted captions hold raw
# model output so the stored data remains a faithful record. Callers opt in.
_CAPTION_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"there\s+(?:is|are)|this\s+is|here\s+is"      # contentless existential opener
    r"|(?:an?\s+)?(?:image|picture)\s+of"          # contentless framing
    r")\s+(?:an?\s+|two\s+|some\s+)?",
    re.IGNORECASE,
)


def strip_caption_boilerplate(caption: str) -> str:
    """Drop the contentless leading phrase from a caption, if present.

    Applied repeatedly, since these phrases stack ("this is" + "a picture
    of"). Captions without one are returned unchanged.

    Medium-bearing openers are deliberately preserved - "photo of",
    "painting of", "screenshot of", "3d rendering of" - because the medium is
    visual evidence and one of the fields being reconstructed.
    """
    text = caption.strip()
    for _ in range(3):
        stripped = _CAPTION_BOILERPLATE_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    # Never return empty just because a caption was nothing but boilerplate -
    # the original is more useful than nothing.
    return text or caption.strip()


def caption_images(images: list[bytes], batch_size: int = 8) -> list[str]:
    """Caption raw image bytes, one caption per input, order preserved.

    Cache hits skip the model; only misses are batched through it.
    """
    results: list[str | None] = [_load_cached(b) for b in images]
    todo = [i for i, r in enumerate(results) if r is None]

    if todo:
        model, processor = _get_model()
        for start in range(0, len(todo), batch_size):
            chunk = todo[start : start + batch_size]
            pil = [Image.open(BytesIO(images[i])).convert("RGB") for i in chunk]
            inputs = processor(images=pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model.generate(
                    **inputs, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS
                )
            captions = [
                processor.decode(seq, skip_special_tokens=True).strip() for seq in out
            ]
            for i, caption in zip(chunk, captions):
                results[i] = caption
                _save_cached(images[i], caption)

    return [r for r in results if r is not None]
