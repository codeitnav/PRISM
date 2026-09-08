"""Task 5.1 - Captioning stage.

caption_images(list[bytes]) -> list[str]   image captions, cached by content hash

A caption is one of the three inputs the decomposition model is conditioned on
in Task 5.2 (the other two being Kajal's retrieved candidate prompts and the
PEZ baseline output). It supplies what neither of those can: a literal,
grounded description of what is actually *in* the image, independent of
whatever the nearest training prompt happened to say.

Captions are cached to disk by content hash - the same pattern as app.embed -
because captioning is the second-slowest stage in the pipeline after PEZ, and
5.2 will read the same images repeatedly while the SFT dataset is iterated on.

Model choice (scope decision, measured not assumed)
---------------------------------------------------
The roadmap asks for BLIP-2, or LLaVA-1.5-7B "if VRAM allows". Neither fits
this machine, and the arithmetic is worth recording rather than hand-waving:

    blip2-opt-2.7b = OPT-2.7b decoder (2560 hidden x 32 layers, ~2.65B)
                   + EVA ViT-g vision encoder (1408 hidden x 39 layers, ~1.0B)
                   + Q-Former (~0.1B)
                   ~= 3.74B params -> ~15.0 GB in fp32, ~7.5 GB in bf16
    LLaVA-1.5-7B   ~= 7B params    -> ~28.0 GB in fp32, ~14.0 GB in bf16

This box is CPU-only (confirmed by scripts/check_env.py) with ~6.3 GB of RAM
actually available, so BLIP-2 does not fit even at bf16 - and torch's CPU
bf16 inference is slow enough that it would not be usable if it did.

So the default is BLIP-1 large (Salesforce/blip-image-captioning-large,
~470M params, ~1.9 GB fp32): the direct predecessor, same Salesforce BLIP
captioning lineage, trained for exactly this task. Captions are shorter and
less compositional than BLIP-2's, which is a real quality cost and is
reported in docs/results/captioning.md.

This is a *configuration* limit, not a code limit. CAPTION_MODEL selects any
BLIP or BLIP-2 checkpoint and the right model class is chosen from its config,
so on a GPU machine:

    CAPTION_MODEL=Salesforce/blip2-opt-2.7b docker compose run --rm ml \
        python -m scripts.run_captioning

runs the roadmap's intended model with no code change. The cache key includes
the model name, so switching models does not serve stale captions.
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

# Beam search rather than sampling: the SFT dataset in Task 5.2 pairs each
# caption with a structured target, so the caption has to be reproducible
# across runs. Sampling would make the training data non-deterministic.
NUM_BEAMS = int(os.environ.get("CAPTION_NUM_BEAMS", "3"))
MAX_NEW_TOKENS = int(os.environ.get("CAPTION_MAX_NEW_TOKENS", "40"))

_model = None
_processor = None


def _model_class(model_name: str):
    """Pick the right architecture from the checkpoint's own config.

    Keeps BLIP-2 (and any future checkpoint transformers maps to these
    classes) reachable by env var alone - see the module docstring.
    """
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
    """Public accessor for the loaded captioning model + processor."""
    return _get_model()


def _model_slug() -> str:
    return CAPTION_MODEL.replace("/", "__")


def _cache_path(image_bytes: bytes) -> Path:
    # Model name is part of the key, not just the directory, so a stale
    # caption from a different checkpoint can never be served.
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


# BLIP's decoder has a strong declarative tic - it opens 79% of captions on
# this test split with "there is a ...", "this is a ...", or "an image of ..."
# (measured, see docs/results/captioning.md). That prefix carries no
# information about the image, so as an input feature in Task 5.2 it is pure
# noise the decomposer would have to learn to ignore. This strips it.
#
# Deliberately NOT applied inside caption_images: the cache and
# captions.parquet hold the raw model output, so the stored data stays a
# faithful record of what the model said. Callers building model inputs opt
# in - keeps the artifact honest and the transformation reviewable.
_CAPTION_BOILERPLATE_RE = re.compile(
    r"^\s*(?:"
    r"there\s+(?:is|are)|this\s+is|here\s+is"      # contentless existential opener
    r"|(?:an?\s+)?(?:image|picture)\s+of"          # contentless framing
    r")\s+(?:an?\s+|two\s+|some\s+)?",
    re.IGNORECASE,
)


def strip_caption_boilerplate(caption: str) -> str:
    """Drop BLIP's contentless leading phrase, if present.

    "there is a poster with a picture of an orange" -> "poster with a picture
    of an orange"; "this is a picture of a wolf head" -> "wolf head". Applied
    repeatedly, since BLIP stacks these ("this is" + "a picture of").

    Note what is deliberately *not* stripped: "photo of", "photograph of",
    "painting of", "screenshot of", "3d rendering of". Those name the medium,
    which is real visual evidence and one of the fields being reconstructed -
    dropping them would destroy signal, not noise. Only the existential
    openers and the vacuous "image/picture of" framing go.

    Captions without the tic are returned unchanged.
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
    """Caption raw image bytes. Returns one caption per input, order preserved.

    Cache hits skip the model entirely; only the misses are batched through it.
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
