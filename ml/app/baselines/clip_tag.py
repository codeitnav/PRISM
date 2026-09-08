"""Task 4.2 - naive CLIP-tag baseline.

Zero-shot classifies an image against curated style/medium/lighting
vocabularies (via CLIP), then concatenates the top match from each category
into a naive "prompt". This is the simpler of the two baselines (see
app.baselines.pez for the other, more expensive one) - whatever the real
reconstruction method produces later has to beat both.

Plain CLIP zero-shot classification: no optimization loop, just embed a
small fixed vocabulary once and compare via cosine similarity. Much cheaper
than PEZ. Uses the same CLIP model as app.embed (ViT-B/32) via
app.embed.get_clip_model(), so results are directly comparable to PEZ and
to retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F

from app.embed import get_clip_model

VOCABULARY = {
    "style": [
        "photorealistic", "oil painting", "watercolor", "anime", "cyberpunk",
        "impressionist", "digital art", "concept art", "3d render", "pixel art",
        "surrealism", "pop art", "minimalist", "steampunk", "gothic",
        "psychedelic", "art deco", "baroque", "vaporwave", "abstract",
    ],
    "medium": [
        "photograph", "digital painting", "pencil sketch", "watercolor painting",
        "acrylic painting", "ink illustration", "pastel drawing", "charcoal drawing",
        "vector art", "mixed media collage", "sculpture", "line art", "3d render",
        "oil painting", "matte painting",
    ],
    "lighting": [
        "cinematic lighting", "soft lighting", "dramatic lighting", "neon lighting",
        "natural lighting", "studio lighting", "backlighting", "golden hour lighting",
        "moody lighting", "harsh lighting", "ambient lighting", "volumetric lighting",
        "rim lighting", "low light", "high contrast lighting",
    ],
}

_tokenizer = None
_tag_embeddings = None  # {category: (embeds tensor, tags list[str])}


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    return _tokenizer


def _get_tag_embeddings():
    global _tag_embeddings
    if _tag_embeddings is None:
        model, _ = get_clip_model()
        tokenizer = _get_tokenizer()
        _tag_embeddings = {}
        with torch.no_grad():
            for category, tags in VOCABULARY.items():
                embeds = F.normalize(model.encode_text(tokenizer(tags)), dim=-1)
                _tag_embeddings[category] = (embeds, tags)
    return _tag_embeddings


def _embed_text(text: str) -> torch.Tensor:
    model, _ = get_clip_model()
    with torch.no_grad():
        return F.normalize(model.encode_text(_get_tokenizer()([text])), dim=-1).squeeze(0)


@dataclass
class ClipTagItemResult:
    prompt: str
    tags: dict[str, str]
    cosine_similarity: float


@dataclass
class ClipTagBatchResult:
    items: list[ClipTagItemResult]
    wall_clock_seconds: float

    @property
    def mean_cosine_similarity(self) -> float:
        return sum(item.cosine_similarity for item in self.items) / len(self.items)


def classify_batch(target_embeddings: torch.Tensor) -> ClipTagBatchResult:
    """For each image embedding in `target_embeddings` (B, D), pick the
    top-matching tag per category (style/medium/lighting) and concatenate
    them into a naive prompt, scored against the original image embedding.
    """
    t0 = time.time()
    tag_embeddings = _get_tag_embeddings()
    targets = target_embeddings if torch.is_tensor(target_embeddings) else torch.tensor(target_embeddings)
    targets = targets.float()

    items = []
    for i in range(targets.shape[0]):
        target = targets[i]
        chosen = {}
        for category, (embeds, tags) in tag_embeddings.items():
            best_idx = (embeds @ target).argmax().item()
            chosen[category] = tags[best_idx]
        prompt = ", ".join(chosen[c] for c in VOCABULARY)
        similarity = float(_embed_text(prompt) @ target)
        items.append(ClipTagItemResult(prompt=prompt, tags=chosen, cosine_similarity=similarity))

    return ClipTagBatchResult(items=items, wall_clock_seconds=time.time() - t0)
