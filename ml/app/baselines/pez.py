"""Task 4.1 - PEZ baseline ("Hard Prompts Made Easy", Wen et al.).

Optimizes a short sequence of REAL CLIP vocabulary tokens so their text
embedding matches a target image's CLIP embedding as closely as possible -
a discrete, human-readable "hard prompt" approximation, with no training
data or labels. This is the project's baseline for the reconstruction task
(see Section 3 of the project brief and Task 4.1 of the roadmap): whatever
the real structured-decomposition model produces later (Task 5.x) has to
beat this.

Algorithm: maintain a continuous embedding per soft token; each step,
project it to its nearest real vocabulary embedding for the forward pass
(straight-through estimator - the discrete projection is used for the
forward pass, but gradients flow to the continuous embedding as if the
projection were the identity function), score the resulting text embedding
against the target image embedding, and take an optimizer step on the
continuous embeddings. The final projection's tokens ARE the hard prompt.

Uses the same CLIP model as app.embed (ViT-B/32, see that module's scope
note) via app.embed.get_clip_model(), so results share the same embedding
space as retrieval and everything else in the system.

Runs all target images in ONE BATCH rather than one at a time: on this
CPU-only machine, per-iteration cost barely increases with batch size (the
transformer processes the same fixed 77-token context regardless), so
batching is a large, nearly-free speedup - see the benchmark note in
docs/results/pez_baseline.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import open_clip
import torch
import torch.nn.functional as F

from app.embed import get_clip_model

NUM_SOFT_TOKENS = 8
ITERATIONS = 100  # reduced from the paper's typical few-hundred-to-thousand default - CPU-only, see scope note above
LEARNING_RATE = 0.1
CONTEXT_LENGTH = 77
SOT_TOKEN_ID = 49406
EOT_TOKEN_ID = 49407
NUM_SPECIAL_TOKEN_IDS = 3  # ids 0, 1, 2 are reserved (e.g. pad) - skip when picking random init tokens

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    return _tokenizer


def _encode_from_embeddings(model, token_embeds: torch.Tensor, eot_position: int) -> torch.Tensor:
    """Same computation as CLIP.encode_text(), but starting from token
    EMBEDDINGS instead of token ids - lets us feed in our optimized soft
    tokens. NOTE: this open_clip version's Transformer.forward() handles the
    batch-first/seq-first transpose internally, so - unlike the classic
    OpenAI CLIP reference code - callers must NOT permute before calling it.
    """
    cast_dtype = model.transformer.get_cast_dtype()
    x = token_embeds.to(cast_dtype) + model.positional_embedding.to(cast_dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    pooled = x[:, eot_position, :] @ model.text_projection
    return F.normalize(pooled, dim=-1)


@dataclass
class PezItemResult:
    prompt: str
    cosine_similarity: float


@dataclass
class PezBatchResult:
    items: list[PezItemResult]
    wall_clock_seconds: float
    iterations: int
    num_tokens: int

    @property
    def mean_cosine_similarity(self) -> float:
        return sum(item.cosine_similarity for item in self.items) / len(self.items)


def reconstruct_batch(
    target_embeddings: torch.Tensor,
    num_tokens: int = NUM_SOFT_TOKENS,
    iterations: int = ITERATIONS,
    lr: float = LEARNING_RATE,
) -> PezBatchResult:
    """Optimize `num_tokens` real vocabulary tokens per image to approximate
    each of `target_embeddings` (shape (B, D), L2-normalized CLIP image
    embeddings - e.g. from app.embed.embed_images).
    """
    t0 = time.time()
    model, _ = get_clip_model()
    tokenizer = _get_tokenizer()
    embedding_matrix = model.token_embedding.weight  # (vocab_size, d_model), frozen
    vocab_normed = F.normalize(embedding_matrix, dim=-1)  # precomputed once, not per-iteration

    targets = target_embeddings if torch.is_tensor(target_embeddings) else torch.tensor(target_embeddings)
    targets = targets.float()
    batch_size, d_model = targets.shape[0], embedding_matrix.shape[1]
    eot_position = 1 + num_tokens  # index 0 = SOT, then soft tokens, then EOT

    # PEZ initializes from real (random) vocabulary embeddings rather than
    # noise - keeps the optimization inside the manifold of real tokens.
    vocab_size = embedding_matrix.shape[0]
    init_ids = torch.randint(NUM_SPECIAL_TOKEN_IDS, vocab_size - 1, (batch_size, num_tokens))
    soft_embeds = embedding_matrix[init_ids].clone().detach().requires_grad_(True)

    sot_embed = embedding_matrix[SOT_TOKEN_ID].detach().view(1, 1, -1).expand(batch_size, 1, -1)
    eot_embed = embedding_matrix[EOT_TOKEN_ID].detach().view(1, 1, -1).expand(batch_size, 1, -1)
    pad_embeds = torch.zeros(batch_size, CONTEXT_LENGTH - num_tokens - 2, d_model)

    optimizer = torch.optim.Adam([soft_embeds], lr=lr)

    def project_to_nearest(embeds: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(embeds, dim=-1)
        sims = torch.einsum("btd,vd->btv", normed, vocab_normed)
        nearest_ids = sims.argmax(dim=-1)
        return nearest_ids, embedding_matrix[nearest_ids]

    for _ in range(iterations):
        optimizer.zero_grad()
        _, projected = project_to_nearest(soft_embeds)
        forward_embeds = soft_embeds + (projected - soft_embeds).detach()  # straight-through estimator

        seq = torch.cat([sot_embed, forward_embeds, eot_embed, pad_embeds], dim=1)
        text_embeds = _encode_from_embeddings(model, seq, eot_position)
        loss = (1.0 - (text_embeds * targets).sum(dim=-1)).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_ids, projected = project_to_nearest(soft_embeds)
        seq = torch.cat([sot_embed, projected, eot_embed, pad_embeds], dim=1)
        final_text_embeds = _encode_from_embeddings(model, seq, eot_position)
        similarities = (final_text_embeds * targets).sum(dim=-1)

    items = [
        PezItemResult(
            prompt=tokenizer.decode(final_ids[i].tolist()),
            cosine_similarity=float(similarities[i]),
        )
        for i in range(batch_size)
    ]

    return PezBatchResult(
        items=items,
        wall_clock_seconds=time.time() - t0,
        iterations=iterations,
        num_tokens=num_tokens,
    )
