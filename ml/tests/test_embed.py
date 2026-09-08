from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app import embed


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the embedding cache at a fresh temp dir for every test."""
    monkeypatch.setattr(embed, "IMAGE_CACHE_DIR", tmp_path / "images")
    monkeypatch.setattr(embed, "TEXT_CACHE_DIR", tmp_path / "texts")


def _solid_color_png(color: tuple[int, int, int]) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_embed_texts_is_deterministic():
    vec_a = embed.embed_texts(["a cat sitting on a mat"])
    embed.TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for f in embed.TEXT_CACHE_DIR.glob("*.npy"):
        f.unlink()  # force a real recompute, not a cache hit
    vec_b = embed.embed_texts(["a cat sitting on a mat"])
    assert np.allclose(vec_a, vec_b, atol=1e-5)


def test_embed_texts_normalized():
    vecs = embed.embed_texts(["a red bicycle", "a blue sky"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_images_is_deterministic():
    png = _solid_color_png((200, 50, 50))
    vec_a = embed.embed_images([png])
    embed.IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for f in embed.IMAGE_CACHE_DIR.glob("*.npy"):
        f.unlink()  # force a real recompute, not a cache hit
    vec_b = embed.embed_images([png])
    assert np.allclose(vec_a, vec_b, atol=1e-5)


def test_embed_images_normalized():
    vecs = embed.embed_images([_solid_color_png((10, 200, 10)), _solid_color_png((10, 10, 200))])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_embed_images_cache_hit_matches_fresh_compute():
    png = _solid_color_png((80, 80, 80))
    vec_fresh = embed.embed_images([png])
    vec_cached = embed.embed_images([png])  # should hit the cache this time
    assert np.allclose(vec_fresh, vec_cached, atol=1e-8)
