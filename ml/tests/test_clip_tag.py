import torch

from app.baselines.clip_tag import VOCABULARY, classify_batch
from app.embed import embed_images


def _solid_color_png(color: tuple[int, int, int]) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_classify_batch_returns_one_result_per_image():
    images = [_solid_color_png((200, 50, 50)), _solid_color_png((10, 200, 10))]
    targets = torch.from_numpy(embed_images(images))

    result = classify_batch(targets)

    assert len(result.items) == 2
    for item in result.items:
        assert set(item.tags) == set(VOCABULARY)
        assert all(item.tags[c] in VOCABULARY[c] for c in VOCABULARY)
        assert -1.0 <= item.cosine_similarity <= 1.0
    assert result.wall_clock_seconds > 0
