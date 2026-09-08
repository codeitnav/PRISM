import torch

from app.baselines.pez import reconstruct_batch
from app.embed import embed_images


def _solid_color_png(color: tuple[int, int, int]) -> bytes:
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (64, 64), color=color).save(buf, format="PNG")
    return buf.getvalue()


def test_reconstruct_batch_returns_one_result_per_image():
    images = [_solid_color_png((200, 50, 50)), _solid_color_png((10, 200, 10))]
    targets = torch.from_numpy(embed_images(images))

    result = reconstruct_batch(targets, num_tokens=4, iterations=3)

    assert len(result.items) == 2
    for item in result.items:
        assert isinstance(item.prompt, str) and len(item.prompt) > 0
        assert -1.0 <= item.cosine_similarity <= 1.0
    assert result.wall_clock_seconds > 0
    assert -1.0 <= result.mean_cosine_similarity <= 1.0
