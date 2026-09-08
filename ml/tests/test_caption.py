"""Task 5.1 - tests for the captioning stage.

The model itself is ~1.9GB, so these tests exercise the parts that carry the
real risk of silent breakage - cache keying, batching, order preservation,
architecture dispatch - against a stub model, and keep exactly one test that
loads the real checkpoint (marked slow-ish but kept, because a broken
processor/generate call would otherwise only surface during a long batch run).
"""

from io import BytesIO

import pytest
from PIL import Image

from app import caption as caption_mod


def _png(color: tuple[int, int, int], size: int = 64) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (size, size), color=color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(caption_mod, "CAPTION_CACHE_DIR", tmp_path / "captions")
    return tmp_path


class _StubProcessor:
    """Mimics the HF processor surface caption_images actually uses."""

    def __call__(self, images, return_tensors=None):
        self.last_batch_size = len(images)

        class _Inputs(dict):
            def to(self, _device):
                return self

        return _Inputs(pixel_values=list(range(len(images))))

    def decode(self, seq, skip_special_tokens=True):
        return f"caption for {seq}"


class _StubModel:
    def __init__(self):
        self.generate_calls = 0
        self.batch_sizes = []

    def generate(self, pixel_values=None, **kwargs):
        self.generate_calls += 1
        self.batch_sizes.append(len(pixel_values))
        return list(pixel_values)


@pytest.fixture
def stub_model(monkeypatch):
    model, processor = _StubModel(), _StubProcessor()
    monkeypatch.setattr(caption_mod, "_get_model", lambda: (model, processor))
    return model


def test_caption_images_returns_one_caption_per_image(isolated_cache, stub_model):
    images = [_png((200, 50, 50)), _png((10, 200, 10)), _png((10, 10, 200))]
    captions = caption_mod.caption_images(images)
    assert len(captions) == 3
    assert all(isinstance(c, str) and c for c in captions)


def test_caption_images_caches_by_content_hash(isolated_cache, stub_model):
    image = _png((123, 45, 67))

    first = caption_mod.caption_images([image])
    assert stub_model.generate_calls == 1

    # Same bytes again -> served from disk, model untouched.
    second = caption_mod.caption_images([image])
    assert second == first
    assert stub_model.generate_calls == 1


def test_cache_key_includes_model_name(isolated_cache, stub_model, monkeypatch):
    # A different checkpoint must not be served a cached caption from another.
    image = _png((9, 9, 9))
    caption_mod.caption_images([image])
    path_a = caption_mod._cache_path(image)

    monkeypatch.setattr(caption_mod, "CAPTION_MODEL", "Salesforce/blip2-opt-2.7b")
    path_b = caption_mod._cache_path(image)

    assert path_a != path_b
    assert not path_b.exists()


def test_only_uncached_images_are_sent_to_the_model(isolated_cache, stub_model):
    cached, fresh = _png((1, 2, 3)), _png((4, 5, 6))
    caption_mod.caption_images([cached])
    assert stub_model.batch_sizes == [1]

    caption_mod.caption_images([cached, fresh])
    # Second call batches only the miss, not both.
    assert stub_model.batch_sizes == [1, 1]


def test_mixed_cache_hits_preserve_input_order(isolated_cache, stub_model):
    a, b, c = _png((1, 1, 1)), _png((2, 2, 2)), _png((3, 3, 3))
    baseline = caption_mod.caption_images([a, b, c])

    # Warm only the middle one, then request all three in the same order.
    caption_mod.caption_images([b])
    assert caption_mod.caption_images([a, b, c]) == baseline


def test_batching_respects_batch_size(isolated_cache, stub_model):
    images = [_png((i, i, i)) for i in range(10, 17)]  # 7 distinct images
    caption_mod.caption_images(images, batch_size=3)
    assert stub_model.batch_sizes == [3, 3, 1]


def test_model_class_dispatches_on_config_model_type(monkeypatch):
    from transformers import BlipForConditionalGeneration, Blip2ForConditionalGeneration

    class _Cfg:
        def __init__(self, model_type):
            self.model_type = model_type

    monkeypatch.setattr(
        caption_mod.AutoConfig, "from_pretrained", staticmethod(lambda _n: _Cfg("blip"))
    )
    assert caption_mod._model_class("x") is BlipForConditionalGeneration

    monkeypatch.setattr(
        caption_mod.AutoConfig, "from_pretrained", staticmethod(lambda _n: _Cfg("blip-2"))
    )
    assert caption_mod._model_class("x") is Blip2ForConditionalGeneration


def test_model_class_rejects_unsupported_architecture(monkeypatch):
    class _Cfg:
        model_type = "llava"

    monkeypatch.setattr(
        caption_mod.AutoConfig, "from_pretrained", staticmethod(lambda _n: _Cfg())
    )
    with pytest.raises(ValueError, match="expected 'blip' or 'blip-2'"):
        caption_mod._model_class("llava-hf/llava-1.5-7b-hf")


def test_real_model_captions_a_real_image(isolated_cache):
    """Loads the actual checkpoint - the one end-to-end guard on this stage."""
    image = _png((240, 180, 60), size=224)
    captions = caption_mod.caption_images([image])
    assert len(captions) == 1
    assert captions[0].strip(), "model returned an empty caption"
    assert len(captions[0].split()) >= 2


class TestStripCaptionBoilerplate:
    """BLIP's contentless opener is noise as a 5.2 input feature; the medium
    words that look similar are signal and must survive."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("there is a poster with an orange on it", "poster with an orange on it"),
            ("there are two ships in the ocean", "ships in the ocean"),
            ("here is a small dog", "small dog"),
            # BLIP stacks openers - has to strip both.
            ("this is a picture of a carved wolf head", "carved wolf head"),
            ("an image of a game with characters", "game with characters"),
        ],
    )
    def test_strips_contentless_openers(self, raw, expected):
        assert caption_mod.strip_caption_boilerplate(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            # Each names the medium - real visual evidence, and one of the
            # fields being reconstructed. Must not be stripped.
            "photo of a great white shark",
            "black and white photo of a shark with its mouth open",
            "photograph of a city street at night",
            "painting of two women playing tennis",
            "screenshot of a game being played",
            "3d rendering of a creepy mask",
            # No opener at all.
            "a red fox in a snowy forest",
        ],
    )
    def test_preserves_medium_bearing_and_clean_captions(self, raw):
        assert caption_mod.strip_caption_boilerplate(raw) == raw

    def test_strips_medium_opener_only_after_an_existential(self):
        # "there is a 3d rendering of X" -> the medium survives the strip.
        assert (
            caption_mod.strip_caption_boilerplate("there is a 3d rendering of a mask")
            == "3d rendering of a mask"
        )

    def test_never_returns_empty(self):
        assert caption_mod.strip_caption_boilerplate("there is a") == "a"
        assert caption_mod.strip_caption_boilerplate("   ") == ""

    def test_caption_images_stores_raw_output_not_stripped(self, isolated_cache, monkeypatch):
        """The cache/parquet must stay a faithful record of what the model said."""
        raw = "there is a dog on a couch"

        class _P:
            def __call__(self, images, return_tensors=None):
                class _I(dict):
                    def to(self, _d):
                        return self

                return _I(pixel_values=[0] * len(images))

            def decode(self, seq, skip_special_tokens=True):
                return raw

        class _M:
            def generate(self, pixel_values=None, **kw):
                return list(pixel_values)

        monkeypatch.setattr(caption_mod, "_get_model", lambda: (_M(), _P()))
        assert caption_mod.caption_images([_png((5, 5, 5))]) == [raw]


class TestCaptionEndpoint:
    """POST /internal/caption - the stage endpoint the backend calls."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    def test_returns_caption_and_model(self, client, monkeypatch):
        monkeypatch.setattr(
            "app.routes.caption.caption_images", lambda imgs: ["a dog on a couch"]
        )
        r = client.post("/internal/caption", files={"image": ("x.png", _png((1, 2, 3)))})
        assert r.status_code == 200
        body = r.json()
        assert body["caption"] == "a dog on a couch"
        assert body["model"]

    def test_empty_upload_is_422(self, client):
        r = client.post("/internal/caption", files={"image": ("x.png", b"")})
        assert r.status_code == 422

    def test_missing_field_is_422(self, client):
        assert client.post("/internal/caption").status_code == 422

    def test_undecodable_image_is_422_not_503(self, client):
        """Bad input must not look transient - the backend (Task 3.1) retries 5xx."""
        r = client.post("/internal/caption", files={"image": ("x.png", b"not-an-image")})
        assert r.status_code == 422
        assert "decodable" in r.json()["detail"]

    def test_model_failure_is_503(self, client, monkeypatch):
        def _boom(_imgs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr("app.routes.caption.caption_images", _boom)
        r = client.post("/internal/caption", files={"image": ("x.png", _png((1, 1, 1)))})
        assert r.status_code == 503
        assert "unavailable" in r.json()["detail"]
