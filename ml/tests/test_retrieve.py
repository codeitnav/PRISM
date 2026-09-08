from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
DATA_DIR = Path("/data")


def test_retrieve_returns_candidates_for_known_training_image():
    image_path = DATA_DIR / "diffusiondb" / "images" / "000002.png"
    with open(image_path, "rb") as f:
        res = client.post(
            "/internal/retrieve",
            files={"image": ("000002.png", f, "image/png")},
            params={"top_k": 10},
        )

    assert res.status_code == 200
    candidates = res.json()
    assert len(candidates) == 10
    # A training image queried against its own index should retrieve itself at rank 1.
    assert candidates[0]["id"] == "000002"
    assert candidates[0]["similarity"] > 0.99
    for field in ("id", "prompt", "source", "similarity"):
        assert field in candidates[0]
