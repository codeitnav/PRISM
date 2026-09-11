from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_decompose_returns_schema_valid_fields_for_a_caption():
    res = client.post(
        "/internal/decompose",
        json={
            "caption": "a red fox standing in a snowy forest",
            "retrieved_prompts": ["a fox in the snow, oil painting, golden hour"],
            "pez_prompt": None,
        },
    )

    assert res.status_code == 200
    body = res.json()
    for field in ("subject", "style", "medium", "lighting", "modifiers", "tone", "negative_constraints"):
        assert field in body
    assert isinstance(body["subject"], str) and body["subject"]


def test_decompose_rejects_an_empty_caption():
    res = client.post("/internal/decompose", json={"caption": "   "})
    assert res.status_code == 422
