from dataclasses import dataclass

from app.eval import run_eval


@dataclass
class _FakeItem:
    prompt: str
    cosine_similarity: float


@dataclass
class _FakeResult:
    items: list
    wall_clock_seconds: float


def _fake_good_reconstructor(_embeddings):
    return _FakeResult(
        items=[_FakeItem(prompt="a red bicycle on a street", cosine_similarity=0.9)],
        wall_clock_seconds=2.0,
    )


def _fake_bad_reconstructor(_embeddings):
    return _FakeResult(
        items=[_FakeItem(prompt="xyz random unrelated words", cosine_similarity=0.1)],
        wall_clock_seconds=1.0,
    )


def test_run_eval_scores_each_reconstructor():
    rows = run_eval(
        {"good": _fake_good_reconstructor, "bad": _fake_bad_reconstructor},
        image_ids=["000001"],
        original_prompts=["a red bicycle parked on a city street"],
        target_embeddings=None,
    )

    assert len(rows) == 2
    good_row = next(r for r in rows if r.reconstructor == "good")
    bad_row = next(r for r in rows if r.reconstructor == "bad")

    # The closer paraphrase should score higher on BERTScore than unrelated words.
    assert good_row.bertscore_f1 > bad_row.bertscore_f1
    assert good_row.clip_score == 0.9
    assert good_row.latency_seconds == 2.0
    assert 0.0 <= good_row.bertscore_f1 <= 1.0
