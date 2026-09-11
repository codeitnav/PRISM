"""Tests for decompose_reliably's retry-then-fallback guarantee.

The serving pipeline must never receive an unparseable result, even though
decompose_batch (scored raw by the eval harness) can fail to parse. These
tests stub decompose_batch directly, so they exercise the retry/fallback
control flow without needing the real model.
"""

from app import decompose as dec


def _result(subject=None, valid=True):
    if valid:
        return dec.DecompositionResult(
            raw_output="...", fields=dec.StructuredFields(subject=subject or "a fox")
        )
    return dec.DecompositionResult(raw_output="not json", fields=None, parse_error="no JSON object found")


class TestDecomposeReliably:
    def test_returns_model_output_when_valid_first_try(self, monkeypatch):
        monkeypatch.setattr(dec, "decompose_batch", lambda texts, **kw: [_result("a red fox") for _ in texts])
        [result] = dec.decompose_reliably(["Caption: a red fox\n\nJSON:"])
        assert result.subject == "a red fox"

    def test_retries_only_the_failed_rows(self, monkeypatch):
        calls = []

        def fake_decompose_batch(texts, **kw):
            calls.append(list(texts))
            if len(calls) == 1:
                # Row 0 fails to parse, row 1 succeeds.
                return [_result(valid=False), _result("a cat")]
            return [_result("a fox, retried")]

        monkeypatch.setattr(dec, "decompose_batch", fake_decompose_batch)
        results = dec.decompose_reliably(
            ["Caption: a fox\n\nJSON:", "Caption: a cat\n\nJSON:"], max_retries=1
        )

        assert results[0].subject == "a fox, retried"
        assert results[1].subject == "a cat"
        assert len(calls) == 2
        assert len(calls[1]) == 1  # only the failed row was retried
        assert calls[1][0].endswith("nothing else.)")

    def test_falls_back_to_the_caption_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr(dec, "decompose_batch", lambda texts, **kw: [_result(valid=False) for _ in texts])
        [result] = dec.decompose_reliably(
            ["Some instruction\n\nCaption: a lonely lighthouse\n\nJSON:"], max_retries=1
        )
        assert result.subject == "a lonely lighthouse"
        assert result.style is None
        assert result.modifiers == []

    def test_fallback_is_schema_valid_without_a_caption_line(self, monkeypatch):
        monkeypatch.setattr(dec, "decompose_batch", lambda texts, **kw: [_result(valid=False) for _ in texts])
        [result] = dec.decompose_reliably(["no caption line here"], max_retries=0)
        assert result.subject
