"""Tests for the decomposition model wrapper.

Exercises the parts that fail silently - output parsing, prompt construction,
adapter and control switching - against stubs, plus a guard on the tokenizer
property that disqualified an earlier base model.
"""

import json

import pytest

from app import decompose as dec
from app.schemas.structured_fields import StructuredFields

VALID_JSON = (
    '{"subject":"a red fox","style":"anime","medium":"oil painting",'
    '"lighting":"golden hour","modifiers":["8 k"],"tone":"serene",'
    '"negative_constraints":[]}'
)


class TestParseOutput:
    def test_parses_clean_json(self):
        result = dec.parse_output(VALID_JSON)
        assert result.is_valid
        assert result.fields.subject == "a red fox"
        assert result.parse_error is None

    def test_tolerates_surrounding_prose_and_code_fences(self):
        for wrapped in (
            f"Here you go:\n{VALID_JSON}\nHope that helps!",
            f"```json\n{VALID_JSON}\n```",
            f"   {VALID_JSON}   ",
        ):
            assert dec.parse_output(wrapped).is_valid

    def test_rejects_output_with_no_json(self):
        result = dec.parse_output("subject, style, medium, lighting")
        assert not result.is_valid
        assert "no JSON object found" in result.parse_error

    def test_rejects_truncated_json(self):
        """The observed real failure mode: output cut off before the closing brace."""
        result = dec.parse_output('{"subject":"a fox","style":"ani')
        assert not result.is_valid
        assert "no JSON object found" in result.parse_error

    def test_rejects_malformed_json(self):
        result = dec.parse_output('{"subject": "a fox",,}')
        assert not result.is_valid
        assert "invalid JSON" in result.parse_error

    def test_rejects_non_object_json(self):
        result = dec.parse_output('["a fox"]')
        assert not result.is_valid

    def test_rejects_extra_keys(self):
        payload = json.loads(VALID_JSON)
        payload["surprise"] = "nope"
        result = dec.parse_output(json.dumps(payload))
        assert not result.is_valid
        assert "schema violation" in result.parse_error

    def test_rejects_empty_subject(self):
        payload = json.loads(VALID_JSON)
        payload["subject"] = ""
        result = dec.parse_output(json.dumps(payload))
        assert not result.is_valid
        assert "schema violation" in result.parse_error

    def test_keeps_raw_output_on_failure_for_debugging(self):
        result = dec.parse_output("total garbage")
        assert result.raw_output == "total garbage"


class TestFewshot:
    def test_prepends_format_examples(self):
        out = dec.to_fewshot("Caption: a fox")
        assert out.endswith("Caption: a fox")
        assert out.count('"subject"') >= 2

    def test_fewshot_examples_are_themselves_schema_valid(self):
        """A malformed demonstration would teach the control the wrong format."""
        found = 0
        for line in dec.FEWSHOT_PREAMBLE.splitlines():
            line = line.strip()
            if line.startswith("JSON: "):
                StructuredFields(**json.loads(line[len("JSON: "):]))
                found += 1
        assert found == 2


class TestTokenizerCapability:
    def test_base_tokenizer_can_round_trip_json_braces(self):
        """The base model must be able to represent JSON braces.

        T5-family tokenizers omit `{` and `}`, mapping both to <unk>, which
        makes parseable JSON output impossible regardless of training. Any
        change to DECOMPOSER_BASE must not reintroduce that.
        """
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(dec.BASE_MODEL)
        assert tokenizer.decode(
            tokenizer(VALID_JSON, add_special_tokens=False).input_ids
        ) == VALID_JSON
        for brace in ("{", "}"):
            token_id = tokenizer.convert_tokens_to_ids(brace)
            assert token_id != tokenizer.unk_token_id, f"{brace!r} maps to <unk>"


class TestLoadModel:
    def test_missing_adapter_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train it first"):
            dec.load_model(adapter_dir=tmp_path / "nope", use_adapter=True)

    def test_control_path_does_not_require_an_adapter(self, monkeypatch):
        """use_adapter=False must work with no adapter on disk at all."""
        calls = {}

        def fake_from_pretrained(name, **kw):
            calls["model"] = name
            return _StubModel()

        monkeypatch.setattr(dec.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
        monkeypatch.setattr(dec.AutoTokenizer, "from_pretrained", lambda n: _StubTokenizer())
        dec._cache.clear()
        model, tokenizer = dec.load_model(adapter_dir=None, use_adapter=False)
        assert calls["model"] == dec.BASE_MODEL
        dec._cache.clear()


class _StubTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "right"
    eos_token = "</s>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "USER: " + messages[0]["content"] + "\nASSISTANT:"

    def __call__(self, texts, **kw):
        import torch

        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.zeros((n, 5), dtype=torch.long),
            "attention_mask": torch.ones((n, 5), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return VALID_JSON


class _StubModel:
    def eval(self):
        return self

    def generate(self, input_ids=None, attention_mask=None, **kw):
        import torch

        return torch.zeros((input_ids.shape[0], input_ids.shape[1] + 3), dtype=torch.long)


class TestDecomposeBatch:
    @pytest.fixture
    def stubbed(self, monkeypatch):
        tokenizer = _StubTokenizer()
        monkeypatch.setattr(dec, "load_model", lambda **kw: (_StubModel(), tokenizer))
        return tokenizer

    def test_returns_one_result_per_input(self, stubbed):
        results = dec.decompose_batch(["a", "b", "c"], batch_size=2)
        assert len(results) == 3
        assert all(r.is_valid for r in results)

    def test_restores_tokenizer_padding_side(self, stubbed):
        """Left padding is required for decoder-only generation, but the
        trainer needs right padding - leaking the mutation would corrupt it."""
        assert stubbed.padding_side == "right"
        dec.decompose_batch(["a"], batch_size=1)
        assert stubbed.padding_side == "right"

    def test_padding_side_is_left_during_generation(self, monkeypatch):
        seen = {}
        tokenizer = _StubTokenizer()

        class _Capture(_StubModel):
            def generate(self, input_ids=None, attention_mask=None, **kw):
                seen["side"] = tokenizer.padding_side
                return super().generate(input_ids=input_ids, attention_mask=attention_mask)

        monkeypatch.setattr(dec, "load_model", lambda **kw: (_Capture(), tokenizer))
        dec.decompose_batch(["a"], batch_size=1)
        assert seen["side"] == "left"


def test_build_prompt_uses_the_chat_template():
    tokenizer = _StubTokenizer()
    prompt = dec.build_prompt(tokenizer, "Caption: a fox")
    assert "Caption: a fox" in prompt
    assert prompt.endswith("ASSISTANT:")
