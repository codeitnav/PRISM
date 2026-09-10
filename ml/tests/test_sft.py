"""Tests for decomposition SFT dataset construction.

The leakage guards are the adversarial cases here: a self-match reaching the
training inputs would teach the model to copy its first candidate, and would
look entirely correct in a spot check.
"""

import json

import pytest

from app.schemas.structured_fields import StructuredFields
from app.sft import (
    PEZ_UNAVAILABLE,
    TARGET_KEY_ORDER,
    build_example,
    clean_target,
    filter_modifiers,
    render_input,
    render_target,
    validate_row,
)


def _fields(**over) -> StructuredFields:
    base = dict(
        subject="a red fox in a snowy forest",
        style="anime",
        medium="oil painting",
        lighting="golden hour",
        tone="serene",
        modifiers=["highly detailed", "8 k"],
        negative_constraints=["watermark"],
    )
    base.update(over)
    return StructuredFields(**base)


def _candidates(*pairs) -> list[dict]:
    return [
        {"id": i, "prompt": p, "similarity": s, "source": "diffusiondb"}
        for i, p, s in pairs
    ]


class TestFilterModifiers:
    def test_keeps_lexicon_recognised_modifiers(self):
        kept = filter_modifiers(["highly detailed", "8 k", "trending on artstation"])
        assert kept == ["highly detailed", "8 k", "trending on artstation"]

    def test_drops_scene_content_that_belongs_in_subject(self):
        # The failure mode measured at 0.26 field-level precision.
        assert filter_modifiers(["lava streams", "wearing a tanktop"]) == []

    def test_preserves_order_and_mixes(self):
        assert filter_modifiers(["lava streams", "8 k", "glass ceilings", "bokeh"]) == [
            "8 k",
            "bokeh",
        ]

    def test_handles_detokenized_corpus_spelling(self):
        # The corpus writes these with inserted spaces; the shared lexicon
        # normalizer has to apply here too, or real modifiers get dropped.
        assert filter_modifiers(["3 d render", "4 k"]) == ["3 d render", "4 k"]

    def test_clean_target_only_touches_modifiers(self):
        original = _fields(modifiers=["8 k", "lava streams"])
        cleaned = clean_target(original)
        assert cleaned.modifiers == ["8 k"]
        for key in ("subject", "style", "medium", "lighting", "tone", "negative_constraints"):
            assert getattr(cleaned, key) == getattr(original, key)

    def test_clean_target_does_not_mutate_its_input(self):
        original = _fields(modifiers=["8 k", "lava streams"])
        clean_target(original)
        assert original.modifiers == ["8 k", "lava streams"]


class TestRenderTarget:
    def test_is_compact_canonical_json_in_schema_key_order(self):
        text = render_target(_fields())
        assert " " not in text.split('"subject":')[0]  # no padding whitespace
        assert list(json.loads(text)) == TARGET_KEY_ORDER

    def test_round_trips_through_the_schema(self):
        fields = _fields()
        assert StructuredFields(**json.loads(render_target(fields))) == fields

    def test_nulls_and_empty_lists_are_explicit(self):
        parsed = json.loads(render_target(_fields(style=None, negative_constraints=[])))
        assert parsed["style"] is None
        assert parsed["negative_constraints"] == []

    def test_non_ascii_survives_unescaped(self):
        text = render_target(_fields(subject="a café at dusk"))
        assert "café" in text


class TestRenderInput:
    def test_contains_every_evidence_source(self):
        text = render_input("a fox", ["prompt one", "prompt two"], "pez words here")
        assert "a fox" in text
        assert "prompt one" in text and "prompt two" in text
        assert "pez words here" in text

    def test_numbers_the_retrieved_prompts(self):
        text = render_input("c", ["alpha", "beta"], None)
        assert "1. alpha" in text and "2. beta" in text

    def test_missing_pez_keeps_the_line_present(self):
        # Input shape must not change when PEZ is absent - otherwise a
        # PEZ-less row is a structurally different prompt.
        with_pez = render_input("c", ["a"], "pez")
        without = render_input("c", ["a"], None)
        assert PEZ_UNAVAILABLE in without
        assert with_pez.count("\n") == without.count("\n")

    def test_no_retrieved_prompts_is_stated_not_omitted(self):
        assert "(none)" in render_input("c", [], "pez")


class TestBuildExample:
    def test_drops_the_rows_own_self_match(self):
        """The core leakage guard: the FAISS index IS the train split."""
        candidates = _candidates(
            ("000001", "THE TRUE PROMPT", 1.0),
            ("000002", "neighbour one", 0.81),
            ("000003", "neighbour two", 0.79),
        )
        example = build_example(
            "000001", "train", "a caption", candidates, None,
            _fields(), "THE TRUE PROMPT", top_k=5,
        )
        assert example.self_match_dropped is True
        assert "THE TRUE PROMPT" not in example.retrieved_prompts
        assert example.retrieved_prompts == ["neighbour one", "neighbour two"]

    def test_records_when_no_self_match_was_present(self):
        example = build_example(
            "000009", "test", "a caption",
            _candidates(("000002", "neighbour", 0.7)), None,
            _fields(), "true prompt", top_k=5,
        )
        assert example.self_match_dropped is False

    def test_top_k_is_honoured_after_self_exclusion(self):
        candidates = _candidates(
            ("000001", "self", 1.0),
            *[(f"00000{i}", f"n{i}", 0.9 - i / 100) for i in range(2, 8)],
        )
        example = build_example(
            "000001", "train", "c", candidates, None, _fields(), "self", top_k=3
        )
        assert len(example.retrieved_prompts) == 3
        assert "self" not in example.retrieved_prompts

    def test_similarities_align_with_prompts(self):
        candidates = _candidates(("000002", "a", 0.8123456789), ("000003", "b", 0.5))
        example = build_example(
            "000001", "train", "c", candidates, None, _fields(), "t", top_k=5
        )
        assert example.retrieved_prompts == ["a", "b"]
        assert example.retrieved_similarities == [0.812346, 0.5]

    def test_target_modifiers_are_filtered_but_raw_is_kept(self):
        example = build_example(
            "000001", "train", "c", [], None,
            _fields(modifiers=["8 k", "lava streams"]), "t",
        )
        assert example.target.modifiers == ["8 k"]
        assert example.modifiers_raw == ["8 k", "lava streams"]


class TestValidateRow:
    def _row(self, **over) -> dict:
        example = build_example(
            "000001", "train", "a fox in snow",
            _candidates(("000002", "neighbour prompt", 0.8)),
            "pez words", _fields(), "the true prompt of this image",
        )
        row = example.to_row()
        row.update(over)
        return row

    def test_accepts_a_well_formed_row(self):
        assert validate_row(self._row()) == clean_target(_fields())

    @pytest.mark.parametrize(
        "key", ["id", "split", "input_text", "target_text", "target", "true_prompt"]
    )
    def test_rejects_a_row_missing_a_required_key(self, key):
        row = self._row()
        del row[key]
        with pytest.raises(ValueError, match=f"missing required key '{key}'"):
            validate_row(row)

    def test_rejects_unparseable_target_text(self):
        with pytest.raises(json.JSONDecodeError):
            validate_row(self._row(target_text="{not json"))

    def test_rejects_target_text_with_wrong_key_order(self):
        row = self._row()
        reordered = dict(reversed(list(json.loads(row["target_text"]).items())))
        row["target_text"] = json.dumps(reordered, separators=(",", ":"))
        with pytest.raises(ValueError, match="target_text keys"):
            validate_row(row)

    def test_rejects_target_text_violating_the_schema(self):
        row = self._row()
        broken = json.loads(row["target_text"])
        broken["subject"] = ""  # schema requires min_length=1
        row["target_text"] = json.dumps(broken, separators=(",", ":"))
        with pytest.raises(Exception):
            validate_row(row)

    def test_rejects_disagreement_between_target_text_and_target(self):
        row = self._row()
        row["target"] = {**row["target"], "style": "something else"}
        with pytest.raises(ValueError, match="disagree"):
            validate_row(row)

    def test_rejects_non_canonical_rendering(self):
        row = self._row()
        parsed = json.loads(row["target_text"])
        row["target_text"] = json.dumps(parsed, indent=2)  # padded, not canonical
        row["target"] = parsed
        with pytest.raises(ValueError, match="canonical"):
            validate_row(row)

    def test_rejects_a_row_that_retrieves_itself(self):
        """The precise leakage invariant - identity, not string similarity."""
        row = self._row()
        row["input"]["retrieved_ids"] = [row["id"], "000002"]
        with pytest.raises(ValueError, match="retrieves itself"):
            validate_row(row)

    def test_accepts_a_neighbour_whose_prompt_contains_the_true_prompt(self):
        """Legitimate retrieval, not leakage: a *different* image can carry a
        superset prompt, and that happens at inference too. Rejecting it would
        make training inputs weaker than production."""
        example = build_example(
            "000028", "train", "batman in a suit",
            _candidates(("000032", "the true prompt plus extra style tags", 0.79)),
            None, _fields(), "the true prompt",
        )
        row = example.to_row()
        assert row["true_prompt_in_retrieved"] is True
        validate_row(row)  # must not raise

    def test_flags_containment_only_when_present(self):
        example = build_example(
            "000001", "train", "c",
            _candidates(("000002", "a totally unrelated prompt", 0.5)),
            None, _fields(), "the true prompt",
        )
        assert example.to_row()["true_prompt_in_retrieved"] is False


class TestCaptionHandling:
    def test_input_uses_the_stripped_caption_not_the_raw_one(self):
        """5.1 measured 79% of captions carrying a contentless opener; 5.2 is
        the documented opt-in point for removing it."""
        example = build_example(
            "000001", "train", "there is a poster with an orange on it",
            _candidates(("000002", "n", 0.7)), None, _fields(), "t",
        )
        assert example.caption == "poster with an orange on it"
        assert example.caption_raw == "there is a poster with an orange on it"
        assert "there is a poster" not in example.to_row()["input_text"]

    def test_raw_caption_is_retained_for_ablation(self):
        row = build_example(
            "000001", "train", "this is a picture of a wolf",
            [], None, _fields(), "t",
        ).to_row()
        assert row["input"]["caption_raw"] == "this is a picture of a wolf"
        assert row["input"]["caption"] == "wolf"

    def test_medium_bearing_caption_is_left_intact(self):
        # "painting of" names the medium - real signal, must survive.
        example = build_example(
            "000001", "train", "painting of two women playing tennis",
            [], None, _fields(), "t",
        )
        assert example.caption == "painting of two women playing tennis"
