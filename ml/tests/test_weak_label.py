"""Task 1.3 - regression tests for the weak structured labeler."""

import pytest

from app.schemas.structured_fields import StructuredFields
from app.weak_label import (
    clean_segment,
    match_segment,
    mine_lexicon_candidates,
    split_prompt,
    weak_label,
    weak_label_row,
)


def test_clean_segment_strips_weight_and_credit_syntax():
    assert clean_segment("(masterpiece:1.4)") == "masterpiece"
    assert clean_segment("  [blurry]  ") == "blurry"
    assert clean_segment("by Greg Rutkowski") == "Greg Rutkowski"
    assert clean_segment("in the style of Van Gogh") == "Van Gogh"


def test_split_prompt_separates_explicit_negatives():
    positives, negatives = split_prompt("a castle on a hill --neg blurry, watermark")
    assert positives == ["a castle on a hill"]
    assert negatives == ["blurry", "watermark"]


def test_split_prompt_treats_inline_no_without_as_negative():
    positives, negatives = split_prompt("a forest path, no people, without text")
    assert positives == ["a forest path"]
    assert negatives == ["people", "text"]


def test_split_prompt_handles_midjourney_separators():
    positives, _ = split_prompt("a raccoon | 35mm photograph :: cinematic lighting")
    assert positives == ["a raccoon", "35mm photograph", "cinematic lighting"]


def test_match_segment_prefers_longest_phrase():
    # "volumetric lighting" must win over the shorter "lighting"-family entries.
    assert match_segment("volumetric lighting")["lighting"] == "volumetric lighting"


def test_weak_label_routes_each_category():
    fields = weak_label(
        "a red fox in a snowy forest, oil painting, golden hour, "
        "highly detailed, serene, no watermark"
    )
    assert fields.subject == "a red fox in a snowy forest"
    assert fields.medium == "oil painting"
    assert fields.lighting == "golden hour"
    assert fields.tone == "serene"
    assert "highly detailed" in fields.modifiers
    assert fields.negative_constraints == ["watermark"]


def test_weak_label_subject_segment_still_donates_style_phrase():
    fields = weak_label("still from a studio ghibli movie about a forest spirit")
    assert fields.style == "studio ghibli"
    assert "forest spirit" in fields.subject


def test_weak_label_drops_numeric_weight_leftovers():
    fields = weak_label("a lighthouse in a storm | cinematic lighting :: 2")
    assert "2" not in fields.modifiers
    assert fields.lighting == "cinematic lighting"


def test_weak_label_always_produces_a_nonempty_subject():
    # A prompt made only of quality boilerplate still has to satisfy the
    # schema's required, min-length-1 subject.
    fields = weak_label("8k, highly detailed, trending on artstation")
    assert fields.subject
    assert isinstance(fields, StructuredFields)


@pytest.mark.parametrize("prompt", ["", "   ", ",,,", "--neg blurry"])
def test_weak_label_survives_degenerate_prompts(prompt):
    # Empty/near-empty prompts can't satisfy the schema, so they must raise
    # loudly rather than silently emitting an invalid record downstream.
    with pytest.raises(Exception):
        weak_label(prompt)


def test_weak_label_output_validates_against_schema():
    fields = weak_label("a knight, digital painting, dramatic lighting, 4k")
    # Round-trip through the Pydantic model to confirm extra="forbid" holds.
    assert StructuredFields(**fields.model_dump()) == fields


def test_llm_seed_label_overrides_rule_output():
    seeds = {
        "000001": {
            "subject": "a hand-labeled subject",
            "style": "baroque",
            "medium": None,
            "lighting": None,
            "modifiers": [],
            "tone": None,
            "negative_constraints": [],
        }
    }
    fields, provenance = weak_label_row("a totally different prompt", "000001", seeds)
    assert provenance == "llm_seed"
    assert fields.subject == "a hand-labeled subject"

    fields, provenance = weak_label_row("a red barn, oil painting", "000002", seeds)
    assert provenance == "rule"
    assert fields.medium == "oil painting"


def test_mine_lexicon_candidates_surfaces_unclassified_segments():
    prompts = [
        "a barn, oil painting, wonkyterm",
        "a shed, oil painting, wonkyterm",
        "a hut, watercolor, wonkyterm",
    ]
    candidates = dict(mine_lexicon_candidates(prompts))
    assert candidates.get("wonkyterm") == 3
    # Already-classified vocabulary must not be proposed again.
    assert "oil painting" not in candidates


def test_explicit_tag_wins_over_phrase_inside_the_subject():
    # "photograph" is explicit in the subject, but "35mm film" is its own
    # tagged segment - the standalone tag is the stronger signal and must
    # claim the medium slot.
    fields = weak_label("a photograph of a harbour at night, 35mm film, soft lighting")
    assert fields.medium == "35mm film"
    assert fields.lighting == "soft lighting"


def test_second_pass_backfills_only_empty_slots():
    # style is never tagged on its own here, so it backfills from the subject.
    fields = weak_label("hyperrealistic photograph of an astronaut, cinematic lighting")
    assert fields.style == "hyperrealistic"
    assert fields.lighting == "cinematic lighting"


def test_normalize_for_match_rejoins_detokenized_spacing():
    from app.weak_label import normalize_for_match

    assert normalize_for_match("3 d render") == "3d render"
    assert normalize_for_match("4 k") == "4k"
    assert normalize_for_match("5 0 mm") == "50mm"
    assert normalize_for_match("trending on art station") == "trending on artstation"
    # Ordinary text must pass through untouched.
    assert normalize_for_match("a red fox in a forest") == "a red fox in a forest"


def test_detokenized_corpus_spelling_still_matches_lexicons():
    # The ingested corpus writes these with inserted spaces; the labeler has
    # to classify them the same as the normal spelling.
    fields = weak_label("a spaceship, 3 d render, 8 k, trending on art station")
    assert fields.medium == "3 d render"
    assert "8 k" in fields.modifiers
    assert "trending on art station" in fields.modifiers


def test_sentence_periods_split_segments():
    fields = weak_label("two flying saucers battling over the ocean. high detailed oil painting. dramatic.")
    assert fields.subject == "two flying saucers battling over the ocean"
    # "high detailed oil painting" is long enough to read as content, so pass 2
    # donates just the matched phrase to the medium slot and keeps the full
    # segment as a modifier.
    assert fields.medium == "oil painting"
    assert "high detailed oil painting" in fields.modifiers
    assert fields.tone == "dramatic"


def test_decimal_points_do_not_split_segments():
    # Lens/CFG specs in the corpus contain decimals; splitting there would
    # shred them into junk modifiers.
    positives, _ = split_prompt("a portrait, 8 5 mm f 1. 8, sharp focus")
    assert "8 5 mm f 1. 8" in positives


def test_initials_in_artist_credits_survive_period_splitting():
    # "j. c. leyendecker" must stay one segment, not shatter into "j" / "c".
    positives, _ = split_prompt("a knight, by alphonse mucha j. c. leyendecker")
    assert positives == ["a knight", "alphonse mucha j. c. leyendecker"]
