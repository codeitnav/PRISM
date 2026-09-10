"""Tests for component-wise precision, recall and F1."""

import pytest

from app.component_eval import (
    ALL_FIELDS,
    HEADLINE_FIELDS,
    LOW_SUPPORT_FIELDS,
    format_report,
    normalize,
    score_predictions,
    tokenize,
)


def _target(**over) -> dict:
    base = dict(
        subject="a red fox in a snowy forest",
        style="anime",
        medium="oil painting",
        lighting="golden hour",
        tone="serene",
        modifiers=["highly detailed", "8 k"],
        negative_constraints=[],
    )
    base.update(over)
    return base


class TestNormalization:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize("Oil Painting, 8K!") == "oil painting 8k"

    def test_collapses_whitespace(self):
        assert normalize("  a   red \n fox ") == "a red fox"

    def test_tokenize_handles_none_and_empty(self):
        assert tokenize(None) == []
        assert tokenize("") == []


class TestScoring:
    def test_perfect_prediction_scores_one(self):
        report = score_predictions([_target()], [_target()])
        for name in ALL_FIELDS:
            if report.fields[name].gold_present:
                assert report.fields[name].f1 == pytest.approx(1.0)
        assert report.headline_macro_f1 == pytest.approx(1.0)

    def test_completely_wrong_prediction_scores_zero(self):
        wrong = _target(
            subject="zzz", style="qqq", medium="www", lighting="eee", tone="rrr",
            modifiers=["ttt"],
        )
        report = score_predictions([wrong], [_target()])
        assert report.headline_macro_f1 == pytest.approx(0.0)

    def test_partial_token_overlap_gets_partial_credit(self):
        """Token-level, not exact match: a near miss is not a total failure."""
        report = score_predictions(
            [_target(lighting="dim volumetric golden hour")],
            [_target(lighting="golden hour")],
        )
        score = report.fields["lighting"]
        assert score.recall == pytest.approx(1.0)      # both gold tokens found
        assert 0.0 < score.precision < 1.0             # but two extra tokens
        assert score.exact_matches == 0

    def test_unparseable_prediction_is_a_total_miss_not_a_skip(self):
        """A model that emits nothing must not outscore one that emits wrong JSON."""
        report = score_predictions([None], [_target()])
        assert report.headline_macro_f1 == pytest.approx(0.0)
        assert report.fields["subject"].false_negatives > 0
        assert report.fields["subject"].true_positives == 0


class TestNullAndEmptyHandling:
    def test_correct_abstention_is_tracked_not_penalised(self):
        report = score_predictions(
            [_target(lighting=None)], [_target(lighting=None)]
        )
        score = report.fields["lighting"]
        assert score.gold_absent == 1
        assert score.correct_abstentions == 1
        assert score.abstention_rate == pytest.approx(1.0)
        assert score.false_positives == 0 and score.false_negatives == 0

    def test_hallucinated_field_costs_precision_only(self):
        report = score_predictions(
            [_target(lighting="neon lighting")], [_target(lighting=None)]
        )
        score = report.fields["lighting"]
        assert score.false_positives == 2  # two tokens
        assert score.false_negatives == 0

    def test_missed_field_costs_recall_only(self):
        report = score_predictions(
            [_target(lighting=None)], [_target(lighting="golden hour")]
        )
        score = report.fields["lighting"]
        assert score.false_negatives == 2
        assert score.false_positives == 0

    def test_empty_list_counts_as_abstention(self):
        report = score_predictions([_target(modifiers=[])], [_target(modifiers=[])])
        assert report.fields["modifiers"].correct_abstentions == 1


class TestListFields:
    def test_set_overlap_scoring(self):
        report = score_predictions(
            [_target(modifiers=["8 k", "bokeh"])],
            [_target(modifiers=["8 k", "highly detailed"])],
        )
        score = report.fields["modifiers"]
        assert score.true_positives == 1
        assert score.false_positives == 1
        assert score.false_negatives == 1

    def test_entry_comparison_is_normalized(self):
        report = score_predictions(
            [_target(modifiers=["Highly Detailed!", "8K"])],
            [_target(modifiers=["highly detailed", "8k"])],
        )
        assert report.fields["modifiers"].f1 == pytest.approx(1.0)


class TestAggregation:
    def test_headline_macro_excludes_low_support_fields(self):
        assert LOW_SUPPORT_FIELDS == ["negative_constraints"]
        assert "negative_constraints" not in HEADLINE_FIELDS
        assert len(HEADLINE_FIELDS) == len(ALL_FIELDS) - 1

    def test_headline_and_full_macro_differ_when_low_support_field_differs(self):
        """The whole point of the split: a 0.6%-support field must not swing
        the headline number."""
        pred = _target(negative_constraints=["watermark"])
        gold = _target(negative_constraints=[])
        report = score_predictions([pred], [gold])
        assert report.headline_macro_f1 == pytest.approx(1.0)
        assert report.full_macro_f1 < report.headline_macro_f1

    def test_micro_averaging_within_a_field(self):
        # Two rows: one perfect, one a total miss on subject.
        report = score_predictions(
            [_target(), _target(subject="zzz")], [_target(), _target()]
        )
        score = report.fields["subject"]
        assert score.true_positives == 6   # 6 tokens from the perfect row
        assert score.false_negatives == 6  # 6 tokens missed on the bad row
        assert score.recall == pytest.approx(0.5)

    def test_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="1 predictions vs 2 targets"):
            score_predictions([_target()], [_target(), _target()])


def test_format_report_renders_every_field_and_both_macros():
    report = score_predictions([_target()], [_target()])
    text = format_report(report, "Variant")
    for name in ALL_FIELDS:
        assert f"`{name}`" in text
    assert "Headline macro F1" in text and "Full macro F1" in text
    assert "*(low support)*" in text  # negative_constraints is flagged
