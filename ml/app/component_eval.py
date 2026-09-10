"""Component-wise precision, recall and F1 for structured reconstructions.

Scores predicted StructuredFields against weak labels, so results measure
agreement with those labels rather than with ground truth; see
docs/label-quality.md for their measured per-field precision.

String fields are scored at token level rather than by exact match. Exact
match is too brittle for this corpus - "cinematic lighting" against "dim
volumetric cinematic lighting" is a near miss, not a total failure - and the
detokenized corpus spelling ("8 k", "3 d render") makes character-level
comparison worse. List fields are scored as set overlap over normalized
entries.

Empty values are handled explicitly, because correct abstention is a
meaningful outcome when most fields are absent from most targets:

    gold empty, pred empty     -> correct abstention (tracked, not in P/R)
    gold empty, pred non-empty -> false positive only
    gold non-empty, pred empty -> false negative only
    both non-empty             -> token or set level TP, FP, FN

Averaging is micro within a field, so a field's score is not dominated by the
many rows where it is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

STRING_FIELDS = ["subject", "style", "medium", "lighting", "tone"]
LIST_FIELDS = ["modifiers", "negative_constraints"]
ALL_FIELDS = STRING_FIELDS + LIST_FIELDS

# Fields with too little support for their F1 to be meaningful. Reported
# separately so a near-empty field cannot swing the headline average;
# negative_constraints is present on well under 1% of the corpus.
LOW_SUPPORT_FIELDS = ["negative_constraints"]
HEADLINE_FIELDS = [f for f in ALL_FIELDS if f not in LOW_SUPPORT_FIELDS]

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace."""
    return _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower())).strip()


def tokenize(text: Optional[str]) -> list[str]:
    return normalize(text).split() if text else []


def _normalize_entries(values) -> set[str]:
    if not values:
        return set()
    return {normalize(v) for v in values if v and normalize(v)}


@dataclass
class FieldScore:
    field: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    # Rows where the target had no value for this field.
    gold_absent: int = 0
    # ...of which the prediction also had none.
    correct_abstentions: int = 0
    # Rows where both sides had a value and matched exactly.
    exact_matches: int = 0
    gold_present: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.gold_present if self.gold_present else 0.0

    @property
    def abstention_rate(self) -> float:
        return self.correct_abstentions / self.gold_absent if self.gold_absent else 0.0


@dataclass
class ComponentReport:
    fields: dict[str, FieldScore] = field(default_factory=dict)
    rows_scored: int = 0

    def macro_f1(self, names: Iterable[str] = None) -> float:
        names = list(names) if names is not None else HEADLINE_FIELDS
        scores = [self.fields[n].f1 for n in names if n in self.fields]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def headline_macro_f1(self) -> float:
        """Macro F1 over fields with meaningful support."""
        return self.macro_f1(HEADLINE_FIELDS)

    @property
    def full_macro_f1(self) -> float:
        return self.macro_f1(ALL_FIELDS)


def _score_bag(pred: list[str] | set[str], gold: list[str] | set[str], score: FieldScore) -> None:
    """Accumulate TP/FP/FN for one row, comparing both sides as sets."""
    pred_set, gold_set = set(pred), set(gold)
    if not gold_set:
        score.gold_absent += 1
        if not pred_set:
            score.correct_abstentions += 1
        else:
            score.false_positives += len(pred_set)
        return

    score.gold_present += 1
    if not pred_set:
        score.false_negatives += len(gold_set)
        return

    overlap = pred_set & gold_set
    score.true_positives += len(overlap)
    score.false_positives += len(pred_set - gold_set)
    score.false_negatives += len(gold_set - pred_set)
    if pred_set == gold_set:
        score.exact_matches += 1


def score_predictions(
    predictions: list[Optional[dict]], targets: list[dict]
) -> ComponentReport:
    """Score predicted structured fields against weak-label targets.

    A `None` prediction, meaning unparseable model output, is scored as a
    total miss rather than skipped; otherwise a model that fails to produce
    JSON would outscore one that produces wrong JSON.
    """
    if len(predictions) != len(targets):
        raise ValueError(
            f"{len(predictions)} predictions vs {len(targets)} targets"
        )

    report = ComponentReport(
        fields={name: FieldScore(field=name) for name in ALL_FIELDS},
        rows_scored=len(targets),
    )

    for pred, gold in zip(predictions, targets):
        for name in STRING_FIELDS:
            _score_bag(
                tokenize((pred or {}).get(name)),
                tokenize(gold.get(name)),
                report.fields[name],
            )
        for name in LIST_FIELDS:
            _score_bag(
                _normalize_entries((pred or {}).get(name)),
                _normalize_entries(gold.get(name)),
                report.fields[name],
            )

    return report


def format_report(report: ComponentReport, title: str = "") -> str:
    """Render the report as a markdown table."""
    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines += [
        "| Field | Precision | Recall | F1 | Exact match (when present) | Correct abstention (when absent) | Support |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ALL_FIELDS:
        s = report.fields[name]
        note = " *(low support)*" if name in LOW_SUPPORT_FIELDS else ""
        lines.append(
            f"| `{name}`{note} | {s.precision:.3f} | {s.recall:.3f} | **{s.f1:.3f}** | "
            f"{s.exact_match_rate:.3f} | {s.abstention_rate:.3f} | {s.gold_present} |"
        )
    lines += [
        "",
        f"**Headline macro F1** (excluding {', '.join(f'`{f}`' for f in LOW_SUPPORT_FIELDS)}): "
        f"**{report.headline_macro_f1:.4f}**",
        f"Full macro F1 (all {len(ALL_FIELDS)} fields): {report.full_macro_f1:.4f}",
        f"Rows scored: {report.rows_scored}",
    ]
    return "\n".join(lines)
