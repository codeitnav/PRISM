#!/usr/bin/env python3
"""Task 1.3 - field-level precision audit of the weak labels.

The roadmap's done-condition for Task 1.3 is a *manual* audit of 100 random
samples reporting field-level precision, logged in docs/label-quality.md.
That number cannot be computed automatically - there is no gold standard to
compare against, which is the whole reason the labels are called "weak". So
this script splits the work in two:

    --emit    sample 100 labeled rows (deterministic seed) and write an audit
              worksheet to data/diffusiondb/label_audit.jsonl, one row per
              sample with every field and a blank verdict per field.

    --score   read the filled-in worksheet and compute field-level precision
              = (# fields marked correct) / (# fields the labeler populated),
              then render docs/label-quality.md.

A verdict is one of: "correct", "wrong", or "" (not yet judged). Precision is
computed over populated fields only - a field the labeler left null is a
recall miss, not a precision error, and is counted separately as coverage.

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.audit_weak_labels --emit
    # ... fill in verdicts in data/diffusiondb/label_audit.jsonl ...
    docker compose run --rm ml python -m scripts.audit_weak_labels --score
    cp data/results/label-quality.md docs/label-quality.md
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
LABELED_PATH = DATA_DIR / "diffusiondb" / "pairs_labeled.parquet"
AUDIT_PATH = DATA_DIR / "diffusiondb" / "label_audit.jsonl"
# Written under /data/results and copied into docs/ afterward - docs/ is
# mounted read-only in the ml container (same convention as the Task 4.x
# baseline scripts).
REPORT_PATH = DATA_DIR / "results" / "label-quality.md"

SAMPLE_SIZE = 100
SAMPLE_SEED = 1337  # fixed so the audited sample is reproducible

AUDITED_FIELDS = ["subject", "style", "medium", "lighting", "tone", "modifiers", "negative_constraints"]


def _populated(value) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    return bool(str(value).strip())


def emit() -> None:
    rows = pq.read_table(LABELED_PATH).to_pylist()
    rng = random.Random(SAMPLE_SEED)
    sample = rng.sample(rows, min(SAMPLE_SIZE, len(rows)))

    with open(AUDIT_PATH, "w") as f:
        for row in sample:
            record = {
                "id": row["id"],
                "prompt": row["prompt"],
                "labels": {name: row[name] for name in AUDITED_FIELDS},
                # Fill each populated field with "correct" or "wrong".
                "verdicts": {name: "" for name in AUDITED_FIELDS if _populated(row[name])},
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(sample)} audit samples -> {AUDIT_PATH}")
    print("Fill in each verdict as 'correct' or 'wrong', then re-run with --score.")


def score() -> None:
    if not AUDIT_PATH.exists():
        raise SystemExit(f"{AUDIT_PATH} not found - run with --emit first.")

    records = [json.loads(line) for line in open(AUDIT_PATH) if line.strip()]
    populated: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    judged: Counter[str] = Counter()

    for record in records:
        for name, verdict in record["verdicts"].items():
            populated[name] += 1
            if verdict:
                judged[name] += 1
            if verdict == "correct":
                correct[name] += 1

    total_judged = sum(judged.values())
    total_correct = sum(correct.values())
    unjudged = sum(populated.values()) - total_judged

    lines = [
        "# Label Quality - Task 1.3 Weak Structured Labels",
        "",
        f"Audit of {len(records)} randomly sampled prompts "
        f"(seed {SAMPLE_SEED}, source `data/diffusiondb/pairs_labeled.parquet`).",
        "",
        "Field-level **precision** is measured over fields the labeler actually populated:",
        "`correct / (correct + wrong)`. A field left null is a recall miss, not a precision",
        "error, so it is reported separately as **coverage** (share of the 100 samples where",
        "the labeler emitted a value at all).",
        "",
        "| Field | Coverage (of 100) | Judged | Correct | Precision |",
        "|---|---|---|---|---|",
    ]
    for name in AUDITED_FIELDS:
        precision = f"{correct[name] / judged[name]:.2f}" if judged[name] else "n/a"
        lines.append(
            f"| `{name}` | {populated[name]} | {judged[name]} | {correct[name]} | {precision} |"
        )

    overall = f"{total_correct / total_judged:.3f}" if total_judged else "n/a"
    lines += [
        "",
        f"**Overall field-level precision: {overall}** "
        f"({total_correct} correct of {total_judged} judged field values).",
        "",
    ]
    if unjudged:
        lines.append(
            f"> {unjudged} populated field values are still unjudged - the number above "
            "covers only the judged subset."
        )
        lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved -> {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit", action="store_true", help="write the audit worksheet")
    group.add_argument("--score", action="store_true", help="score the filled worksheet")
    args = parser.parse_args()
    emit() if args.emit else score()


if __name__ == "__main__":
    main()
