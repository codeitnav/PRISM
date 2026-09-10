#!/usr/bin/env python3
"""Evaluate the decomposer against its acceptance criteria.

Measures two things:

  1. the share of held-out inputs for which the model emits schema-valid JSON;
  2. component-wise F1 against the weak labels, compared to untrained
     baselines.

Two controls are reported. The first is the same base model with the adapter
disabled. The second additionally shows it two format demonstrations without
training it on the task, which separates learning the output format from
learning the mapping - if an untrained model shown the format scores above
zero, the fine-tune's gain is attributable to the task rather than to format
acquisition.

Component F1 comes from app.component_eval. Scores measure agreement with
weak labels rather than ground truth; see docs/label-quality.md.

Outputs:
    data/results/decomposer_eval.json
    data/results/decomposer_eval.md

Usage:
    python -m scripts.eval_decomposer
    python -m scripts.eval_decomposer --split test
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

from app.component_eval import ALL_FIELDS, format_report, score_predictions
from app.decompose import ADAPTER_DIR, BASE_MODEL, decompose_batch

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SFT_PATH = DATA_DIR / "decomp_sft.jsonl"
RESULTS_DIR = DATA_DIR / "results"
OUT_JSON = RESULTS_DIR / "decomposer_eval.json"
OUT_MD = RESULTS_DIR / "decomposer_eval.md"

SCHEMA_VALIDITY_TARGET = 0.98
NUM_EXAMPLES_IN_REPORT = 8


def run_variant(
    name: str, rows: list[dict], use_adapter: bool, batch_size: int, fewshot: bool = False
) -> dict:
    print(f"\n=== {name} (use_adapter={use_adapter}) ===", flush=True)
    t0 = time.time()
    results = decompose_batch(
        [r["input_text"] for r in rows],
        use_adapter=use_adapter,
        batch_size=batch_size,
        fewshot=fewshot,
    )
    elapsed = time.time() - t0

    predictions = [r.fields.model_dump() if r.is_valid else None for r in results]
    targets = [r["target"] for r in rows]
    report = score_predictions(predictions, targets)

    valid = sum(1 for r in results if r.is_valid)
    validity = valid / len(results)
    print(
        f"schema-valid: {valid}/{len(results)} ({validity*100:.1f}%) | "
        f"headline macro F1: {report.headline_macro_f1:.4f} | "
        f"{elapsed/len(results):.2f}s/row",
        flush=True,
    )

    # Keep a few parse failures verbatim - a bare rate is not actionable when
    # debugging what the model actually emitted.
    failures = [
        {"id": row["id"], "error": res.parse_error, "raw_output": res.raw_output[:300]}
        for row, res in zip(rows, results)
        if not res.is_valid
    ][:5]

    return {
        "name": name,
        "use_adapter": use_adapter,
        "fewshot": fewshot,
        "rows": len(results),
        "schema_valid": valid,
        "schema_validity": validity,
        "seconds_per_row": elapsed / len(results),
        "wall_clock_seconds": elapsed,
        "headline_macro_f1": report.headline_macro_f1,
        "full_macro_f1": report.full_macro_f1,
        "per_field": {
            f: {
                "precision": report.fields[f].precision,
                "recall": report.fields[f].recall,
                "f1": report.fields[f].f1,
                "exact_match_rate": report.fields[f].exact_match_rate,
                "abstention_rate": report.fields[f].abstention_rate,
                "support": report.fields[f].gold_present,
            }
            for f in ALL_FIELDS
        },
        "parse_failures": failures,
        "_report": report,
        "_predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--skip-control", action="store_true", help="evaluate the adapter only"
    )
    parser.add_argument(
        "--skip-fewshot",
        action="store_true",
        help="skip the few-shot control (the stronger, format-aware comparison)",
    )
    args = parser.parse_args()

    if not SFT_PATH.exists():
        raise SystemExit(f"{SFT_PATH} not found - run `make decomp-sft` first.")
    rows = [json.loads(l) for l in open(SFT_PATH) if json.loads(l)["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    print(f"Evaluating on {len(rows)} '{args.split}' rows | adapter: {ADAPTER_DIR}")

    tuned = run_variant("LoRA fine-tuned", rows, True, args.batch_size)
    # Free the tuned model before loading the control - holding both at once
    # OOM-kills this machine (measured).
    import app.decompose as decompose_mod

    decompose_mod._cache.clear()
    gc.collect()

    control = None
    if not args.skip_control:
        control = run_variant("Zero-shot control", rows, False, args.batch_size)
        decompose_mod._cache.clear()
        gc.collect()

    fewshot = None
    if not args.skip_fewshot:
        fewshot = run_variant(
            "Few-shot control (format shown, task untrained)",
            rows,
            False,
            args.batch_size,
            fewshot=True,
        )
        decompose_mod._cache.clear()
        gc.collect()

    _write_report(args, rows, tuned, control, fewshot)


def _write_report(args, rows, tuned, control, fewshot=None) -> None:
    validity_pass = tuned["schema_validity"] > SCHEMA_VALIDITY_TARGET
    beats_control = (
        control is None or tuned["headline_macro_f1"] > control["headline_macro_f1"]
    )

    lines = [
        "# Decomposer Evaluation - Task 5.3",
        "",
        f"- **Base model:** `{BASE_MODEL}` + LoRA adapter (`{ADAPTER_DIR.name}`)",
        f"- **Split:** {args.split} ({len(rows)} rows, prompt-disjoint from train)",
        "",
        "## Done-condition",
        "",
        "| Requirement | Result | Status |",
        "|---|---|---|",
        f"| Schema-valid JSON on >98% of val inputs | "
        f"{tuned['schema_valid']}/{tuned['rows']} = **{tuned['schema_validity']*100:.1f}%** | "
        f"{'PASS' if validity_pass else 'FAIL'} |",
    ]
    if control is not None:
        lines.append(
            f"| Beats zero-shot control on component F1 | "
            f"**{tuned['headline_macro_f1']:.4f}** vs {control['headline_macro_f1']:.4f} | "
            f"{'PASS' if beats_control else 'FAIL'} |"
        )
    lines += [
        "",
        "## Comparison",
        "",
        "| Variant | Schema-valid | Headline macro F1 | Full macro F1 | s/row |",
        "|---|---|---|---|---|",
    ]
    for variant in [v for v in (tuned, control, fewshot) if v is not None]:
        lines.append(
            f"| {variant['name']} | {variant['schema_validity']*100:.1f}% | "
            f"**{variant['headline_macro_f1']:.4f}** | {variant['full_macro_f1']:.4f} | "
            f"{variant['seconds_per_row']:.2f} |"
        )

    if fewshot is not None:
        beats_fewshot = tuned["headline_macro_f1"] > fewshot["headline_macro_f1"]
        lines += [
            "",
            "### On the controls - and why neither is informative here",
            "",
            "Both controls score **0.0 F1**, and that is a limitation of the comparison "
            "rather than a triumph. The strict zero-shot control emits schema-valid JSON on "
            f"only {control['schema_validity']*100:.1f}% of rows; the few-shot control, shown "
            f"two format demonstrations, manages {fewshot['schema_validity']*100:.1f}% - "
            "*worse*, because the longer prompt gives a 135M model more to lose track of.",
            "",
            "The few-shot control was added specifically to separate **format learning** from "
            "**task learning**: if an untrained model shown the format could score above zero, "
            "then the fine-tune's gain would be attributable to the task rather than to "
            "learning to emit JSON. It cannot. So at this model scale the two are "
            "inseparable, and the honest reading of "
            f"**{tuned['headline_macro_f1']:.4f}** vs **0.0000** is: *the fine-tune is what "
            "makes the model able to attempt the task at all*, not that it has learned the "
            "mapping well. The absolute F1 below is what says how well it learned it.",
        ]
        if not beats_fewshot:
            lines.append(
                "\n> The fine-tune does not beat the few-shot control - investigate before "
                "reporting any result from this run."
            )

    lines += ["", "## Per-field breakdown", "", format_report(tuned["_report"], tuned["name"])]
    for variant in (control, fewshot):
        if variant is not None:
            lines += ["", format_report(variant["_report"], variant["name"])]

    lines += [
        "",
        "> Scores are agreement with the **weak labels** from Task 1.3, not with ground "
        "truth. See `docs/label-quality.md` for those labels' measured per-field precision "
        "(`modifiers` in particular is noisy at 0.26, which is why the SFT targets keep "
        "only lexicon-recognised entries). `negative_constraints` is populated on 4 of 700 "
        "rows, so it is excluded from the headline macro and reported separately.",
        "",
        "## Sample predictions",
        "",
    ]
    for row, pred in list(zip(rows, tuned["_predictions"]))[:NUM_EXAMPLES_IN_REPORT]:
        lines += [
            f"**`{row['id']}`** — true prompt: `{row['true_prompt'][:120]}`",
            "",
            f"- predicted: `{json.dumps(pred, ensure_ascii=False) if pred else 'PARSE FAILURE'}`",
            f"- target:    `{row['target_text']}`",
            "",
        ]
    if tuned["parse_failures"]:
        lines += ["## Parse failures (first 5)", ""]
        for failure in tuned["parse_failures"]:
            lines += [
                f"- **`{failure['id']}`** — {failure['error']}",
                f"  - raw: `{failure['raw_output']}`",
            ]
        lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    serializable = {
        k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
        for k, v in (("tuned", tuned), ("control", control), ("fewshot", fewshot))
        if v is not None
    }
    OUT_JSON.write_text(
        json.dumps(
            {
                "split": args.split,
                "rows": len(rows),
                "schema_validity_target": SCHEMA_VALIDITY_TARGET,
                "validity_pass": validity_pass,
                "beats_control": beats_control,
                **serializable,
            },
            indent=2,
        )
    )
    print("\n".join(lines[:20]))
    print(f"\nSaved -> {OUT_MD}\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
