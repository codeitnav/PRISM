#!/usr/bin/env python3
"""Build the decomposition SFT dataset.

Joins the upstream artifacts into one training file:

    captions              data/captions/captions.parquet
    retrieved candidates  data/faiss/index.faiss + MongoDB
    PEZ prompts           data/results/pez*.json
    weak labels + true prompts
                          data/diffusiondb/pairs_labeled.parquet

Writes data/decomp_sft.jsonl, one row per example. Each row carries its own
"split", so the held-out evaluation set travels with the dataset rather than
in a separate file that can drift out of sync.

Every row is re-validated after serialization by app.sft.validate_row, and the
script exits non-zero if any row fails, so an invalid dataset cannot be
written silently.

Usage:
    python -m scripts.build_decomp_sft
    python -m scripts.build_decomp_sft --splits train val
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq

from app.embed import embed_images
from app.retrieval import retrieve_candidates
from app.schemas.structured_fields import StructuredFields
from app.sft import (
    DEFAULT_TOP_K,
    build_example,
    lexicon_vocabulary_size,
    validate_row,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
LABELED_PATH = DATA_DIR / "diffusiondb" / "pairs_labeled.parquet"
CAPTIONS_PATH = DATA_DIR / "captions" / "captions.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = DATA_DIR / "results"
OUT_PATH = DATA_DIR / "decomp_sft.jsonl"
REPORT_PATH = RESULTS_DIR / "decomp_sft.md"

STRUCTURED_KEYS = [
    "subject",
    "style",
    "medium",
    "lighting",
    "tone",
    "modifiers",
    "negative_constraints",
]


def load_pez_prompts() -> tuple[dict[str, str], list[str]]:
    """Collect PEZ prompts from every pez*.json in data/results.

    Reading by glob means coverage can be extended by dropping in another
    results file, without modifying this script or the published baseline's
    own artifact.
    """
    prompts: dict[str, str] = {}
    sources: list[str] = []
    for path in sorted(glob.glob(str(RESULTS_DIR / "pez*.json"))):
        with open(path) as f:
            data = json.load(f)
        for item in data.get("items", []):
            prompts[item["id"]] = item["prompt"]
        sources.append(Path(path).name)
    return prompts, sources


def load_split_map(splits: list[str]) -> dict[str, str]:
    """dataset_id -> split name, for the requested splits."""
    split_of: dict[str, str] = {}
    for split in splits:
        path = SPLITS_DIR / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `make split` first.")
        with open(path) as f:
            for dataset_id in json.load(f):
                split_of[dataset_id] = split
    return split_of


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap rows per split (smoke runs)"
    )
    parser.add_argument(
        "--allow-missing-captions",
        action="store_true",
        help="skip ids with no caption instead of failing (default: fail loudly)",
    )
    args = parser.parse_args()

    if not LABELED_PATH.exists():
        raise SystemExit(f"{LABELED_PATH} not found - run `make weak-label` first.")
    if not CAPTIONS_PATH.exists():
        raise SystemExit(f"{CAPTIONS_PATH} not found - run `make caption` first.")

    labeled = {r["id"]: r for r in pq.read_table(LABELED_PATH).to_pylist()}
    captions = {r["id"]: r["caption"] for r in pq.read_table(CAPTIONS_PATH).to_pylist()}
    pez_prompts, pez_sources = load_pez_prompts()
    split_of = load_split_map(args.splits)

    ids_by_split: dict[str, list[str]] = {s: [] for s in args.splits}
    for dataset_id, split in split_of.items():
        ids_by_split[split].append(dataset_id)
    for split in ids_by_split:
        ids_by_split[split].sort()
        if args.limit:
            ids_by_split[split] = ids_by_split[split][: args.limit]

    ordered_ids = [i for s in args.splits for i in ids_by_split[s]]

    missing_caption = [i for i in ordered_ids if i not in captions]
    if missing_caption and not args.allow_missing_captions:
        raise SystemExit(
            f"{len(missing_caption)} of {len(ordered_ids)} ids have no caption "
            f"(e.g. {missing_caption[:3]}).\n"
            "Caption them first:  make caption SPLITS=\"" + " ".join(args.splits) + "\"\n"
            "Or pass --allow-missing-captions to skip them (recorded in the report)."
        )
    ordered_ids = [i for i in ordered_ids if i in captions]

    print(f"Building SFT rows for {len(ordered_ids)} ids across splits {args.splits}")
    print(f"PEZ coverage source(s): {pez_sources or 'none found'}")

    rows: list[dict] = []
    pez_hits = 0
    t0 = time.time()

    for n, dataset_id in enumerate(ordered_ids, 1):
        labeled_row = labeled[dataset_id]
        weak_fields = StructuredFields(
            **{k: labeled_row[k] for k in STRUCTURED_KEYS}
        )

        image_bytes = (IMAGES_DIR / labeled_row["image_path"]).read_bytes()
        query_vec = embed_images([image_bytes])[0]
        # +1 so that dropping the row's own self-match (see app.sft.build_example)
        # still leaves top_k real neighbours.
        candidates = retrieve_candidates(query_vec, top_k=args.top_k + 1)

        pez_prompt = pez_prompts.get(dataset_id)
        if pez_prompt:
            pez_hits += 1

        example = build_example(
            dataset_id=dataset_id,
            split=split_of[dataset_id],
            caption=captions[dataset_id],
            candidates=candidates,
            pez_prompt=pez_prompt,
            weak_fields=weak_fields,
            true_prompt=labeled_row["prompt"],
            top_k=args.top_k,
        )
        rows.append(example.to_row())

        if n % 50 == 0 or n == len(ordered_ids):
            print(f"  {n}/{len(ordered_ids)} ({time.time() - t0:.0f}s)")

    # Validate before writing - a dataset that fails its own schema check must
    # never reach disk.
    print("\nValidating every row...")
    for row in rows:
        validate_row(row)
    print(f"All {len(rows)} rows valid.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved -> {OUT_PATH}")

    _report(rows, args, pez_hits, pez_sources, missing_caption)


def _report(rows, args, pez_hits, pez_sources, missing_caption) -> None:
    total = len(rows)
    by_split: dict[str, int] = {}
    for row in rows:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1

    filled = {k: 0 for k in STRUCTURED_KEYS}
    for row in rows:
        for key in STRUCTURED_KEYS:
            if row["target"][key]:
                filled[key] += 1

    raw_mods = sum(len(row["modifiers_raw"]) for row in rows)
    kept_mods = sum(len(row["target"]["modifiers"]) for row in rows)
    self_dropped = sum(1 for row in rows if row["self_match_dropped"])
    train_rows = sum(1 for row in rows if row["split"] == "train")
    empty_retrieval = sum(1 for row in rows if not row["input"]["retrieved_prompts"])
    mean_neighbours = (
        sum(len(row["input"]["retrieved_prompts"]) for row in rows) / total if total else 0
    )
    input_chars = [len(row["input_text"]) for row in rows]
    top1 = [
        row["input"]["retrieved_similarities"][0]
        for row in rows
        if row["input"]["retrieved_similarities"]
    ]
    margins = [
        row["input"]["retrieved_similarities"][0] - row["input"]["retrieved_similarities"][1]
        for row in rows
        if len(row["input"]["retrieved_similarities"]) > 1
    ]
    stripped = sum(
        1 for row in rows if row["input"]["caption"] != row["input"]["caption_raw"]
    )
    copyable = sum(1 for row in rows if row["true_prompt_in_retrieved"])

    lines = [
        "# Decomposition SFT Dataset - Task 5.2",
        "",
        f"- **Rows:** {total}  (" + ", ".join(f"{k}: {v}" for k, v in sorted(by_split.items())) + ")",
        f"- **Retrieved prompts per row:** top-{args.top_k}, mean actually present {mean_neighbours:.2f}",
        f"- **Rows with no retrieved neighbours:** {empty_retrieval}",
        f"- **Self-match dropped (leakage guard):** {self_dropped} rows, "
        f"of {train_rows} train rows (the FAISS index is the train split, so every "
        "train row retrieves itself at rank 1; val/test rows should not)",
        f"- **PEZ coverage:** {pez_hits}/{total} rows ({pez_hits / total * 100:.1f}%) "
        f"from {pez_sources or 'no PEZ results found'}",
        f"- **Input length (chars):** min {min(input_chars)}, max {max(input_chars)}, "
        f"mean {sum(input_chars) / total:.0f}",
        f"- **Top-1 retrieval similarity (after self-exclusion):** "
        f"min {min(top1):.4f}, mean {sum(top1) / len(top1):.4f}, max {max(top1):.4f}"
        if top1
        else "- **Top-1 retrieval similarity:** n/a",
        f"- **Retrieval margin (top1-top2):** mean {sum(margins) / len(margins):.4f}"
        if margins
        else "- **Retrieval margin:** n/a",
        f"- **Captions with boilerplate stripped:** {stripped}/{total} "
        f"({stripped / total * 100:.0f}%)",
        f"- **True prompt verbatim inside a retrieved neighbour:** {copyable}/{total} "
        f"({copyable / total * 100:.1f}%) - not leakage, but an upper bound on what "
        "copying alone could achieve (see below)",
        f"- **Lexicon vocabulary:** {lexicon_vocabulary_size()} curated phrases",
        f"- **Schema validation:** {total}/{total} rows pass `app.sft.validate_row`",
        "",
        "## Target field coverage",
        "",
        "| Field | Rows with a value | Share |",
        "|---|---|---|",
    ]
    for key in STRUCTURED_KEYS:
        lines.append(f"| `{key}` | {filled[key]} | {filled[key] / total * 100:.1f}% |")

    lines += [
        "",
        "## Leakage guard vs. legitimate retrieval",
        "",
        "The FAISS index is built from the 560 train images, so a train row retrieves "
        "**itself** at rank 1 with similarity ~1.0 - and that prompt is exactly what its "
        "target was derived from. Training on that would teach the model to copy "
        "`retrieved_prompts[0]` and ignore the caption entirely, and it would collapse at "
        "inference, where an unseen image is never in the index. Every row's own id is "
        "therefore dropped from its candidates, and `app.sft.validate_row` re-checks the "
        "invariant per row after serialization.",
        "",
        "A *different* image whose prompt is near-identical to - or even verbatim contains "
        "- this one's is a separate matter, and is deliberately **kept**. That is "
        "legitimate retrieval success which also occurs at inference (the train split "
        "contains near-duplicate prompt clusters). Filtering it would make the training "
        f"inputs systematically weaker than production. It affects {copyable}/{total} rows "
        f"({copyable / total * 100:.1f}%), recorded per row as `true_prompt_in_retrieved` - "
        "useful as a ceiling on the copy-only strategy when interpreting Task 5.3's "
        "component F1.",
        "",
        "## Modifier filtering",
        "",
        f"Weak labels produced **{raw_mods}** modifier entries across {total} rows; "
        f"**{kept_mods}** survive lexicon filtering (**{kept_mods / raw_mods * 100:.1f}%** kept)"
        if raw_mods
        else "No modifier entries.",
        "",
        "The Task 1.3 audit measured `modifiers` at 0.26 field-level precision "
        "(`docs/label-quality.md`) - it is the sink for every segment the lexicons did not "
        "classify, so it absorbs scene detail belonging in `subject` alongside genuine "
        "quality modifiers. Training on it verbatim would teach the model to imitate a junk "
        "drawer, so targets keep only lexicon-recognised modifiers. The unfiltered list "
        "survives in each row as `modifiers_raw`, so Task 8.2 can ablate this choice.",
        "",
    ]
    if missing_caption:
        lines += [
            f"> **{len(missing_caption)} ids skipped for having no caption.** Caption them "
            f'with `make caption SPLITS="{" ".join(args.splits)}"` and rebuild.',
            "",
        ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print("\n".join(lines[:14]))
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
