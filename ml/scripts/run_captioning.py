#!/usr/bin/env python3
"""Generate and persist captions for dataset splits.

Outputs:
    data/captions/captions.parquet   id, image_path, caption, model, prompt
    data/results/captioning.md       throughput and a sample of caption/prompt pairs

The true prompt is carried alongside each caption so downstream joins do not
need to re-read pairs.parquet, and so the report can show the two side by side.

Re-running is cheap: app.caption caches by image content hash, so only new
images cost model time. The parquet is merged rather than overwritten, so
captioning additional splits accumulates.

Usage:
    python -m scripts.run_captioning
    python -m scripts.run_captioning --splits train val test
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from app.caption import CAPTION_MODEL, caption_images

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
SPLITS_DIR = DATA_DIR / "splits"
CAPTIONS_PATH = DATA_DIR / "captions" / "captions.parquet"
REPORT_PATH = DATA_DIR / "results" / "captioning.md"

SAMPLE_ROWS_IN_REPORT = 15


def load_split_ids(splits: list[str]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for split in splits:
        path = SPLITS_DIR / f"{split}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found - run `make split` first.")
        with open(path) as f:
            for dataset_id in json.load(f):
                if dataset_id not in seen:
                    seen.add(dataset_id)
                    ids.append(dataset_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test"],
        help="split names to caption (default: test)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=None, help="cap the number of images (for smoke runs)"
    )
    args = parser.parse_args()

    ids = load_split_ids(args.splits)
    if args.limit:
        ids = ids[: args.limit]

    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    missing = [i for i in ids if i not in rows_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} split ids absent from pairs.parquet, e.g. {missing[:3]}")
    rows = [rows_by_id[i] for i in ids]

    print(f"Captioning {len(rows)} images from split(s) {args.splits} with {CAPTION_MODEL}...")
    image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in rows]

    t0 = time.time()
    captions = caption_images(image_bytes, batch_size=args.batch_size)
    elapsed = time.time() - t0

    new_rows = [
        {
            "id": r["id"],
            "image_path": r["image_path"],
            "caption": caption,
            "model": CAPTION_MODEL,
            "prompt": r["prompt"],
        }
        for r, caption in zip(rows, captions)
    ]

    # Merge with anything captioned by an earlier run, keyed by (id, model).
    merged: dict[tuple[str, str], dict] = {}
    if CAPTIONS_PATH.exists():
        for row in pq.read_table(CAPTIONS_PATH).to_pylist():
            merged[(row["id"], row["model"])] = row
    for row in new_rows:
        merged[(row["id"], row["model"])] = row

    CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_rows = sorted(merged.values(), key=lambda r: (r["model"], r["id"]))
    pq.write_table(pa.Table.from_pylist(out_rows), CAPTIONS_PATH)
    print(f"Saved {len(out_rows)} captions ({len(new_rows)} from this run) -> {CAPTIONS_PATH}")

    per_image = elapsed / len(rows) if rows else 0.0
    empty = sum(1 for c in captions if not c.strip())
    words = [len(c.split()) for c in captions]
    print(f"\nWall clock: {elapsed:.1f}s total, {per_image:.2f}s/image (cache hits included)")
    print(f"Caption length (words) - min {min(words)}, max {max(words)}, mean {sum(words)/len(words):.1f}")
    if empty:
        print(f"WARNING: {empty} empty captions")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("# Captioning Results - Task 5.1\n\n")
        f.write(f"- **Model:** `{CAPTION_MODEL}`\n")
        f.write(f"- **Split(s):** {', '.join(args.splits)} ({len(rows)} images)\n")
        f.write(f"- **Wall clock:** {elapsed:.1f}s total, {per_image:.2f}s/image ")
        f.write("(includes cache hits; a fully cached re-run is near-instant)\n")
        f.write(
            f"- **Caption length (words):** min {min(words)}, max {max(words)}, "
            f"mean {sum(words)/len(words):.1f}\n"
        )
        f.write(f"- **Empty captions:** {empty}\n\n")
        f.write("## Caption vs. true prompt\n\n")
        f.write(
            "The caption describes what is *in* the image; the true prompt is what "
            "generated it. The gap between the two columns is exactly the gap the "
            "decomposition model (Task 5.3) has to close - the caption grounds the "
            "subject, but carries almost none of the style/medium/quality vocabulary.\n\n"
        )
        f.write("| id | Caption | True prompt |\n|---|---|---|\n")
        for row in new_rows[:SAMPLE_ROWS_IN_REPORT]:
            caption = row["caption"].replace("|", "\\|")
            prompt = row["prompt"].replace("|", "\\|")
            f.write(f"| `{row['id']}` | {caption} | {prompt[:140]} |\n")
    print(f"Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
