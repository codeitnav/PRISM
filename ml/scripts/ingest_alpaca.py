#!/usr/bin/env python3
"""Task 1.4 - Text-side dataset ingestion (Alpaca subset).

Downloads the cleaned Stanford Alpaca instruction dataset from Hugging Face
(yahma/alpaca-cleaned) and curates a small set of (prompt, output) pairs for
the text pipeline's reference set - the text-only counterpart to Task 1.1's
image-prompt pairs.

Mapping to PRISM's "reconstruct the prompt behind the content" framing: an
Alpaca row's `output` (the generated text) plays the role an image plays on
the image side, and its `instruction` (+ `input`, if present) plays the role
of the prompt to be reconstructed.

Scope note: optional task, slotted in whenever bandwidth allows (per the
plan). Capped at MAX_PAIRS to match Task 1.1's image-side scope and keep this
a quick, CPU-only, low-storage step - not a from-scratch text pipeline.

Output:
    data/alpaca/pairs.parquet   (id, prompt, output, source)

Usage (inside the ml container):
    docker compose run --rm ml python scripts/ingest_alpaca.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data")) / "alpaca"
PAIRS_PATH = DATA_DIR / "pairs.parquet"

MIN_PROMPT_TOKENS = 3
MAX_PROMPT_TOKENS = 60
MAX_PAIRS = 700  # match Task 1.1's image-side scope


def normalize(text: str) -> str:
    """Collapse case/whitespace so near-identical prompts dedupe as equal."""
    return " ".join(text.lower().split())


def main() -> None:
    print("Downloading Alpaca (cleaned) from Hugging Face...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    print(f"Raw rows: {len(ds)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seen_prompts: set[str] = set()
    kept_rows = []
    all_lengths = []
    dropped_length = 0
    dropped_dupe = 0

    for row in ds:
        instruction = row["instruction"].strip()
        extra_input = (row.get("input") or "").strip()
        prompt = f"{instruction}\n{extra_input}" if extra_input else instruction
        output = row["output"].strip()

        token_count = len(prompt.split())
        all_lengths.append(token_count)

        if not (MIN_PROMPT_TOKENS <= token_count <= MAX_PROMPT_TOKENS) or not output:
            dropped_length += 1
            continue

        key = normalize(prompt)
        if key in seen_prompts:
            dropped_dupe += 1
            continue
        seen_prompts.add(key)

        kept_rows.append(
            {
                "id": f"{len(kept_rows):06d}",
                "prompt": prompt,
                "output": output,
                "source": "alpaca",
            }
        )

        if len(kept_rows) >= MAX_PAIRS:
            print(f"Reached MAX_PAIRS={MAX_PAIRS}, stopping early.")
            break

    pq.write_table(pa.Table.from_pylist(kept_rows), PAIRS_PATH)

    print("\n=== Stats report ===")
    print(f"Kept:    {len(kept_rows)} pairs")
    print(f"Dropped: {dropped_length} bad length/empty, {dropped_dupe} duplicate")
    print(
        f"Prompt length (tokens) - min: {min(all_lengths)}, max: {max(all_lengths)}, "
        f"mean: {sum(all_lengths) / len(all_lengths):.1f}"
    )
    print(f"Saved -> {PAIRS_PATH}")


if __name__ == "__main__":
    main()
