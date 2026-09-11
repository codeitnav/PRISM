#!/usr/bin/env python3
"""Task 4.3 - run the evaluation harness on all reconstructors.

Scores PEZ (Task 4.1) and the naive CLIP-tag baseline (Task 4.2) through
the same harness (app.eval.run_eval), reusing their already-computed
results (data/results/{pez,cliptag}_baseline.json) instead of re-running
PEZ's ~20+ minute optimization, and additionally scores the LoRA
decomposition model (Task 5.3) live - see app.eval's module docstring for
the "reconstructor" interface this relies on.

Output:
    data/results/baselines.csv - one row per (reconstructor, image)
    data/results/baselines.md  - summary + per-image detail
    (copy the .md into docs/results/ afterward - docs/ is mounted read-only
    in the ml container)

Usage (inside the ml container):
    docker compose run --rm ml python -m scripts.run_eval
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import open_clip
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from app.decompose import decompose_batch
from app.embed import embed_images, get_clip_model
from app.eval import run_eval

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
PAIRS_PATH = DATA_DIR / "diffusiondb" / "pairs.parquet"
IMAGES_DIR = DATA_DIR / "diffusiondb"
TEST_SPLIT_PATH = DATA_DIR / "splits" / "test.json"
DECOMP_SFT_PATH = DATA_DIR / "decomp_sft.jsonl"
RESULTS_DIR = DATA_DIR / "results"

# The trained adapter ships inside the repo (small enough to commit, unlike
# data/), so it lives under the ml package rather than under DATA_DIR.
DECOMPOSER_ADAPTER_DIR = Path(__file__).resolve().parent.parent / "models" / "decomposer-lora"

NUM_TEST_IMAGES = 20


@dataclass
class _Item:
    prompt: str
    cosine_similarity: float


@dataclass
class _BatchResult:
    items: list[_Item]
    wall_clock_seconds: float


def _load_precomputed(path: Path):
    """Wraps an already-computed baseline's JSON as a "reconstructor"
    callable matching app.eval's expected interface - ignores whatever
    embeddings it's called with and just returns the cached results.
    """
    with open(path) as f:
        data = json.load(f)
    items_by_id = {item["id"]: _Item(item["prompt"], item["cosine_similarity"]) for item in data["items"]}
    wall_clock = data["wall_clock_seconds"]

    def _reconstructor(_target_embeddings):
        return _BatchResult(items=list(items_by_id.values()), wall_clock_seconds=wall_clock)

    return _reconstructor, items_by_id


def _flatten_fields(result) -> str:
    """Render a decomposition result as a single prompt string, so it can be
    scored on the same footing as PEZ's hard prompt and the CLIP-tag
    baseline's tag concatenation. Empty when parsing failed - a low
    CLIP-score for that row is the correct, informative outcome.
    """
    if not result.is_valid:
        return ""
    fields = result.fields
    parts = [fields.subject]
    parts.extend(v for v in (fields.style, fields.medium, fields.lighting, fields.tone) if v)
    parts.extend(fields.modifiers)
    return ", ".join(parts)


def _build_decomposer_reconstructor(image_ids: list[str]):
    """Builds a "reconstructor" callable for the Task 5.3 LoRA decomposer.

    Unlike the precomputed baselines, this one runs live: it loads each
    image's evidence block from data/decomp_sft.jsonl (built by Navya's
    Task 5.2 pipeline), decomposes it, and scores the flattened result
    against the image embeddings passed in at call time.
    """
    with open(DECOMP_SFT_PATH, encoding="utf-8") as f:
        input_by_id = {row["id"]: row["input_text"] for row in (json.loads(line) for line in f)}
    input_texts = [input_by_id[i] for i in image_ids]

    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")

    def _reconstructor(target_embeddings):
        t0 = time.time()
        decompositions = decompose_batch(
            input_texts, use_adapter=True, adapter_dir=DECOMPOSER_ADAPTER_DIR
        )
        prompts = [_flatten_fields(r) for r in decompositions]

        model, _ = get_clip_model()
        with torch.no_grad():
            prompt_embeds = F.normalize(model.encode_text(tokenizer(prompts)), dim=-1)
        targets = target_embeddings if torch.is_tensor(target_embeddings) else torch.tensor(target_embeddings)
        similarities = (prompt_embeds.float() @ targets.float().T).diagonal()

        items = [
            _Item(prompt=prompt, cosine_similarity=float(sim))
            for prompt, sim in zip(prompts, similarities)
        ]
        return _BatchResult(items=items, wall_clock_seconds=time.time() - t0)

    return _reconstructor


def main() -> None:
    with open(TEST_SPLIT_PATH) as f:
        test_ids = json.load(f)[:NUM_TEST_IMAGES]
    rows_by_id = {r["id"]: r for r in pq.read_table(PAIRS_PATH).to_pylist()}
    rows = [rows_by_id[i] for i in test_ids]
    original_prompts = [r["prompt"] for r in rows]
    image_ids = [r["id"] for r in rows]

    pez_fn, pez_items = _load_precomputed(RESULTS_DIR / "pez_baseline.json")
    cliptag_fn, cliptag_items = _load_precomputed(RESULTS_DIR / "cliptag_baseline.json")
    assert list(pez_items) == image_ids, "pez_baseline.json image order doesn't match test split"
    assert list(cliptag_items) == image_ids, "cliptag_baseline.json image order doesn't match test split"
    decomposer_fn = _build_decomposer_reconstructor(image_ids)

    print(f"Embedding {len(rows)} test images (needed to score the decomposer live)...")
    image_bytes = [(IMAGES_DIR / r["image_path"]).read_bytes() for r in rows]
    target_embeddings = torch.from_numpy(embed_images(image_bytes))

    print("Scoring PEZ, CLIP-tag, and the LoRA decomposer through the same harness (computing BERTScore)...")
    eval_rows = run_eval(
        {"pez": pez_fn, "clip_tag": cliptag_fn, "decomposer": decomposer_fn},
        image_ids=image_ids,
        original_prompts=original_prompts,
        target_embeddings=target_embeddings,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "baselines.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(vars(eval_rows[0])))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow(vars(row))
    print(f"Saved -> {csv_path}")

    by_name: dict[str, list] = {}
    for row in eval_rows:
        by_name.setdefault(row.reconstructor, []).append(row)

    md_path = RESULTS_DIR / "baselines.md"
    with open(md_path, "w") as f:
        f.write("# Reconstruction Method Evaluation\n\n")
        f.write(
            f"PEZ, the naive CLIP-tag baseline, and the LoRA decomposition model scored through the "
            f"same harness (`ml/app/eval.py`) on the same {len(rows)} test-split images.\n\n"
        )
        f.write(
            "**Decomposer note:** its output is structured JSON (subject/style/medium/lighting/"
            "modifiers/tone/negative_constraints), flattened into a single string here so it can be "
            "scored on the same footing as the other two methods. Component-wise precision/recall/F1 "
            "against weak structured labels is a separate, complementary evaluation - see "
            "`docs/results/decomposer_eval.md`.\n\n"
        )
        f.write("## Summary\n\n")
        f.write("| Reconstructor | Mean CLIP-score | Mean BERTScore F1 | Mean latency (s/image) |\n")
        f.write("|---|---|---|---|\n")
        for name, group in by_name.items():
            n = len(group)
            mean_clip = sum(r.clip_score for r in group) / n
            mean_bert_f1 = sum(r.bertscore_f1 for r in group) / n
            mean_latency = sum(r.latency_seconds for r in group) / n
            print(f"{name}: mean CLIP-score={mean_clip:.4f}, mean BERTScore F1={mean_bert_f1:.4f}")
            f.write(f"| {name} | {mean_clip:.4f} | {mean_bert_f1:.4f} | {mean_latency:.2f} |\n")

        f.write("\n## Per-image detail\n\n")
        f.write("| Reconstructor | Image ID | CLIP-score | BERTScore F1 | Generated prompt |\n")
        f.write("|---|---|---|---|---|\n")
        for row in eval_rows:
            prompt = row.generated_prompt.replace("|", "/")
            f.write(
                f"| {row.reconstructor} | {row.image_id} | {row.clip_score:.4f} | "
                f"{row.bertscore_f1:.4f} | {prompt} |\n"
            )

    print(f"Saved -> {md_path}")


if __name__ == "__main__":
    main()
