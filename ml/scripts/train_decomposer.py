#!/usr/bin/env python3
"""Fine-tune the decomposition model with LoRA.

Trains the base model configured in app.decompose on data/decomp_sft.jsonl.

Outputs:
    data/models/decomposer-lora/          the saved adapter
    data/results/train_decomposer.json    per-step train loss, per-epoch val loss
    data/results/train_decomposer.md      training curve summary

Implementation notes:

  - Only target tokens contribute to the loss; prompt positions are masked to
    -100. Without this the model spends capacity reproducing the evidence
    block, which it is given anyway.
  - Length-grouped batching. Sequences vary widely, so batching at random pads
    almost everything to the longest example. Shuffling into megabatches and
    sorting within them cuts wasted compute while keeping the ordering
    stochastic across epochs.
  - The adapter saved is the one with the lowest validation loss, not the one
    from the final step, since overfitting is the expected failure mode on a
    dataset this size.

Usage:
    python -m scripts.train_decomposer
    python -m scripts.train_decomposer --limit 16 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from app.decompose import (
    BASE_MODEL,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_TOKENS,
    MAX_TARGET_TOKENS,
    NUM_THREADS,
    build_prompt,
)

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SFT_PATH = DATA_DIR / "decomp_sft.jsonl"
ADAPTER_OUT = DATA_DIR / "models" / "decomposer-lora"
RESULTS_DIR = DATA_DIR / "results"
LOG_JSON = RESULTS_DIR / "train_decomposer.json"
LOG_MD = RESULTS_DIR / "train_decomposer.md"

SEED = 1337


def load_rows(limit: int | None) -> tuple[list[dict], list[dict]]:
    if not SFT_PATH.exists():
        raise SystemExit(f"{SFT_PATH} not found - run `make decomp-sft` first.")
    rows = [json.loads(line) for line in open(SFT_PATH)]
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    if limit:
        train, val = train[:limit], val[: max(2, limit // 4)]
    return train, val


def encode_example(tokenizer, row: dict) -> tuple[list[int], list[int]]:
    """Tokenize one row into (input_ids, labels) with the prompt masked out.

    The target is tokenized separately and never truncated - only the prompt
    is trimmed if the pair would exceed MAX_SEQ_TOKENS. Truncating the
    combined string instead would silently cut the *end*, which is the target,
    and train the model on incomplete JSON.
    """
    prompt_ids = tokenizer(
        build_prompt(tokenizer, row["input_text"]), add_special_tokens=False
    ).input_ids
    target_ids = tokenizer(
        row["target_text"] + tokenizer.eos_token, add_special_tokens=False
    ).input_ids
    target_ids = target_ids[:MAX_TARGET_TOKENS]

    budget = MAX_SEQ_TOKENS - len(target_ids)
    if len(prompt_ids) > budget:
        # Keep the tail of the prompt: it holds the retrieved prompts, the PEZ
        # line and the "JSON:" cue, which matter more than the leading
        # boilerplate instruction the model sees on every single row.
        prompt_ids = prompt_ids[-budget:]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + list(target_ids)
    return input_ids, labels


def make_batches(
    rows: list[dict], tokenizer, batch_size: int, shuffle: bool, rng: random.Random
) -> list[dict]:
    """Tokenize into length-grouped, right-padded batches.

    Right padding is correct for training (the loss ignores pad positions via
    -100); generation is the case that needs left padding, handled in
    app.decompose.decompose_batch.
    """
    encoded = [encode_example(tokenizer, row) for row in rows]

    if shuffle:
        rng.shuffle(encoded)
        # Megabatches of 8x, sorted inside, so padding stays small while the
        # ordering still differs every epoch.
        mega = batch_size * 8
        grouped: list[tuple[list[int], list[int]]] = []
        for start in range(0, len(encoded), mega):
            chunk = encoded[start : start + mega]
            chunk.sort(key=lambda pair: len(pair[0]))
            grouped.extend(chunk)
        encoded = grouped
    else:
        encoded.sort(key=lambda pair: len(pair[0]))

    pad_id = tokenizer.pad_token_id
    batches = []
    for start in range(0, len(encoded), batch_size):
        chunk = encoded[start : start + batch_size]
        width = max(len(ids) for ids, _ in chunk)
        input_ids, attention_mask, labels = [], [], []
        for ids, lab in chunk:
            padding = width - len(ids)
            input_ids.append(ids + [pad_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            labels.append(lab + [-100] * padding)
        batches.append(
            {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
            }
        )

    if shuffle:
        rng.shuffle(batches)  # decorrelate the length ordering across steps
    return batches


@torch.no_grad()
def evaluate(model, batches) -> float:
    """Mean val loss, weighted by the number of scored target tokens."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        out = model(**batch)
        n = int((batch["labels"] != -100).sum())
        total_loss += float(out.loss) * n
        total_tokens += n
    model.train()
    return total_loss / total_tokens if total_tokens else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, help="effective batch = 4x4 = 16")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--log-every", type=int, default=10, help="optimizer steps")
    parser.add_argument("--limit", type=int, default=None, help="smoke-run row cap")
    args = parser.parse_args()

    torch.set_num_threads(NUM_THREADS)
    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    train_rows, val_rows = load_rows(args.limit)
    print(f"Train rows: {len(train_rows)} | val rows: {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Base: {BASE_MODEL} | trainable {trainable/1e6:.2f}M / {total/1e6:.1f}M "
        f"({trainable/total*100:.2f}%)"
    )

    val_batches = make_batches(val_rows, tokenizer, args.batch_size, shuffle=False, rng=rng)

    steps_per_epoch = math.ceil(
        math.ceil(len(train_rows) / args.batch_size) / args.grad_accum
    )
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    print(f"Optimizer steps: {total_steps} ({steps_per_epoch}/epoch)")

    model.train()
    history: list[dict] = []
    best_val = float("inf")
    best_epoch = -1
    t0 = time.time()
    step = 0

    # Val loss before any training - the honest zero-shot starting point.
    initial_val = evaluate(model, val_batches)
    print(f"Initial val loss (untrained adapter): {initial_val:.4f}")
    history.append({"event": "initial", "val_loss": initial_val})

    for epoch in range(1, args.epochs + 1):
        batches = make_batches(train_rows, tokenizer, args.batch_size, shuffle=True, rng=rng)
        running, running_n = 0.0, 0

        for i, batch in enumerate(batches, 1):
            out = model(**batch)
            # Scale so gradient accumulation averages rather than sums.
            (out.loss / args.grad_accum).backward()
            running += float(out.loss)
            running_n += 1

            if i % args.grad_accum == 0 or i == len(batches):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % args.log_every == 0:
                    mean_loss = running / max(running_n, 1)
                    elapsed = time.time() - t0
                    print(
                        f"  epoch {epoch} step {step}/{total_steps} "
                        f"train_loss {mean_loss:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                        f"({elapsed/60:.1f} min)",
                        flush=True,
                    )
                    history.append(
                        {
                            "event": "train",
                            "epoch": epoch,
                            "step": step,
                            "train_loss": mean_loss,
                            "lr": scheduler.get_last_lr()[0],
                            "elapsed_min": elapsed / 60,
                        }
                    )
                    running, running_n = 0.0, 0

        val_loss = evaluate(model, val_batches)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch} done | val_loss {val_loss:.4f} | {elapsed/60:.1f} min",
            flush=True,
        )
        history.append(
            {"event": "val", "epoch": epoch, "val_loss": val_loss, "elapsed_min": elapsed / 60}
        )

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ADAPTER_OUT))
            print(f"  new best val loss - adapter saved to {ADAPTER_OUT}", flush=True)
        else:
            print(
                f"  val loss did not improve on epoch {best_epoch} "
                f"({best_val:.4f}) - keeping that adapter",
                flush=True,
            )

    wall_min = (time.time() - t0) / 60
    print(f"\nDone in {wall_min:.1f} min. Best val loss {best_val:.4f} at epoch {best_epoch}.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_JSON, "w") as f:
        json.dump(
            {
                "base_model": BASE_MODEL,
                "lora": {
                    "r": LORA_R,
                    "alpha": LORA_ALPHA,
                    "dropout": LORA_DROPOUT,
                    "target_modules": LORA_TARGET_MODULES,
                },
                "trainable_params": trainable,
                "total_params": total,
                "hyperparameters": vars(args),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "initial_val_loss": initial_val,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "wall_clock_minutes": wall_min,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"Saved -> {LOG_JSON}")

    val_points = [h for h in history if h["event"] == "val"]
    train_points = [h for h in history if h["event"] == "train"]
    with open(LOG_MD, "w") as f:
        f.write("# Decomposer LoRA Training - Task 5.3\n\n")
        f.write(f"- **Base model:** `{BASE_MODEL}`\n")
        f.write(
            f"- **LoRA:** r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}, "
            f"targets={LORA_TARGET_MODULES}\n"
        )
        f.write(
            f"- **Trainable:** {trainable/1e6:.2f}M / {total/1e6:.1f}M "
            f"({trainable/total*100:.2f}%)\n"
        )
        f.write(
            f"- **Data:** {len(train_rows)} train / {len(val_rows)} val "
            "(prompt-disjoint splits from Task 1.2)\n"
        )
        f.write(
            f"- **Schedule:** {args.epochs} epochs, batch {args.batch_size} x "
            f"grad-accum {args.grad_accum} (effective {args.batch_size*args.grad_accum}), "
            f"lr {args.lr}, linear decay with {args.warmup_ratio:.0%} warmup\n"
        )
        f.write(f"- **Wall clock:** {wall_min:.1f} min on CPU ({NUM_THREADS} threads)\n")
        f.write(
            f"- **Val loss:** {initial_val:.4f} untrained -> **{best_val:.4f}** "
            f"best (epoch {best_epoch})\n\n"
        )
        f.write("## Val loss by epoch\n\n| Epoch | Val loss |\n|---|---|\n")
        f.write(f"| 0 (untrained) | {initial_val:.4f} |\n")
        for point in val_points:
            marker = " **(best, saved)**" if point["epoch"] == best_epoch else ""
            f.write(f"| {point['epoch']} | {point['val_loss']:.4f}{marker} |\n")
        if train_points:
            f.write("\n## Train loss curve\n\n| Step | Train loss | LR |\n|---|---|---|\n")
            for point in train_points:
                f.write(
                    f"| {point['step']} | {point['train_loss']:.4f} | {point['lr']:.2e} |\n"
                )
    print(f"Saved -> {LOG_MD}")


if __name__ == "__main__":
    main()
