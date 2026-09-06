#!/usr/bin/env python3
"""Environment + GPU check.

Run this after `pip install -r requirements.txt` (or inside the ml container)
to find out what hardware budget the rest of the project has to work with.
This gates real scope decisions: LoRA fine-tuning, CLIP contrastive
fine-tuning, and Stable Diffusion regeneration all need a GPU with real VRAM
headroom; retrieval, embeddings, and baseline reconstructors are CPU-feasible,
just slower.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass
from typing import Optional


def section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def human_bytes(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def check_python_platform() -> None:
    section("Platform")
    print(f"Python:   {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")


def check_torch():
    section("PyTorch / CUDA")
    try:
        import torch
    except ImportError as e:
        print(f"FAIL: could not import torch ({e}). Run `pip install -r requirements.txt` first.")
        sys.exit(1)

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        print(f"CUDA version (torch built against): {torch.version.cuda}")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(
                f"  [{i}] {props.name} - {human_bytes(props.total_memory)} VRAM "
                f"(compute capability {props.major}.{props.minor})"
            )
    else:
        print("No CUDA device detected - running on CPU only.")
        print("This gates scope: LoRA fine-tuning, CLIP contrastive fine-tuning, and")
        print("Stable Diffusion regeneration need GPU compute to be practical. Embedding")
        print("extraction, FAISS retrieval, and the baseline reconstructors remain")
        print("feasible on CPU, just slower.")

    return torch


def check_system_memory() -> None:
    section("System memory (relevant when running on CPU)")
    try:
        import psutil  # not a pinned dependency - fall back below if unavailable

        vm = psutil.virtual_memory()
        print(f"Total RAM:     {human_bytes(vm.total)}")
        print(f"Available RAM: {human_bytes(vm.available)}")
        return
    except ImportError:
        pass

    try:
        with open("/proc/meminfo") as f:
            meminfo = {
                line.split(":")[0]: line.split(":")[1].strip()
                for line in f.readlines()
                if ":" in line
            }
        total_kb = int(meminfo["MemTotal"].split()[0])
        avail_kb = int(meminfo.get("MemAvailable", "0").split()[0])
        print(f"Total RAM:     {human_bytes(total_kb * 1024)}")
        print(f"Available RAM: {human_bytes(avail_kb * 1024)}")
    except Exception:
        print("Could not determine system memory (psutil not installed, /proc/meminfo unavailable).")


@dataclass
class ClipLoadResult:
    ok: bool
    device: str
    param_count: int
    param_bytes: int
    peak_bytes: Optional[int]
    load_seconds: float
    forward_seconds: float
    error: Optional[str] = None


def check_clip_vit_l14(torch_module) -> ClipLoadResult:
    section("Loading CLIP ViT-L/14 (via OpenCLIP)")
    torch = torch_module

    try:
        import open_clip
    except ImportError as e:
        print(f"FAIL: could not import open_clip_torch ({e}).")
        return ClipLoadResult(False, "n/a", 0, 0, None, 0.0, 0.0, error=str(e))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target device: {device}")
    print("Loading openai ViT-L-14 weights (first run downloads ~890MB)...")

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    try:
        # 'ViT-L-14-quickgelu' (not plain 'ViT-L-14') is the correct config for the
        # OpenAI-trained weights, which use QuickGELU activation - the plain config
        # silently mismatches activation and would quietly degrade embedding quality.
        model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14-quickgelu", pretrained="openai", device=device
        )
        model.eval()
    except Exception as e:  # noqa: BLE001 - report any load failure, don't crash the check
        print(f"FAIL: model failed to load: {e}")
        return ClipLoadResult(False, device, 0, 0, None, time.time() - t0, 0.0, error=str(e))
    load_seconds = time.time() - t0

    param_count = sum(p.numel() for p in model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(
        f"Loaded in {load_seconds:.1f}s - {param_count / 1e6:.1f}M params, "
        f"{human_bytes(param_bytes)} in weights"
    )

    # Dummy forward pass to confirm it actually fits and runs, not just loads.
    print("Running a dummy forward pass (batch of 4, 224x224 image)...")
    t0 = time.time()
    try:
        with torch.no_grad():
            dummy = torch.randn(4, 3, 224, 224, device=device)
            _ = model.encode_image(dummy)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: forward pass failed (likely OOM): {e}")
        return ClipLoadResult(
            False, device, param_count, param_bytes, None, load_seconds, time.time() - t0, error=str(e)
        )
    forward_seconds = time.time() - t0

    peak_bytes = None
    if device == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated()
        print(f"Peak CUDA memory allocated: {human_bytes(peak_bytes)}")
    print(f"Forward pass OK in {forward_seconds:.2f}s")

    return ClipLoadResult(True, device, param_count, param_bytes, peak_bytes, load_seconds, forward_seconds)


def print_summary(result: ClipLoadResult) -> None:
    section("Summary")
    if not result.ok:
        print("RESULT: FAIL - ViT-L/14 could not be loaded/run. See error above.")
        print(f"Error: {result.error}")
        sys.exit(1)

    print(f"RESULT: PASS - ViT-L/14 loads and runs on '{result.device}'.")
    if result.device == "cpu":
        print("Running on CPU: fine for embedding extraction at modest throughput.")
        print("LoRA fine-tuning, CLIP contrastive fine-tuning, and SD-based regeneration")
        print("should be treated as stretch/cloud-only goals unless a GPU machine becomes")
        print("available - flag this to the team now, before planning those phases.")
    else:
        print(f"Peak VRAM during a batch-of-4 forward pass: {human_bytes(result.peak_bytes)}")
        print("Compare this against the total VRAM printed above to judge headroom for")
        print("larger batches, the LoRA fine-tune, and running SD for regeneration.")


def main() -> None:
    check_python_platform()
    torch = check_torch()
    check_system_memory()
    result = check_clip_vit_l14(torch)
    print_summary(result)


if __name__ == "__main__":
    main()
