"""Structured prompt decomposition model.

Wraps the LoRA-adapted decomposer that turns an evidence block (caption,
retrieved prompts, PEZ output) into StructuredFields.

Base model: SmolLM2-135M-Instruct with LoRA on the attention q/v projections.
Two constraints drove this choice and are worth knowing before changing it:

  - 4-bit quantization via bitsandbytes is CUDA-only, so the 7B-class models
    the design originally called for need ~14 GB at bf16 on CPU. They do not
    fit.
  - T5-family models cannot be used at all: `{` and `}` are absent from their
    SentencePiece vocabulary and map to `<unk>`, making JSON output
    impossible. test_decompose.py guards against reintroducing this.

The base is instruction-tuned, which keeps the untrained control meaningful,
and decoder-only, which is what grammar-constrained decoding libraries target.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.schemas.structured_fields import StructuredFields

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
ADAPTER_DIR = Path(
    os.environ.get("DECOMPOSER_ADAPTER", str(DATA_DIR / "models" / "decomposer-lora"))
)

BASE_MODEL = os.environ.get("DECOMPOSER_BASE", "HuggingFaceTB/SmolLM2-135M-Instruct")
# Prompt and target share one sequence. The observed worst case is ~850
# tokens, so this leaves headroom without padding cost.
MAX_SEQ_TOKENS = int(os.environ.get("DECOMPOSER_MAX_SEQ", "1024"))
MAX_TARGET_TOKENS = int(os.environ.get("DECOMPOSER_MAX_TARGET", "224"))
NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "12"))

# Imported by the trainer so a saved adapter and this loader cannot disagree
# about the architecture.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

_cache: dict[str, tuple] = {}

# An untrained model has never seen this output contract, so a zero-shot
# control mostly measures whether it can guess the format rather than whether
# it can do the task. Showing the format without training the task separates
# the two, which is what makes the comparison interpretable.
FEWSHOT_PREAMBLE = """Here are two examples of the required output.

JSON: {"subject":"a red fox in a snowy forest","style":"anime","medium":"oil painting","lighting":"golden hour","modifiers":["highly detailed","8 k"],"tone":"serene","negative_constraints":[]}

JSON: {"subject":"portrait of a knight","style":null,"medium":"digital painting","lighting":"cinematic lighting","modifiers":["sharp focus"],"tone":null,"negative_constraints":[]}

Now do the same for the following.

"""


def to_fewshot(input_text: str) -> str:
    """Prepend format demonstrations to an SFT input block."""
    return FEWSHOT_PREAMBLE + input_text


@dataclass
class DecompositionResult:
    """The outcome of one decomposition attempt.

    `fields` is None exactly when the output could not be parsed into a
    schema-valid object. Parse failures are surfaced rather than silently
    substituted so that schema-validity remains measurable; callers apply
    their own fallback.
    """

    raw_output: str
    fields: Optional[StructuredFields]
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.fields is not None


def build_prompt(tokenizer, input_text: str) -> str:
    """Render an evidence block through the model's chat template.

    Applied identically at training and inference time; diverging would
    silently shift the input distribution the adapter was trained on.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(adapter_dir: Optional[Path] = ADAPTER_DIR, use_adapter: bool = True):
    """Load the decomposer, or the unadapted base when `use_adapter` is False.

    Results are cached per configuration; callers evaluating several variants
    should clear `_cache` between them, since holding two copies of the model
    exceeds available memory.
    """
    torch.set_num_threads(NUM_THREADS)
    key = f"{BASE_MODEL}|{adapter_dir}|{use_adapter}"
    if key in _cache:
        return _cache[key]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)

    if use_adapter:
        if adapter_dir is None or not Path(adapter_dir).exists():
            raise FileNotFoundError(
                f"LoRA adapter not found at {adapter_dir} - train it first "
                "(make train-decomposer), or pass use_adapter=False for the "
                "zero-shot control."
            )
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

    model.eval()
    _cache[key] = (model, tokenizer)
    return model, tokenizer


def parse_output(text: str) -> DecompositionResult:
    """Parse model output into StructuredFields.

    Tolerant of surrounding noise - the outermost {...} span is extracted, so
    prose or code fences around the object are ignored. Strict about content:
    the result must be valid JSON satisfying StructuredFields, which forbids
    extra keys and requires a non-empty subject.
    """
    raw = text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return DecompositionResult(raw, None, "no JSON object found in output")

    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        return DecompositionResult(raw, None, f"invalid JSON: {e}")
    if not isinstance(payload, dict):
        return DecompositionResult(raw, None, "JSON is not an object")

    try:
        return DecompositionResult(raw, StructuredFields(**payload))
    except Exception as e:
        return DecompositionResult(raw, None, f"schema violation: {type(e).__name__}: {e}")


def decompose_batch(
    input_texts: list[str],
    use_adapter: bool = True,
    adapter_dir: Optional[Path] = ADAPTER_DIR,
    batch_size: int = 4,
    num_beams: int = 1,
    fewshot: bool = False,
) -> list[DecompositionResult]:
    """Decompose rendered evidence blocks into structured fields.

    Greedy decoding by default: the target is a single canonical string, so
    beam search adds cost without meaningfully improving output.
    """
    model, tokenizer = load_model(adapter_dir=adapter_dir, use_adapter=use_adapter)
    if fewshot:
        input_texts = [to_fewshot(t) for t in input_texts]

    # Batched decoder-only generation requires left padding. With right
    # padding, pad tokens sit between the prompt and the first generated
    # token, so every shorter row in the batch decodes from invalid context.
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    results: list[DecompositionResult] = []
    try:
        for start in range(0, len(input_texts), batch_size):
            chunk = [
                build_prompt(tokenizer, text)
                for text in input_texts[start : start + batch_size]
            ]
            encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_TOKENS - MAX_TARGET_TOKENS,
                add_special_tokens=False,  # the chat template already adds them
            )
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=MAX_TARGET_TOKENS,
                    num_beams=num_beams,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # generate() returns prompt + completion.
            prompt_len = encoded["input_ids"].shape[1]
            for sequence in generated:
                completion = sequence[prompt_len:]
                results.append(
                    parse_output(tokenizer.decode(completion, skip_special_tokens=True))
                )
    finally:
        tokenizer.padding_side = original_side

    return results
