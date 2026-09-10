"""Construction of supervised fine-tuning examples for the decomposer.

Each example pairs an evidence block with a structured target:

    input  = {caption, top-k retrieved prompts, PEZ prompt}
    target = the weak-labeled structured JSON of the image's true prompt

The three input signals are complementary: the caption is image-grounded but
carries little style or medium vocabulary; the retrieved prompts carry that
vocabulary in the idiom prompt authors use, but describe a different image;
the PEZ prompt is CLIP's own reading of the image in real vocabulary tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from app.caption import strip_caption_boilerplate
from app.schemas.structured_fields import StructuredFields
from app.weak_label import LEXICONS, match_segment

# Bounded because prompts in this corpus are long and the decomposer's context
# is shared with the caption and PEZ evidence.
DEFAULT_TOP_K = 5

TARGET_KEY_ORDER = [
    "subject",
    "style",
    "medium",
    "lighting",
    "modifiers",
    "tone",
    "negative_constraints",
]

PEZ_UNAVAILABLE = "(unavailable)"

INSTRUCTION = (
    "You reconstruct the prompt behind an AI-generated image. Given the evidence "
    "below, output the most likely structured prompt as a single JSON object with "
    "exactly these keys: subject, style, medium, lighting, modifiers, tone, "
    "negative_constraints. Use null for a field the evidence does not support, and "
    "[] for an empty list. Output only the JSON object."
)


# --- Target cleaning ------------------------------------------------------

def filter_modifiers(modifiers: list[str]) -> list[str]:
    """Keep only modifiers that a lexicon recognises.

    The weak labeller routes every unclassified segment into `modifiers`, so
    the raw field mixes genuine quality modifiers ("highly detailed", "8 k")
    with scene detail belonging in `subject` ("lava streams") - measured at
    0.26 field-level precision in docs/label-quality.md. Training on it
    unfiltered teaches the model to reproduce that noise.

    Artist credits are also dropped, since the schema has no field for them.
    Callers retain the unfiltered list so the decision remains ablatable.
    """
    return [m for m in modifiers if match_segment(m)]


def clean_target(fields: StructuredFields) -> StructuredFields:
    """Return `fields` with noisy modifier entries removed."""
    return fields.model_copy(update={"modifiers": filter_modifiers(fields.modifiers)})


def render_target(fields: StructuredFields) -> str:
    """Serialize a target to the canonical JSON string the model must emit.

    Fixed key order and no padding whitespace: the target is a single exact
    surface form, which downstream constrained decoding and the
    schema-validity metric both compare against.
    """
    payload = {key: getattr(fields, key) for key in TARGET_KEY_ORDER}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# --- Input assembly -------------------------------------------------------

def render_input(
    caption: str, retrieved_prompts: list[str], pez_prompt: Optional[str]
) -> str:
    """Render the evidence block the model is conditioned on.

    The PEZ line is always emitted, reading "(unavailable)" when absent rather
    than being omitted, so that a row without PEZ is not a structurally
    different prompt. PEZ is the slowest stage in the pipeline and may be
    skipped for latency at serving time, so the model must tolerate its
    absence.
    """
    lines = [INSTRUCTION, "", f"Caption: {caption}", "", "Similar known prompts:"]
    if retrieved_prompts:
        lines += [f"{i}. {p}" for i, p in enumerate(retrieved_prompts, 1)]
    else:
        lines.append("(none)")
    lines += ["", f"PEZ baseline: {pez_prompt or PEZ_UNAVAILABLE}", "", "JSON:"]
    return "\n".join(lines)


@dataclass
class SftExample:
    """A single serialized training example."""

    id: str
    split: str
    caption: str      # boilerplate-stripped; the text the model reads
    caption_raw: str  # verbatim captioner output, retained for ablation
    retrieved_ids: list[str]
    retrieved_prompts: list[str]
    retrieved_similarities: list[float]
    pez_prompt: Optional[str]
    target: StructuredFields
    true_prompt: str
    modifiers_raw: list[str] = field(default_factory=list)
    # Set when this row's own image was found among its candidates and
    # removed; recorded so callers can verify the guard fired.
    self_match_dropped: bool = False
    # Set when the true prompt appears verbatim inside a neighbour's prompt.
    # Not leakage (see build_example), but it bounds how much of the target is
    # reachable by copying rather than synthesis.
    true_prompt_in_retrieved: bool = False

    def to_row(self) -> dict:
        return {
            "id": self.id,
            "split": self.split,
            # The rendered strings are what training consumes.
            "input_text": render_input(
                self.caption, self.retrieved_prompts, self.pez_prompt
            ),
            "target_text": render_target(self.target),
            # Structured mirror of the same data, so the prompt can be
            # re-rendered or an input ablated without rebuilding the dataset.
            "input": {
                "caption": self.caption,
                "caption_raw": self.caption_raw,
                "retrieved_ids": self.retrieved_ids,
                "retrieved_prompts": self.retrieved_prompts,
                "retrieved_similarities": self.retrieved_similarities,
                "pez_prompt": self.pez_prompt,
            },
            "target": {key: getattr(self.target, key) for key in TARGET_KEY_ORDER},
            # Reference only, never an input: used for scoring and debugging.
            "true_prompt": self.true_prompt,
            "modifiers_raw": self.modifiers_raw,
            "self_match_dropped": self.self_match_dropped,
            "true_prompt_in_retrieved": self.true_prompt_in_retrieved,
        }


def build_example(
    dataset_id: str,
    split: str,
    caption: str,
    candidates: list[dict],
    pez_prompt: Optional[str],
    weak_fields: StructuredFields,
    true_prompt: str,
    top_k: int = DEFAULT_TOP_K,
) -> SftExample:
    """Build one training example from an image's evidence and weak labels.

    Drops the row's own image from its candidates. The retrieval index is
    built from the training images, so a training query returns itself at rank
    1 with similarity ~1.0, carrying the exact prompt its target was derived
    from. Training on that teaches the model to copy the first candidate and
    ignore the other evidence - a shortcut that does not exist at inference,
    where the query image is never indexed.

    Near-duplicate neighbours are deliberately kept. A different image with a
    similar or even containing prompt is legitimate retrieval that also occurs
    at inference, so filtering it would make training inputs systematically
    weaker than production. Only exact self-matches are removed.
    """
    # The captioner stores raw output; stripping its contentless opening
    # phrase is the caller's choice, exercised here.
    stripped_caption = strip_caption_boilerplate(caption)
    self_match = any(c["id"] == dataset_id for c in candidates)
    kept = [c for c in candidates if c["id"] != dataset_id][:top_k]
    cleaned = clean_target(weak_fields)
    return SftExample(
        id=dataset_id,
        split=split,
        caption=stripped_caption,
        caption_raw=caption,
        retrieved_ids=[c["id"] for c in kept],
        retrieved_prompts=[c["prompt"] for c in kept],
        retrieved_similarities=[round(float(c["similarity"]), 6) for c in kept],
        pez_prompt=pez_prompt,
        target=cleaned,
        true_prompt=true_prompt,
        modifiers_raw=list(weak_fields.modifiers),
        self_match_dropped=self_match,
        true_prompt_in_retrieved=any(
            true_prompt and true_prompt in c["prompt"] for c in kept
        ),
    )


def validate_row(row: dict) -> StructuredFields:
    """Validate a serialized row, raising on any violation.

    Enforces the target contract at the artifact level rather than in memory
    only: `target_text` must parse, carry exactly the schema's keys in
    canonical order, and satisfy StructuredFields.
    """
    for key in ("id", "split", "input_text", "target_text", "target", "true_prompt"):
        if key not in row:
            raise ValueError(f"row {row.get('id')!r} missing required key {key!r}")

    parsed = json.loads(row["target_text"])
    if list(parsed) != TARGET_KEY_ORDER:
        raise ValueError(
            f"row {row['id']!r} target_text keys {list(parsed)} != {TARGET_KEY_ORDER}"
        )
    fields = StructuredFields(**parsed)  # extra="forbid" + required subject

    if parsed != row["target"]:
        raise ValueError(f"row {row['id']!r} target_text and target disagree")
    if row["target_text"] != render_target(fields):
        raise ValueError(f"row {row['id']!r} target_text is not the canonical rendering")

    # Checks identity, not string similarity: a different image with a
    # containing prompt is legitimate retrieval, not leakage.
    if row["id"] in row["input"].get("retrieved_ids", []):
        raise ValueError(
            f"row {row['id']!r} retrieves itself - self-match leakage guard failed"
        )
    return fields


def lexicon_vocabulary_size() -> int:
    """Total number of curated lexicon phrases."""
    return sum(len(set(v)) for v in LEXICONS.values())
