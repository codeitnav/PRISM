"""Rule-based extraction of structured fields from raw prompts.

Parses a prompt into the StructuredFields schema using segmentation plus
curated lexicons. The output is intentionally weak supervision: cheap and
noisy, produced without human annotation, for use as model training targets
and as the reference for component-wise scoring. Measured per-field precision
is recorded in docs/label-quality.md.

How a prompt is parsed:

 1. Split on the separators these prompts actually use - commas, sentence
    periods, and the "|" / "::" weighting syntax from Midjourney and
    AUTOMATIC1111.
 2. Strip prompt-engineering noise: attention weights ("(detailed:1.3)"),
    bracket emphasis, trailing CLI flags, and artist-credit prefixes.
 3. Classify each segment against the lexicons below, longest phrase first.
 4. Assign in two passes - explicitly tagged segments claim the single-slot
    fields first, then phrases found inside longer content-bearing segments
    backfill whatever remains empty.
 5. Take negative constraints from explicit negative syntax: a leading "no" /
    "without", or the "--neg" / "negative prompt:" markers.

The lexicons are curated by hand and extended from corpus frequency via
mine_lexicon_candidates(), which surfaces the most common segments no lexicon
classifies.

An optional LLM-labelled seed set can override the rule output per prompt via
load_llm_seed_labels(); it is absent by default, so labels are rule-derived.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from app.schemas.structured_fields import StructuredFields

# --- Lexicons -------------------------------------------------------------
#
# Entries are lowercase and matching is case-insensitive. Ordering by phrase
# length happens at match time, so these lists stay readable and appendable.

STYLE_LEXICON = [
    "studio ghibli", "art nouveau", "art deco", "ukiyo-e", "vaporwave",
    "synthwave", "retrofuturism", "cyberpunk", "steampunk", "dieselpunk",
    "solarpunk", "photorealistic", "hyperrealistic", "hyper realistic",
    "realistic", "surrealism", "surreal", "impressionism", "impressionist",
    "expressionism", "expressionist", "cubism", "cubist", "baroque",
    "renaissance", "romanticism", "gothic", "brutalist", "bauhaus",
    "minimalist", "minimalism", "maximalist", "abstract", "psychedelic",
    "pop art", "op art", "outrun", "low poly", "isometric", "flat design",
    "cel shaded", "cel shading", "anime", "manga", "cartoon", "comic book",
    "graphic novel", "concept art", "character design", "fantasy art",
    "dark fantasy", "high fantasy", "sci-fi", "science fiction", "noir",
    "film noir", "cottagecore", "kawaii", "chibi", "pixel art", "voxel art",
    "vector art", "fauvism", "pointillism", "trompe l'oeil", "vintage",
    "retro", "art brut", "naive art", "folk art", "tribal", "baroque punk",
    # Added from corpus-frequency mining:
    "fantasy", "hyper realism", "realism", "fine art", "digital fantasy",
    "traditional drawing style", "warframe", "dark souls", "studio trigger",
]

MEDIUM_LEXICON = [
    "oil painting", "oil on canvas", "acrylic painting", "watercolor painting",
    "watercolour painting", "watercolor", "watercolour", "gouache",
    "digital painting", "digital art", "digital illustration", "matte painting",
    "speed painting", "pencil sketch", "pencil drawing", "graphite drawing",
    "charcoal drawing", "charcoal sketch", "ink drawing", "ink illustration",
    "pen and ink", "linocut", "woodcut", "engraving", "etching", "lithograph",
    "screen print", "silkscreen", "airbrush", "spray paint", "mural",
    "fresco", "mosaic", "stained glass", "tapestry", "collage", "mixed media",
    "sculpture", "marble sculpture", "bronze sculpture", "clay sculpture",
    "claymation", "stop motion", "papercraft", "origami", "embroidery",
    "3d render", "3d rendering", "octane render", "unreal engine",
    "unreal engine 5", "blender render", "raytraced", "ray tracing",
    "photograph", "photography", "polaroid", "daguerreotype", "tintype",
    "35mm film", "35mm photograph", "medium format", "macro photograph",
    "aerial photograph", "drone photograph", "long exposure", "tilt shift",
    "cinematography", "film still", "movie still", "screenshot", "line art",
    "lineart", "storyboard", "blueprint", "technical drawing", "cross stitch",
    # Added from corpus-frequency mining:
    "illustration", "oil pastels", "oil pastel", "textured canvas",
    "digital illustration", "3d model", "photo manipulation",
]

LIGHTING_LEXICON = [
    "cinematic lighting", "dramatic lighting", "volumetric lighting",
    "volumetric light", "god rays", "crepuscular rays", "rim lighting",
    "rim light", "backlighting", "backlit", "silhouette lighting",
    "studio lighting", "softbox lighting", "soft lighting", "soft light",
    "hard lighting", "harsh lighting", "diffused lighting", "ambient lighting",
    "ambient occlusion", "global illumination", "subsurface scattering",
    "golden hour", "blue hour", "sunset lighting", "sunrise lighting",
    "moonlight", "moonlit", "candlelight", "candlelit", "firelight",
    "torchlight", "neon lighting", "neon lights", "neon glow", "bioluminescent",
    "bioluminescence", "iridescent", "glowing", "backlight", "lens flare",
    "chiaroscuro", "low key lighting", "high key lighting", "overcast",
    "natural lighting", "natural light", "window light", "spotlight",
    "underlighting", "moody lighting", "atmospheric lighting", "hazy light",
    "dappled light", "caustics", "light rays", "sunbeams", "dim lighting",
    "high contrast lighting", "flat lighting", "twilight", "dusk", "dawn",
    # Added from corpus-frequency mining:
    "dynamic lighting", "radiant light", "radiant lighting", "hdr",
    "global illumination", "soft shadows", "long shadows", "glow",
]

TONE_LEXICON = [
    "serene", "tranquil", "peaceful", "calm", "melancholic", "melancholy",
    "wistful", "nostalgic", "somber", "sombre", "bleak", "desolate",
    "ominous", "foreboding", "sinister", "menacing", "eerie", "unsettling",
    "creepy", "horrifying", "terrifying", "nightmarish", "grim", "gritty",
    "dark", "moody", "brooding", "dramatic", "epic", "heroic", "majestic",
    "triumphant", "awe-inspiring", "ethereal", "dreamlike", "dreamy",
    "otherworldly", "mystical", "mysterious", "enigmatic", "whimsical",
    "playful", "quirky", "cheerful", "joyful", "uplifting", "vibrant",
    "energetic", "chaotic", "frenetic", "romantic", "intimate", "tender",
    "lonely", "solitary", "hopeful", "optimistic", "surreal and dreamy",
    "cozy", "warm", "cold", "clinical", "sterile", "opulent", "decadent",
    # Added from corpus-frequency mining:
    "stunning", "gorgeous", "haunting", "serene and peaceful", "savage",
]

# Quality and detail boilerplate: carries no semantic field, so it goes to
# modifiers.
MODIFIER_LEXICON = [
    "highly detailed", "high detail", "extremely detailed", "ultra detailed",
    "intricate details", "intricate", "finely detailed", "detailed",
    "sharp focus", "in focus", "depth of field", "bokeh", "wide angle",
    "ultra wide angle", "fisheye", "close up", "closeup", "extreme close up",
    "portrait", "full body", "full shot", "wide shot", "medium shot",
    "establishing shot", "bird's eye view", "worm's eye view", "top down",
    "symmetrical", "asymmetrical", "centered", "rule of thirds",
    "8k", "4k", "16k", "2k", "1080p", "uhd", "hd", "high resolution",
    "ultra high definition", "high quality", "best quality", "masterpiece",
    "award winning", "award-winning", "professional", "trending on artstation",
    "artstation", "artstation hq", "deviantart", "behance", "pinterest",
    "cgsociety", "featured on artstation", "unreal", "photorealism",
    "very coherent", "coherent", "smooth", "clean", "crisp", "elegant",
    "hyper detailed", "insanely detailed", "octane", "vray", "redshift",
    "wallpaper", "poster", "key visual", "official art", "concept",
    "vivid colors", "vivid colours", "muted colors", "muted colours",
    "pastel colors", "pastel colours", "monochrome", "black and white",
    "sepia", "grainy", "film grain", "chromatic aberration", "vignette",
    # Added from corpus-frequency mining:
    "pixiv", "art station", "trending on art station", "ultra detail",
    "fine details", "fine detail", "perfect symmetry", "strong line",
    "strong lines", "pbr", "seamless", "cinematic", "beautiful", "raw",
    "50mm", "35mm", "85mm", "bokeh background", "volumetric",
    "subsurface scattering", "path traced", "high definition", "detailed face",
]

LEXICONS: dict[str, list[str]] = {
    "style": STYLE_LEXICON,
    "medium": MEDIUM_LEXICON,
    "lighting": LIGHTING_LEXICON,
    "tone": TONE_LEXICON,
    "modifiers": MODIFIER_LEXICON,
}

# Single-slot fields in priority order, used when a segment matches more than
# one lexicon. "oil painting" is both a style and a medium phrase; medium is
# the more specific claim.
_SINGLE_SLOT_PRIORITY = ["medium", "lighting", "style", "tone"]

# --- Segment cleaning -----------------------------------------------------

# Commas, semicolons, newlines, and the "|" / "::" weighting syntax that
# leaks in from Midjourney and AUTOMATIC1111 users. Sentence periods count
# too - DiffusionDB prompts routinely use them as tag separators ("... over
# the ocean. high detailed oil painting. dramatic.") - but not when the period
# follows a single-character token, which keeps decimal weights and lens specs
# ("f 1. 8") and initials in artist credits ("j. c. leyendecker") intact.
_SEGMENT_SPLIT_RE = re.compile(r"[,;\n]|\|{1,2}|::|(?<!\b\w)(?<!\d)\.(?=\s|$)")
# Weight and emphasis syntax: "(masterpiece:1.4)", "[blurry]", "{{detailed}}".
_WEIGHT_SUFFIX_RE = re.compile(r":\s*-?\d+(?:\.\d+)?\s*$")
_BRACKET_RE = re.compile(r"[()\[\]{}<>]")
_ARTIST_CREDIT_RE = re.compile(r"^(?:art\s+)?by\s+", re.IGNORECASE)
_INSPIRED_BY_RE = re.compile(r"^(?:in the style of|inspired by|style of)\s+", re.IGNORECASE)
_NEGATIVE_PREFIX_RE = re.compile(r"^(?:no|without|not|avoid|excluding)\s+", re.IGNORECASE)
_NEGATIVE_BLOCK_RE = re.compile(
    r"(?:--neg(?:ative)?\b|negative\s+prompt\s*:|\bneg\s*:)", re.IGNORECASE
)
_TRAILING_ARGS_RE = re.compile(r"--\w+(?:\s+[\w.:]+)?")
_NUMERIC_ONLY_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_WORD_RE = re.compile(r"\w")

# Prompts in this corpus are stored detokenized, which inserts spaces inside
# short alphanumeric tokens: "3 d render", "4 k", "5 0 mm", "art station".
# This was the largest single source of missed lexicon hits, so matching runs
# against a re-joined form. Only matching is normalized; stored labels keep
# the original segment text.
_DIGIT_LETTER_RE = re.compile(r"\b(\d) (?=[a-z]\b)")
_DIGIT_DIGIT_RE = re.compile(r"\b(\d) (?=\d\b)")
# No leading \b here: after "5 0" is joined to "50", the trailing digit is
# mid-word, so a boundary assertion would stop "50 mm" becoming "50mm".
_DIGIT_UNIT_RE = re.compile(r"(\d) (?=(?:mm|k|d)\b)")
_ART_STATION_RE = re.compile(r"\bart station\b", re.IGNORECASE)


def normalize_for_match(text: str) -> str:
    """Re-join detokenized spacing so lexicons can match."""
    out = _ART_STATION_RE.sub("artstation", text)
    for _ in range(3):  # "5 0 0 mm" needs more than one pass
        out = _DIGIT_DIGIT_RE.sub(r"\1", out)
    out = _DIGIT_UNIT_RE.sub(r"\1", out)
    out = _DIGIT_LETTER_RE.sub(r"\1", out)
    return out
_WS_RE = re.compile(r"\s+")


def clean_segment(segment: str) -> str:
    """Strip weight syntax, brackets and credit prefixes from a segment."""
    text = _BRACKET_RE.sub(" ", segment)
    text = _WEIGHT_SUFFIX_RE.sub("", text)
    text = _TRAILING_ARGS_RE.sub(" ", text)
    text = _ARTIST_CREDIT_RE.sub("", text.strip())
    text = _INSPIRED_BY_RE.sub("", text.strip())
    return _WS_RE.sub(" ", text).strip(" .-_\"'")


def split_prompt(prompt: str) -> tuple[list[str], list[str]]:
    """Split a prompt into (positive segments, negative segments).

    Text following an explicit negative marker is treated as negative.
    """
    positive_text, negative_text = prompt, ""
    match = _NEGATIVE_BLOCK_RE.search(prompt)
    if match:
        positive_text = prompt[: match.start()]
        negative_text = prompt[match.end() :]

    def segments_of(text: str) -> list[str]:
        return [c for c in (clean_segment(s) for s in _SEGMENT_SPLIT_RE.split(text)) if c]

    positives, negatives = [], []
    for segment in segments_of(positive_text):
        # An inline "no X" / "without X" segment is a negative constraint
        # even without an explicit marker.
        stripped = _NEGATIVE_PREFIX_RE.sub("", segment)
        if stripped != segment and stripped:
            negatives.append(stripped)
        else:
            positives.append(segment)
    negatives.extend(segments_of(negative_text))
    return positives, negatives


# --- Lexicon matching -----------------------------------------------------

def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE)


# Compiled once at import, longest phrase first so the most specific match
# in a segment wins.
_COMPILED_LEXICONS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    category: [
        (phrase, _phrase_pattern(phrase))
        for phrase in sorted(set(phrases), key=len, reverse=True)
    ]
    for category, phrases in LEXICONS.items()
}


def match_segment(segment: str) -> dict[str, str]:
    """Return {category: matched phrase} for every lexicon matching `segment`."""
    normalized = normalize_for_match(segment)
    matches: dict[str, str] = {}
    for category, phrases in _COMPILED_LEXICONS.items():
        for phrase, pattern in phrases:
            if pattern.search(normalized):
                matches[category] = phrase
                break
    return matches


def _looks_like_subject(segment: str, matches: dict[str, str]) -> bool:
    """Whether a segment carries content beyond the phrases that matched.

    "cyberpunk" is pure style, while "a cyberpunk street market at night" is a
    subject that happens to mention one. They are distinguished by how much of
    the segment the matched phrase covers.
    """
    if not matches:
        return True
    covered = max(len(phrase) for phrase in matches.values())
    return len(segment) - covered > 12


# --- Main entry point -----------------------------------------------------

def weak_label(prompt: str) -> StructuredFields:
    """Parse one prompt into StructuredFields.

    Two passes, because an explicitly tagged segment is a stronger signal than
    a phrase merely mentioned inside the subject. The first fills single-slot
    fields from short lexicon-dominated segments and collects longer
    content-bearing segments as subject and modifier candidates; the second
    backfills any still-empty slot from phrases found inside those longer
    segments.
    """
    positives, negatives = split_prompt(prompt)

    fields: dict[str, Optional[str]] = {k: None for k in _SINGLE_SLOT_PRIORITY}
    modifiers: list[str] = []
    subject_parts: list[str] = []
    # (segment, matches) for the long segments deferred to pass 2.
    deferred: list[tuple[str, dict[str, str]]] = []

    # Pass 1: explicitly tagged segments claim slots first.
    for segment in positives:
        if _NUMERIC_ONLY_RE.match(segment):
            continue  # leftover weight value from "::" / "|" syntax
        matches = match_segment(segment)

        if _looks_like_subject(segment, matches):
            # The first content-bearing segment is the subject; later ones
            # are descriptive detail. Matched phrases are held for pass 2.
            (subject_parts if not subject_parts else modifiers).append(segment)
            if matches:
                deferred.append((segment, matches))
            continue

        # Short, lexicon-dominated segment: assign to its best free slot.
        assigned = False
        for category in _SINGLE_SLOT_PRIORITY:
            if category in matches and fields[category] is None:
                fields[category] = segment
                assigned = True
                break
        if assigned:
            continue
        if matches:
            modifiers.append(segment)
        else:
            (subject_parts if not subject_parts else modifiers).append(segment)

    # Pass 2: backfill remaining slots from content-bearing segments.
    for _segment, matches in deferred:
        for category, phrase in matches.items():
            if category in fields and fields[category] is None:
                fields[category] = phrase

    # The schema requires a non-empty subject. A prompt of nothing but
    # style or quality tags falls back to the whole cleaned prompt. One with
    # no word-bearing content at all yields an empty subject and so fails
    # validation - deliberately loud, since a junk subject would silently
    # corrupt training targets. Ingest filtering makes this unreachable for
    # real corpus rows.
    subject = subject_parts[0] if subject_parts else clean_segment(prompt)
    if not _WORD_RE.search(subject):
        subject = ""

    return StructuredFields(
        subject=subject,
        style=fields["style"],
        medium=fields["medium"],
        lighting=fields["lighting"],
        tone=fields["tone"],
        modifiers=_dedupe(modifiers),
        negative_constraints=_dedupe(negatives),
    )


def _dedupe(values: Iterable[str]) -> list[str]:
    """Deduplicate case-insensitively, preserving order."""
    seen, out = set(), []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


# --- LLM seed-label override ---------------------------------------------

def load_llm_seed_labels(path: Path) -> dict[str, dict]:
    """Load optional LLM-labelled seed labels keyed by dataset id.

    Expects JSONL with one {"id": ..., "structured_fields": {...}} per line.
    A missing file yields an empty mapping.
    """
    if not path.exists():
        return {}
    seeds: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seeds[str(row["id"])] = row["structured_fields"]
    return seeds


def weak_label_row(prompt: str, dataset_id: str, seeds: dict[str, dict]) -> tuple[StructuredFields, str]:
    """Label one row, preferring a seed label when one exists.

    Returns (fields, provenance) where provenance is "llm_seed" or "rule".
    """
    if dataset_id in seeds:
        return StructuredFields(**seeds[dataset_id]), "llm_seed"
    return weak_label(prompt), "rule"


# --- Corpus frequency mining ---------------------------------------------

def mine_lexicon_candidates(prompts: Iterable[str], top_n: int = 40) -> list[tuple[str, int]]:
    """Most frequent short segments that no lexicon classifies.

    Run over the corpus to find vocabulary worth promoting into the lexicons
    above. Restricted to short segments, since long ones are subjects rather
    than vocabulary.
    """
    counter: Counter[str] = Counter()
    for prompt in prompts:
        positives, _ = split_prompt(prompt)
        for segment in positives:
            if len(segment.split()) > 4:
                continue
            if match_segment(segment):
                continue
            counter[segment.lower()] += 1
    return counter.most_common(top_n)
