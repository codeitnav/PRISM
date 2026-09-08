"""Task 1.3 - weak structured labels.

Parses a raw DiffusionDB prompt into the StructuredFields schema
(subject/style/medium/lighting/modifiers/tone/negative_constraints) using a
rule + lexicon pass. These are *weak* labels: cheap, noisy, and generated
without human annotation. They exist to serve as supervision targets for the
decomposition model (Task 5.2/5.3) and as the ground truth for the eval
harness's component-wise precision/recall (Task 4.3's missing piece).

How the rule pass works:

1. Split the prompt on the separators DiffusionDB prompts actually use -
   commas, sentence periods, and the "|" / "::" weighting syntax that leaks in
   from Midjourney and AUTOMATIC1111 users.
2. Strip prompt-engineering noise from each segment: attention weights
   ("(detailed:1.3)"), artist-credit boilerplate, trailing weight numbers.
3. Classify each segment against curated lexicons. A segment matches a
   category when a lexicon phrase appears in it as a whole word/phrase; the
   longest matching phrase wins, so "soft volumetric lighting" beats "soft".
4. Whatever is left unclassified becomes the subject - the first unmatched
   segment is the primary subject, later ones fold into modifiers. Segments
   that match nothing but look like quality boilerplate ("8k", "trending on
   artstation") are routed to modifiers by a separate boilerplate lexicon.
5. Negative constraints come from explicit negative syntax: a leading "no "/
   "without ", or AUTOMATIC1111's "--neg"/"negative prompt:" markers.

The lexicons are seeded by hand and then *extended from corpus frequency*
via mine_lexicon_candidates(), which surfaces the highest-frequency
unclassified segments in the actual ingested corpus so the vocabulary is
grounded in this dataset rather than in generic art vocabulary.

Scope note - LLM seed labels: the roadmap pairs this rule pass with an
LLM-labeled seed set of 5-10K prompts for the ambiguous cases. Two things
make that inapplicable as written here: the ingested corpus is 700 prompts
total (see scripts/ingest_diffusiondb.py's scope note), so a 5-10K seed set
is larger than the dataset, and this dev environment has no LLM API
credentials configured. The hook is built and wired instead of faked:
load_llm_seed_labels() reads data/diffusiondb/llm_seed_labels.jsonl if it
exists and those labels override the rule output per prompt, so a seed set
can be dropped in later without touching the pipeline. The audit in
docs/label-quality.md reports rule-pass-only precision.
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
# Ordered longest-phrase-first at match time, not here, so these stay
# readable/appendable. Entries are lowercase; matching is case-insensitive.

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
    # Promoted from corpus-frequency mining on the ingested 700 prompts:
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
    # Promoted from corpus-frequency mining on the ingested 700 prompts:
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
    # Promoted from corpus-frequency mining on the ingested 700 prompts:
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
    # Promoted from corpus-frequency mining on the ingested 700 prompts:
    "stunning", "gorgeous", "haunting", "serene and peaceful", "savage",
]

# Quality/detail boilerplate: matches nothing semantic, belongs in modifiers.
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
    # Promoted from corpus-frequency mining on the ingested 700 prompts:
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

# Single-slot fields, in the priority order used when one segment matches
# more than one lexicon (e.g. "oil painting" is both a style and a medium
# phrase - medium is the more specific claim, so it wins).
_SINGLE_SLOT_PRIORITY = ["medium", "lighting", "style", "tone"]

# --- Segment cleaning -----------------------------------------------------

# Commas, semicolons, newlines, and the "|" / "::" weighting syntax that
# leaks in from Midjourney and AUTOMATIC1111 users. Sentence periods count
# too - DiffusionDB prompts routinely use them as tag separators ("... over
# the ocean. high detailed oil painting. dramatic.") - but not when the period
# follows a single-character token, which keeps decimal weights and lens specs
# ("f 1. 8") and initials in artist credits ("j. c. leyendecker") intact.
_SEGMENT_SPLIT_RE = re.compile(r"[,;\n]|\|{1,2}|::|(?<!\b\w)(?<!\d)\.(?=\s|$)")
# "(masterpiece:1.4)", "[blurry]", "{{detailed}}" - weight/emphasis syntax.
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

# DiffusionDB prompts in the 2m_random_* configs are stored detokenized, which
# inserts spaces inside short alphanumeric tokens: "3 d render", "4 k", "8 k",
# "5 0 mm", "art station". Corpus-frequency mining surfaced these as the single
# largest source of missed lexicon hits (~230 across 700 prompts), so matching
# runs against a re-joined form. Only matching is normalized - the value stored
# in the labels stays the original segment text.
_DIGIT_LETTER_RE = re.compile(r"\b(\d) (?=[a-z]\b)")
_DIGIT_DIGIT_RE = re.compile(r"\b(\d) (?=\d\b)")
# No leading \b here: after "5 0" is joined to "50", the trailing digit is
# mid-word, so a boundary assertion would stop "50 mm" becoming "50mm".
_DIGIT_UNIT_RE = re.compile(r"(\d) (?=(?:mm|k|d)\b)")
_ART_STATION_RE = re.compile(r"\bart station\b", re.IGNORECASE)


def normalize_for_match(text: str) -> str:
    """Re-join DiffusionDB's detokenized spacing so lexicons can match."""
    out = _ART_STATION_RE.sub("artstation", text)
    for _ in range(3):  # "5 0 0 mm" needs more than one pass
        out = _DIGIT_DIGIT_RE.sub(r"\1", out)
    out = _DIGIT_UNIT_RE.sub(r"\1", out)
    out = _DIGIT_LETTER_RE.sub(r"\1", out)
    return out
_WS_RE = re.compile(r"\s+")


def clean_segment(segment: str) -> str:
    """Strip weight syntax, brackets and credit boilerplate from a segment."""
    text = _BRACKET_RE.sub(" ", segment)
    text = _WEIGHT_SUFFIX_RE.sub("", text)
    text = _TRAILING_ARGS_RE.sub(" ", text)
    text = _ARTIST_CREDIT_RE.sub("", text.strip())
    text = _INSPIRED_BY_RE.sub("", text.strip())
    return _WS_RE.sub(" ", text).strip(" .-_\"'")


def split_prompt(prompt: str) -> tuple[list[str], list[str]]:
    """Split a raw prompt into (positive segments, negative segments).

    Text after an explicit negative marker ("--neg", "negative prompt:") is
    treated as negative; everything before it is positive.
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
        # An inline "no X" / "without X" segment is a negative constraint even
        # without an explicit negative-prompt marker.
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


# Compiled once at import; each category's phrases are ordered longest-first
# so the most specific phrase in a segment is the one that matches.
_COMPILED_LEXICONS: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    category: [
        (phrase, _phrase_pattern(phrase))
        for phrase in sorted(set(phrases), key=len, reverse=True)
    ]
    for category, phrases in LEXICONS.items()
}


def match_segment(segment: str) -> dict[str, str]:
    """Return {category: matched phrase} for every lexicon hitting `segment`."""
    normalized = normalize_for_match(segment)
    matches: dict[str, str] = {}
    for category, phrases in _COMPILED_LEXICONS.items():
        for phrase, pattern in phrases:
            if pattern.search(normalized):
                matches[category] = phrase
                break
    return matches


def _looks_like_subject(segment: str, matches: dict[str, str]) -> bool:
    """True when a segment carries content beyond the phrases that matched.

    "cyberpunk" is pure style; "a cyberpunk street market at night" matched
    style too, but it is clearly the subject. Distinguish them by how much
    of the segment the matched phrase actually covers.
    """
    if not matches:
        return True
    covered = max(len(phrase) for phrase in matches.values())
    return len(segment) - covered > 12


# --- Main entry point -----------------------------------------------------

def weak_label(prompt: str) -> StructuredFields:
    """Parse one raw prompt into weak StructuredFields via the rule pass.

    Two passes, because an explicitly-tagged segment is a stronger signal
    than a phrase merely mentioned inside the subject:

      Pass 1 fills the single-slot fields from short, lexicon-dominated
      segments ("oil painting", "cinematic lighting") and collects the
      long content-bearing segments as subject/modifier candidates.
      Pass 2 backfills whichever single slots are *still* empty from
      phrases matched inside those long segments, so "hyperrealistic
      photograph of an astronaut" can still yield style=hyperrealistic.
    """
    positives, negatives = split_prompt(prompt)

    fields: dict[str, Optional[str]] = {k: None for k in _SINGLE_SLOT_PRIORITY}
    modifiers: list[str] = []
    subject_parts: list[str] = []
    # (segment, matches) for the long segments deferred to pass 2.
    deferred: list[tuple[str, dict[str, str]]] = []

    # --- Pass 1: explicit tags win ---
    for segment in positives:
        if _NUMERIC_ONLY_RE.match(segment):
            continue  # leftover weight value from "::"/"|" syntax
        matches = match_segment(segment)

        if _looks_like_subject(segment, matches):
            # Long, content-bearing segment: first one is the subject, the
            # rest are extra descriptive detail -> modifiers. Its matched
            # phrases are held back for pass 2.
            (subject_parts if not subject_parts else modifiers).append(segment)
            if matches:
                deferred.append((segment, matches))
            continue

        # Short, lexicon-dominated segment: file it under its best free slot.
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

    # --- Pass 2: backfill still-empty slots from the long segments ---
    for _segment, matches in deferred:
        for category, phrase in matches.items():
            if category in fields and fields[category] is None:
                fields[category] = phrase

    # Every prompt must yield a non-empty subject (schema requires it). If the
    # prompt was nothing but style/quality tags, fall back to the whole
    # cleaned prompt rather than dropping the record. A prompt with no
    # word-bearing content at all (empty, punctuation-only, or nothing but a
    # negative block) gets an empty subject and so fails schema validation -
    # deliberately loud, since silently emitting a junk subject would poison
    # the SFT targets in Task 5.2. The ingest filter's >=3-token rule means
    # this should never fire on real corpus rows.
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
    """Order-preserving, case-insensitive dedupe."""
    seen, out = set(), []
    for value in values:
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


# --- LLM seed-label override ---------------------------------------------

def load_llm_seed_labels(path: Path) -> dict[str, dict]:
    """Load optional LLM-labeled seed labels keyed by dataset id.

    Format: JSONL, one {"id": "000002", "structured_fields": {...}} per line.
    Absent file -> empty mapping, which is the current state (see the module
    docstring's scope note).
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
    """Label one row, preferring an LLM seed label when one exists.

    Returns (fields, provenance) where provenance is "llm_seed" or "rule".
    """
    if dataset_id in seeds:
        return StructuredFields(**seeds[dataset_id]), "llm_seed"
    return weak_label(prompt), "rule"


# --- Corpus frequency mining ---------------------------------------------

def mine_lexicon_candidates(prompts: Iterable[str], top_n: int = 40) -> list[tuple[str, int]]:
    """Most frequent short segments that no lexicon currently classifies.

    This is the "vocabularies mined from corpus frequency" half of Task 1.3:
    run it over the ingested corpus, read the output, and promote genuine
    style/medium/lighting terms into the lexicons above. Restricted to short
    segments because long ones are subjects, not vocabulary.
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
