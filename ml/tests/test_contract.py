"""Validates the shared example fixture against both the raw JSON Schema
(docs/schema/*.schema.json) and the Pydantic models that mirror it
(app/schemas/*.py). A failure here after a schema edit means the Pydantic
mirror is out of sync and needs to be updated to match.
"""

import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from app.schemas import Reconstruction, ReconstructRequest

# Locally (repo checkout) docs/ sits three levels up from this file. In Docker,
# only ml/ is mounted as the app - docs/ is mounted separately (docker-compose.yml)
# and SCHEMA_DIR is set to point at it directly.
_DEFAULT_SCHEMA_DIR = Path(__file__).resolve().parents[2] / "docs" / "schema"
SCHEMA_DIR = Path(os.environ.get("SCHEMA_DIR", _DEFAULT_SCHEMA_DIR))
EXAMPLES_DIR = SCHEMA_DIR / "examples"

SCHEMA_FILES = [
    "structured-fields.schema.json",
    "candidate.schema.json",
    "confidence.schema.json",
    "baselines.schema.json",
    "timings.schema.json",
    "graph.schema.json",
    "error.schema.json",
    "reconstruct-request.schema.json",
    "reconstruction.schema.json",
]


def load_schema(filename: str) -> dict:
    return json.loads((SCHEMA_DIR / filename).read_text())


def load_example(filename: str) -> dict:
    return json.loads((EXAMPLES_DIR / filename).read_text())


def build_registry() -> Registry:
    """All schema files registered by their $id, so cross-file $ref resolves."""
    resources = [Resource.from_contents(load_schema(f)) for f in SCHEMA_FILES]
    return Registry().with_resources([(r.id(), r) for r in resources])


def test_reconstruction_example_matches_json_schema():
    schema = load_schema("reconstruction.schema.json")
    example = load_example("reconstruction.example.json")
    Draft202012Validator(schema, registry=build_registry()).validate(example)


def test_reconstruction_example_matches_pydantic_model():
    example = load_example("reconstruction.example.json")
    # Should not raise - the Pydantic model must accept exactly what the JSON Schema accepts.
    Reconstruction.model_validate(example)


def test_reconstruct_request_example_matches_json_schema():
    schema = load_schema("reconstruct-request.schema.json")
    example = load_example("reconstruct-request.example.json")
    Draft202012Validator(schema).validate(example)


def test_reconstruct_request_example_matches_pydantic_model():
    example = load_example("reconstruct-request.example.json")
    ReconstructRequest.model_validate(example)


def test_reconstruct_request_rejects_missing_text_for_text_modality():
    schema = load_schema("reconstruct-request.schema.json")
    bad_instance = {"modality": "text"}  # missing required 'text'

    validator = Draft202012Validator(schema)
    assert not validator.is_valid(bad_instance)

    try:
        ReconstructRequest.model_validate(bad_instance)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Pydantic model should also reject modality='text' with no text"
