"""Mirrors docs/schema/candidate.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.structured_fields import StructuredFields


class Candidate(BaseModel):
    """A single retrieved reference prompt (FAISS neighbor, resolved to its Mongo document)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    similarity: float = Field(..., ge=0, le=1)
    structured_fields: Optional[StructuredFields] = None
    thumbnail_url: Optional[str] = None
