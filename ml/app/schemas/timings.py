"""Mirrors docs/schema/timings.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Timings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_ms: float = Field(..., ge=0)
    retrieval_ms: float = Field(..., ge=0)
    captioning_ms: Optional[float] = Field(None, ge=0)
    decomposition_ms: Optional[float] = Field(None, ge=0)
    pez_ms: Optional[float] = Field(None, ge=0)
    regeneration_ms: Optional[float] = Field(None, ge=0)
    total_ms: float = Field(..., ge=0)
