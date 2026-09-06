"""Mirrors docs/schema/structured-fields.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class StructuredFields(BaseModel):
    """Decomposed prompt structure - the system's core contribution."""

    model_config = ConfigDict(extra="forbid")

    subject: str = Field(..., min_length=1)
    style: Optional[str] = None
    medium: Optional[str] = None
    lighting: Optional[str] = None
    modifiers: list[str] = Field(default_factory=list)
    tone: Optional[str] = None
    negative_constraints: list[str] = Field(default_factory=list)
