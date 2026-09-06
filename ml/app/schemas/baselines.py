"""Mirrors docs/schema/baselines.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PezBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    clip_score: Optional[float] = Field(None, ge=-1, le=1)
    latency_ms: Optional[float] = Field(None, ge=0)


class ClipTagBaseline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(..., min_length=1)
    tags: list[str] = Field(default_factory=list)
    latency_ms: Optional[float] = Field(None, ge=0)


class Baselines(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pez: Optional[PezBaseline] = None
    clip_tag: Optional[ClipTagBaseline] = None
