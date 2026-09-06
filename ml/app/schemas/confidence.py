"""Mirrors docs/schema/confidence.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceComponents(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cos_sim: Optional[float] = Field(None, ge=0, le=1)
    retrieval_margin: float = Field(..., ge=0, le=1)
    component_agreement: float = Field(..., ge=0, le=1)


class ConfidenceWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha: float
    beta: float
    gamma: float


class Confidence(BaseModel):
    """confidence = alpha*cos_sim + beta*retrieval_margin + gamma*component_agreement, temperature-scaled."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(..., ge=0, le=1)
    components: ConfidenceComponents
    weights: Optional[ConfidenceWeights] = None
