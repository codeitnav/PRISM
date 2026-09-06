"""Mirrors docs/schema/reconstruction.schema.json."""

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.baselines import Baselines
from app.schemas.candidate import Candidate
from app.schemas.confidence import Confidence
from app.schemas.graph import Graph
from app.schemas.structured_fields import StructuredFields
from app.schemas.timings import Timings


class ReconstructionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: Optional[str] = None
    content_type: Optional[str] = None
    text: Optional[str] = None
    storage_path: str = Field(..., min_length=1)


class Reconstruction(BaseModel):
    """The full result of a reconstruction request - returned by POST /api/reconstruct
    and GET /api/reconstruct/:id, and persisted as-is in MongoDB's `reconstructions`
    collection."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    modality: Literal["image", "text"]
    status: Literal["pending", "processing", "completed", "failed", "degraded"]
    input: ReconstructionInput
    structured_fields: Optional[StructuredFields] = None
    candidates: list[Candidate] = Field(default_factory=list)
    confidence: Optional[Confidence] = None
    baselines: Baselines
    graph: Optional[Graph] = None
    timings: Optional[Timings] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
