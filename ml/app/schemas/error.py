"""Mirrors docs/schema/error.schema.json."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Error(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    details: Optional[dict] = None
