"""Mirrors docs/schema/reconstruct-request.schema.json.

Describes the non-file fields of the multipart POST /api/reconstruct request.
The 'image' file part itself is handled separately by FastAPI's UploadFile,
not modeled here (binary data doesn't fit JSON Schema/Pydantic validation).
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReconstructRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Literal["image", "text"]
    text: Optional[str] = Field(None, min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    include_regeneration: bool = False

    @model_validator(mode="after")
    def text_required_when_modality_is_text(self) -> "ReconstructRequest":
        if self.modality == "text" and not self.text:
            raise ValueError("text is required when modality='text'")
        return self
