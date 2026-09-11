from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.decompose import decompose_reliably
from app.schemas.structured_fields import StructuredFields
from app.sft import render_input

router = APIRouter()


class DecomposeRequest(BaseModel):
    """Evidence for one image, assembled by the caller from captioning and
    retrieval - mirrors the input app.sft renders training examples from.
    """

    caption: str
    retrieved_prompts: list[str] = []
    pez_prompt: str | None = None


@router.post("/internal/decompose", response_model=StructuredFields)
async def decompose(request: DecomposeRequest) -> StructuredFields:
    if not request.caption.strip():
        raise HTTPException(status_code=422, detail="caption must not be empty")
    input_text = render_input(request.caption, request.retrieved_prompts, request.pez_prompt)
    try:
        return decompose_reliably([input_text])[0]
    except Exception as e:
        # Genuinely unavailable: model load failure, out of memory. A failed
        # parse never reaches here - decompose_reliably guarantees a result.
        raise HTTPException(status_code=503, detail=f"decomposition unavailable: {e}")
