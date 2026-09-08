from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from app.caption import CAPTION_MODEL, caption_images

router = APIRouter()


class CaptionResponse(BaseModel):
    """Not part of the frozen API contract (docs/api-contract.md) - this is an
    internal stage endpoint, like /internal/retrieve. The caption reaches the
    public Reconstruction object only indirectly, as one of the decomposer's
    inputs in Task 5.2.
    """

    caption: str
    model: str


@router.post("/internal/caption", response_model=CaptionResponse)
async def caption(image: UploadFile = File(...)) -> dict:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="empty image upload")
    try:
        return {"caption": caption_images([image_bytes])[0], "model": CAPTION_MODEL}
    except UnidentifiedImageError:
        # Bad input, not a broken service - 503 would tell the caller to retry,
        # and the backend's orchestration (Task 3.1) treats 5xx as transient.
        raise HTTPException(status_code=422, detail="upload is not a decodable image")
    except Exception as e:
        # Genuinely unavailable: model download/load failure, OOM.
        raise HTTPException(status_code=503, detail=f"captioning unavailable: {e}")
