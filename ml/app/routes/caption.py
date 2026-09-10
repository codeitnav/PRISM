from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError
from pydantic import BaseModel

from app.caption import CAPTION_MODEL, caption_images

router = APIRouter()


class CaptionResponse(BaseModel):
    """Response body for the internal captioning endpoint.

    Not part of the public API contract; the caption reaches the public
    Reconstruction object only indirectly, as a decomposer input.
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
        # Malformed input, not a service failure: 5xx would signal the caller
        # to retry a request that can never succeed.
        raise HTTPException(status_code=422, detail="upload is not a decodable image")
    except Exception as e:
        # Genuinely unavailable: model load failure, out of memory.
        raise HTTPException(status_code=503, detail=f"captioning unavailable: {e}")
