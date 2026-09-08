from fastapi import APIRouter, File, HTTPException, UploadFile

from app.embed import embed_images
from app.retrieval import retrieve_candidates
from app.schemas.candidate import Candidate

router = APIRouter()


@router.post("/internal/retrieve", response_model=list[Candidate])
async def retrieve(image: UploadFile = File(...), top_k: int = 10) -> list[dict]:
    image_bytes = await image.read()
    try:
        query_vec = embed_images([image_bytes])[0]
        return retrieve_candidates(query_vec, top_k=top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
