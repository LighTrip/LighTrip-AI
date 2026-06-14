from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.embedding_service import (
    embed_texts,
    get_embedding_llm,
    is_embedding_model_loaded,
)


router = APIRouter(tags=["embedding"])
logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1)


class EmbedResponse(BaseModel):
    dim: int
    embeddings: list[list[float]]


@router.post("/get-embedding", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    llm = get_embedding_llm()
    if llm is None or not is_embedding_model_loaded():
        raise HTTPException(status_code=500, detail="임베딩 모델이 아직 로드되지 않았습니다.")

    try:
        vectors = embed_texts(llm, req.texts)
    except Exception as exc:
        logger.exception("Embedding inference failed")
        raise HTTPException(status_code=500, detail="임베딩 추론 중 오류가 발생했습니다.") from exc

    return {
        "dim": len(vectors[0]) if vectors else 0,
        "embeddings": vectors,
    }
