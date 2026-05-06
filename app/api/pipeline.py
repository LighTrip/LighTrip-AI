from __future__ import annotations

import logging
import time
from typing import Dict, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.services.blog_pipeline_service import generate_draft_and_classify
from app.services.category_service import is_category_model_loaded
from app.services.gemma_service import (
    ALLOWED_IMAGE_TYPES,
    get_llm,
    is_model_loaded,
)


router = APIRouter(prefix="/pipeline", tags=["pipeline"])
logger = logging.getLogger(__name__)


class CategoryResponse(BaseModel):
    label: str
    raw_label: str
    confidence: Optional[float] = None
    score: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    model: Optional[str] = None


class PipelineDataResponse(BaseModel):
    generated_text: str
    category: str


class PipelineDebugResponse(BaseModel):
    category: CategoryResponse
    filename: Optional[str] = None
    prompt: str
    elapsed_seconds: float


class PipelineResponse(BaseModel):
    success: bool
    data: PipelineDataResponse
    debug: Optional[PipelineDebugResponse] = None


@router.post(
    "/generate-and-classify",
    response_model=PipelineResponse,
    response_model_exclude_none=True,
)
async def generate_and_classify(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    unknown_threshold: Optional[float] = Form(None),
    debug: bool = Query(False),
):
    llm = get_llm()
    if llm is None or not is_model_loaded():
        raise HTTPException(status_code=500, detail="Gemma 모델이 아직 로드되지 않았습니다.")
    if not is_category_model_loaded():
        raise HTTPException(status_code=500, detail="카테고리 분류 모델이 아직 로드되지 않았습니다.")

    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="지원하지 않는 이미지 형식입니다. jpg, png, webp만 가능합니다.",
        )

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")

        start_time = time.perf_counter()
        result = generate_draft_and_classify(
            llm=llm,
            image_bytes=image_bytes,
            filename=image.filename or "upload.jpg",
            user_prompt=prompt,
            unknown_threshold=unknown_threshold,
        )
        elapsed = time.perf_counter() - start_time

        response = {
            "success": True,
            "data": {
                "generated_text": result.generated_text,
                "category": result.category.label,
            },
        }

        if debug:
            response["debug"] = {
                "category": result.category.to_dict(),
                "filename": image.filename,
                "prompt": prompt,
                "elapsed_seconds": round(elapsed, 2),
            }

        return response

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Blog pipeline inference failed")
        raise HTTPException(status_code=500, detail="통합 파이프라인 추론 중 오류가 발생했습니다.") from exc
