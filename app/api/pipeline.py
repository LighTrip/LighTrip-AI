from __future__ import annotations

import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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


class PipelineResponse(BaseModel):
    draft: str
    category: str


@router.post(
    "/generate",
    response_model=PipelineResponse,
    response_model_exclude_none=True,
)
async def generate(
    image: UploadFile = File(...),
    text: str = Form(""),
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
            user_prompt=text,
        )
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Pipeline inference completed: filename=%s category=%s source=%s fallback_reason=%s elapsed=%.2f",
            image.filename,
            result.category,
            result.category_source,
            result.fallback_reason,
            elapsed,
        )
        return {
            "draft": result.draft,
            "category": result.category,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Blog pipeline inference failed")
        raise HTTPException(status_code=500, detail="통합 파이프라인 추론 중 오류가 발생했습니다.") from exc
