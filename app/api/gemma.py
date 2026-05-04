import logging
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.gemma_service import (
    ALLOWED_IMAGE_TYPES,
    generate_blog_draft_from_bytes,
    get_llm,
)

router = APIRouter(prefix="/gemma", tags=["gemma"])
logger = logging.getLogger(__name__)


@router.post("/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form(""),
):
    llm = get_llm()
    if llm is None:
        raise HTTPException(status_code=500, detail="모델이 아직 로드되지 않았습니다.")

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

        result = generate_blog_draft_from_bytes(
            llm=llm,
            image_bytes=image_bytes,
            filename=image.filename or "upload.jpg",
            user_prompt=prompt,
        )

        elapsed = time.perf_counter() - start_time

        return {
            "result": result,
            "prompt": prompt,
            "filename": image.filename,
            "elapsed_seconds": round(elapsed, 2),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemma inference failed")
        raise HTTPException(status_code=500, detail="추론 중 오류가 발생했습니다.") from exc
