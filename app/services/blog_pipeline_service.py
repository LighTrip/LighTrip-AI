from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama

from app.services.category_service import CategoryPrediction, classify_text
from app.services.category_policy import ALLOWED_CATEGORIES, category_fallback_reason
from app.services.gemma_service import generate_blog_draft_and_category_from_bytes


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlogPipelineResult:
    draft: str
    category: str
    category_source: str
    fallback_reason: Optional[str] = None
    gemma_category: Optional[str] = None
    svm_prediction: Optional[CategoryPrediction] = None


def generate_draft_and_classify(
    llm: Llama,
    image_bytes: bytes,
    filename: str,
    user_prompt: Optional[str] = None,
    unknown_threshold: Optional[float] = None,
) -> BlogPipelineResult:
    direct_result = generate_blog_draft_and_category_from_bytes(
        llm=llm,
        image_bytes=image_bytes,
        filename=filename,
        user_prompt=user_prompt,
        allowed_categories=ALLOWED_CATEGORIES,
    )

    fallback_reason = category_fallback_reason(direct_result.raw_category)
    if fallback_reason is None and direct_result.category is not None:
        return BlogPipelineResult(
            draft=direct_result.draft,
            category=direct_result.category,
            category_source="gemma_direct",
            gemma_category=direct_result.category,
        )

    draft = direct_result.draft.strip()
    if not draft:
        raise ValueError("Gemma가 생성한 초안이 비어 있어 SVM fallback을 수행할 수 없습니다.")

    svm_prediction = classify_text(
        text=draft,
        unknown_threshold=unknown_threshold,
    )
    logger.warning(
        "Gemma category fallback used: reason=%s raw_category=%r svm_category=%s",
        fallback_reason,
        direct_result.raw_category,
        svm_prediction.label,
    )

    return BlogPipelineResult(
        draft=draft,
        category=svm_prediction.label,
        category_source="svm_fallback",
        fallback_reason=fallback_reason,
        gemma_category=direct_result.category,
        svm_prediction=svm_prediction,
    )
