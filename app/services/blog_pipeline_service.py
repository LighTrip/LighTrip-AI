from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llama_cpp import Llama

from app.services.category_service import CategoryPrediction, classify_text
from app.services.gemma_service import generate_blog_draft_from_bytes


@dataclass(frozen=True)
class BlogPipelineResult:
    generated_text: str
    category: CategoryPrediction


def generate_draft_and_classify(
    llm: Llama,
    image_bytes: bytes,
    filename: str,
    user_prompt: Optional[str] = None,
    unknown_threshold: Optional[float] = None,
) -> BlogPipelineResult:
    generated_text = generate_blog_draft_from_bytes(
        llm=llm,
        image_bytes=image_bytes,
        filename=filename,
        user_prompt=user_prompt,
    )
    category = classify_text(
        text=generated_text,
        unknown_threshold=unknown_threshold,
    )

    return BlogPipelineResult(
        generated_text=generated_text,
        category=category,
    )
