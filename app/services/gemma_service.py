"""Gemma GGUF 모델 로딩, 프롬프트 구성, 응답 파싱을 담당하는 서비스."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Optional

from llama_cpp import Llama

from app.config.gemma_config import (
    GEMMA_CONFIG,
    required_bool_env,
    required_env,
    required_float_env,
    required_int_env,
    required_list_env,
    required_path_env,
)
from app.config.gemma_runtime import (
    Gemma4VisionChatHandler,
    create_chat_handler,
    create_llm,
)
from app.prompts.gemma_formatter import (
    build_vision_messages,
    bytes_to_data_uri,
    extract_chat_message_text,
    generate_text_from_image_uri,
)
from app.prompts.gemma_prompt import (
    build_prompt as render_prompt,
    dedupe_sentences,
    load_prompt_template as read_prompt_template,
)
from app.services.category_policy import ALLOWED_CATEGORIES, normalize_category


MODEL_PATH: Final[str] = GEMMA_CONFIG.paths.model_path
MMPROJ_PATH: Final[str] = GEMMA_CONFIG.paths.mmproj_path
PROMPT_PATH: Final[str] = GEMMA_CONFIG.paths.prompt_path

N_CTX: Final[int] = GEMMA_CONFIG.model.n_ctx
MAX_TOKENS: Final[int] = GEMMA_CONFIG.generation.max_tokens
TEMPERATURE: Final[float] = GEMMA_CONFIG.generation.temperature
TOP_P: Final[float] = GEMMA_CONFIG.generation.top_p
TOP_K: Final[int] = GEMMA_CONFIG.generation.top_k
REPEAT_PENALTY: Final[float] = GEMMA_CONFIG.generation.repeat_penalty
STOP_TOKENS: Final[list[str]] = GEMMA_CONFIG.generation.stop_tokens
N_GPU_LAYERS: Final[int] = GEMMA_CONFIG.model.n_gpu_layers
MAIN_GPU: Final[int] = GEMMA_CONFIG.model.main_gpu
OFFLOAD_KQV: Final[bool] = GEMMA_CONFIG.model.offload_kqv
MMPROJ_USE_GPU: Final[bool] = GEMMA_CONFIG.model.mmproj_use_gpu

ALLOWED_IMAGE_TYPES: Final[set[str]] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}
JSON_OBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"\{.*\}", re.DOTALL)
DRAFT_MAX_CHARS: Final[int] = 80
DRAFT_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
DRAFT_SENTENCE_ENDINGS: Final[tuple[str, ...]] = (".", "!", "?", "。", "！", "？")


@dataclass(frozen=True)
class GemmaDirectResult:
    draft: str
    category: Optional[str]
    raw_category: object
    raw_output: str
    parse_status: str

__all__ = [
    "ALLOWED_IMAGE_TYPES",
    "build_direct_prompt",
    "Gemma4VisionChatHandler",
    "build_prompt",
    "create_chat_handler",
    "create_llm",
    "dedupe_sentences",
    "normalize_draft_text",
    "GemmaDirectResult",
    "generate_blog_draft_from_bytes",
    "generate_blog_draft_and_category_from_bytes",
    "get_llm",
    "image_bytes_to_data_uri",
    "is_model_loaded",
    "load_model",
    "load_prompt_template",
    "parse_direct_output",
    "required_bool_env",
    "required_env",
    "required_float_env",
    "required_int_env",
    "required_list_env",
    "required_path_env",
    "unload_model",
]

_llm: Optional[Llama] = None


def get_llm() -> Optional[Llama]:
    return _llm


def is_model_loaded() -> bool:
    return _llm is not None


def unload_model() -> None:
    global _llm
    _llm = None


def image_bytes_to_data_uri(image_bytes: bytes, filename: str = "upload.jpg") -> str:
    return bytes_to_data_uri(image_bytes, filename)


def load_prompt_template(prompt_path: str = PROMPT_PATH) -> str:
    return read_prompt_template(prompt_path)


def build_prompt(
    user_prompt: str | None = None,
    references: str | None = None,
) -> str:
    return render_prompt(PROMPT_PATH, user_prompt, references)


def build_direct_prompt(
    user_prompt: str | None = None,
    references: str | None = None,
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> str:
    draft_prompt = build_prompt(user_prompt, references)
    labels_block = "\n".join(f"- {label}" for label in allowed_categories)
    # direct 모드는 후처리가 쉽도록 draft/category만 담은 JSON을 강제한다.
    fallback_instruction = (
        '주요 카테고리에 맞지 않거나 애매하면 "기타"를 선택해라.'
        if "기타" in allowed_categories
        else "애매해도 선택 가능한 카테고리 중 가장 가까운 1개를 선택해라."
    )
    direct_rules = f"""
위 기준으로 draft를 쓰고, 최종 응답은 JSON 객체 1개만 출력해라.
선택 가능한 category:
{labels_block}

- key는 "draft", "category" 두 개만 사용해라.
- draft는 한국어 2문장, 한 줄, 전체 80자 이하로 작성해라.
- draft 문자열 안에 줄바꿈, \\n, 빈 줄을 넣지 마라.
- category는 위 목록 중 정확히 1개만 작성해라.
- {fallback_instruction}
- 설명, 마크다운 코드블록, 추가 문장은 출력하지 마라.

예시:
{{"draft":"한강 근처를 뛰고 나니 기분이 가벼웠다. 선선한 바람 덕분에 오래 기억날 러닝이다.","category":"운동"}}
""".strip()
    return f"{draft_prompt}\n\n{direct_rules}"


def load_model(verbose: bool = True) -> None:
    global _llm

    if _llm is not None:
        return

    # 비전 입력은 mmproj chat handler를 붙인 Llama 인스턴스로 처리한다.
    chat_handler = create_chat_handler(MMPROJ_PATH, verbose=verbose)
    _llm = create_llm(MODEL_PATH, chat_handler, verbose=verbose)


def _parse_json_object(text: str) -> dict[str, object] | None:
    match = JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None

    try:
        decoded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def _first_present(payload: dict[str, object], *keys: str) -> object:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def normalize_draft_text(text: str, max_chars: int = DRAFT_MAX_CHARS) -> str:
    draft = DRAFT_WHITESPACE_PATTERN.sub(" ", text).strip()
    if len(draft) <= max_chars:
        return draft

    trimmed = draft[:max_chars].rstrip()
    last_sentence_end = max(trimmed.rfind(ending) for ending in DRAFT_SENTENCE_ENDINGS)
    if last_sentence_end >= max_chars // 2:
        return trimmed[: last_sentence_end + 1].rstrip()

    return trimmed.rstrip(" ,，;；")


def parse_direct_output(
    raw_output: str,
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> GemmaDirectResult:
    decoded = _parse_json_object(raw_output)
    if decoded is None:
        # JSON 파싱 실패 시에도 draft 후보는 남겨 SVM fallback이 사용할 수 있게 한다.
        return GemmaDirectResult(
            draft=normalize_draft_text(raw_output),
            category=None,
            raw_category=None,
            raw_output=raw_output,
            parse_status="failed",
        )

    draft_value = _first_present(decoded, "draft", "초안", "generated_text")
    raw_draft = str(draft_value).strip() if isinstance(draft_value, str) else raw_output.strip()
    draft = normalize_draft_text(raw_draft)
    raw_category = _first_present(decoded, "category", "카테고리", "label")
    normalized_category = normalize_category(raw_category)
    category = normalized_category if normalized_category in set(allowed_categories) else None

    return GemmaDirectResult(
        draft=draft,
        category=category,
        raw_category=raw_category,
        raw_output=raw_output,
        parse_status="json",
    )


def generate_blog_draft_from_bytes(
    llm: Llama,
    image_bytes: bytes,
    filename: str,
    user_prompt: str | None = None,
    references: str | None = None,
) -> str:
    image_data_uri = image_bytes_to_data_uri(image_bytes, filename)
    prompt_text = build_prompt(user_prompt, references)

    return generate_text_from_image_uri(
        llm=llm,
        image_data_uri=image_data_uri,
        prompt_text=prompt_text,
        completion_kwargs=GEMMA_CONFIG.generation.as_chat_completion_kwargs(),
    )


def generate_blog_draft_and_category_from_bytes(
    llm: Llama,
    image_bytes: bytes,
    filename: str,
    user_prompt: str | None = None,
    references: str | None = None,
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> GemmaDirectResult:
    image_data_uri = image_bytes_to_data_uri(image_bytes, filename)
    prompt_text = build_direct_prompt(
        user_prompt=user_prompt,
        references=references,
        allowed_categories=allowed_categories,
    )
    # pipeline API의 기본 경로: 이미지에서 draft와 category를 한 번에 받는다.
    response = llm.create_chat_completion(
        messages=build_vision_messages(image_data_uri, prompt_text),
        **GEMMA_CONFIG.generation.as_chat_completion_kwargs(),
    )
    raw_output = extract_chat_message_text(response)
    return parse_direct_output(
        raw_output=raw_output,
        allowed_categories=allowed_categories,
    )
