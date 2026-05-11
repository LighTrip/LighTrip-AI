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


def build_prompt(user_prompt: str | None = None) -> str:
    return render_prompt(PROMPT_PATH, user_prompt)


def build_direct_prompt(
    user_prompt: str | None = None,
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> str:
    draft_prompt = build_prompt(user_prompt)
    labels_block = "\n".join(f"- {label}" for label in allowed_categories)
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
- draft는 한국어 2줄이며 줄바꿈은 JSON 문자열 안에서 \\n으로 표현해라.
- category는 위 목록 중 정확히 1개만 작성해라.
- {fallback_instruction}
- 설명, 마크다운 코드블록, 추가 문장은 출력하지 마라.

예시:
{{"draft":"한강 근처를 뛰고 나니 땀이 나도 이상하게 기분이 가벼웠다.\\n바람도 선선해서 오늘 러닝은 오래 기억에 남을 것 같다.","category":"운동"}}
""".strip()
    return f"{draft_prompt}\n\n{direct_rules}"


def load_model(verbose: bool = True) -> None:
    global _llm

    if _llm is not None:
        return

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


def parse_direct_output(
    raw_output: str,
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> GemmaDirectResult:
    decoded = _parse_json_object(raw_output)
    if decoded is None:
        return GemmaDirectResult(
            draft=raw_output.strip(),
            category=None,
            raw_category=None,
            raw_output=raw_output,
            parse_status="failed",
        )

    draft_value = _first_present(decoded, "draft", "초안", "generated_text")
    draft = str(draft_value).strip() if isinstance(draft_value, str) else raw_output.strip()
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
) -> str:
    image_data_uri = image_bytes_to_data_uri(image_bytes, filename)
    prompt_text = build_prompt(user_prompt)

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
    allowed_categories: tuple[str, ...] = ALLOWED_CATEGORIES,
) -> GemmaDirectResult:
    image_data_uri = image_bytes_to_data_uri(image_bytes, filename)
    prompt_text = build_direct_prompt(
        user_prompt=user_prompt,
        allowed_categories=allowed_categories,
    )
    response = llm.create_chat_completion(
        messages=build_vision_messages(image_data_uri, prompt_text),
        **GEMMA_CONFIG.generation.as_chat_completion_kwargs(),
    )
    raw_output = extract_chat_message_text(response)
    return parse_direct_output(
        raw_output=raw_output,
        allowed_categories=allowed_categories,
    )
