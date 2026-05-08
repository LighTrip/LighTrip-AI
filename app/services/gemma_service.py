from __future__ import annotations

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
    bytes_to_data_uri,
    generate_text_from_image_uri,
)
from app.prompts.gemma_prompt import (
    build_prompt as render_prompt,
    dedupe_sentences,
    load_prompt_template as read_prompt_template,
)


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

__all__ = [
    "ALLOWED_IMAGE_TYPES",
    "Gemma4VisionChatHandler",
    "build_prompt",
    "create_chat_handler",
    "create_llm",
    "dedupe_sentences",
    "generate_blog_draft_from_bytes",
    "get_llm",
    "image_bytes_to_data_uri",
    "is_model_loaded",
    "load_model",
    "load_prompt_template",
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


def load_model(verbose: bool = True) -> None:
    global _llm

    if _llm is not None:
        return

    chat_handler = create_chat_handler(MMPROJ_PATH, verbose=verbose)
    _llm = create_llm(MODEL_PATH, chat_handler, verbose=verbose)


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
