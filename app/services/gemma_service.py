from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Final, Optional

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, suppress_stdout_stderr


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = BASE_DIR.parent.parent


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"필수 환경변수가 설정되지 않았습니다: {name}")
    return value.strip()


def required_int_env(name: str) -> int:
    return int(required_env(name))


def required_float_env(name: str) -> float:
    return float(required_env(name))


def required_bool_env(name: str) -> bool:
    return required_env(name).lower() in {"1", "true", "yes", "on"}


def required_list_env(name: str) -> list[str]:
    return [
        item.strip()
        for item in required_env(name).split(",")
        if item.strip()
    ]


def required_path_env(name: str) -> str:
    path = Path(required_env(name))
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


MODEL_PATH: Final[str] = required_path_env("GEMMA_MODEL_PATH")
MMPROJ_PATH: Final[str] = required_path_env("GEMMA_MMPROJ_PATH")
PROMPT_PATH: Final[str] = required_path_env("GEMMA_PROMPT_PATH")

N_CTX: Final[int] = required_int_env("GEMMA_N_CTX")
MAX_TOKENS: Final[int] = required_int_env("GEMMA_MAX_TOKENS")
TEMPERATURE: Final[float] = required_float_env("GEMMA_TEMPERATURE")
TOP_P: Final[float] = required_float_env("GEMMA_TOP_P")
TOP_K: Final[int] = required_int_env("GEMMA_TOP_K")
REPEAT_PENALTY: Final[float] = required_float_env("GEMMA_REPEAT_PENALTY")
STOP_TOKENS: Final[list[str]] = required_list_env("GEMMA_STOP_TOKENS")
N_GPU_LAYERS: Final[int] = required_int_env("GEMMA_N_GPU_LAYERS")
MAIN_GPU: Final[int] = required_int_env("GEMMA_MAIN_GPU")
OFFLOAD_KQV: Final[bool] = required_bool_env("GEMMA_OFFLOAD_KQV")
MMPROJ_USE_GPU: Final[bool] = required_bool_env("GEMMA_MMPROJ_USE_GPU")

ALLOWED_IMAGE_TYPES: Final[set[str]] = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

_llm: Optional[Llama] = None


def get_llm() -> Optional[Llama]:
    return _llm


def is_model_loaded() -> bool:
    return _llm is not None


def unload_model() -> None:
    global _llm
    _llm = None


def dedupe_sentences(text: str) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    seen: set[str] = set()
    kept: list[str] = []

    for part in parts:
        sentence = part.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            kept.append(sentence)

    return "\n".join(kept)


def image_bytes_to_data_uri(image_bytes: bytes, filename: str = "upload.jpg") -> str:
    mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def load_prompt_template(prompt_path: str = PROMPT_PATH) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")

    prompt_template = path.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise ValueError("프롬프트 파일이 비어 있습니다.")
    return prompt_template


def build_prompt(user_prompt: str | None = None) -> str:
    prompt_template = load_prompt_template()
    safe_user_prompt = user_prompt.strip() if user_prompt else ""

    if "{user_prompt}" in prompt_template:
        return prompt_template.replace("{user_prompt}", safe_user_prompt)
    if safe_user_prompt:
        return f"{prompt_template}\n\n{safe_user_prompt}"
    return prompt_template


class Gemma4VisionChatHandler(Llava15ChatHandler):
    DEFAULT_SYSTEM_MESSAGE = None
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "<start_of_turn>user\n{{ message.content }}<end_of_turn>\n"
        "{% endif %}"
        "{% if message.role == 'user' %}"
        "<start_of_turn>user\n"
        "{% if message.content is string %}{{ message.content }}{% endif %}"
        "{% if message.content is iterable %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}\n"
        "{% endif %}"
        "{% if content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% for content in message.content %}"
        "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<end_of_turn>\n"
        "{% endif %}"
        "{% if message.role == 'assistant' and message.content is not none %}"
        "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )

    def _init_mtmd_context(self, llama_model) -> None:
        if self.mtmd_ctx is not None:
            return

        with suppress_stdout_stderr(disable=self.verbose):
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = MMPROJ_USE_GPU
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED

            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(),
                llama_model.model,
                ctx_params,
            )

            if self.mtmd_ctx is None:
                raise ValueError(f"Failed to load mtmd context from: {self.clip_model_path}")

            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")


def create_chat_handler(mmproj_path: str, verbose: bool = True) -> Gemma4VisionChatHandler:
    mmproj_file = Path(mmproj_path)
    if not mmproj_file.exists():
        raise FileNotFoundError(f"mmproj 파일을 찾을 수 없습니다: {mmproj_path}")

    return Gemma4VisionChatHandler(
        clip_model_path=mmproj_path,
        verbose=verbose,
    )


def create_llm(
    model_path: str,
    chat_handler: Gemma4VisionChatHandler,
    verbose: bool = True,
) -> Llama:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    return Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,
        offload_kqv=OFFLOAD_KQV,
        main_gpu=MAIN_GPU,
        verbose=verbose,
    )


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

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPEAT_PENALTY,
        stop=STOP_TOKENS,
    )

    content = response["choices"][0]["message"]["content"].strip()
    return dedupe_sentences(content)
