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
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"

MODEL_FILENAME: Final[str] = os.getenv("GEMMA_MODEL_FILENAME", "gemma-4-E2B-it-Q4_K_M.gguf")
MMPROJ_FILENAME: Final[str] = os.getenv("GEMMA_MMPROJ_FILENAME", "mmproj-F16.gguf")
MODEL_PATH: Final[str] = str(MODEL_DIR / MODEL_FILENAME)
MMPROJ_PATH: Final[str] = str(MODEL_DIR / MMPROJ_FILENAME)

N_CTX: Final[int] = int(os.getenv("GEMMA_N_CTX", "1024"))
MAX_TOKENS: Final[int] = int(os.getenv("GEMMA_MAX_TOKENS", "128"))
TEMPERATURE: Final[float] = 1.0
TOP_P: Final[float] = 0.95
TOP_K: Final[int] = 64
REPEAT_PENALTY: Final[float] = 1.2
STOP_TOKENS: Final[list[str]] = ["<end_of_turn>"]
N_GPU_LAYERS: Final[int] = int(os.getenv("GEMMA_N_GPU_LAYERS", "-1"))
MAIN_GPU: Final[int] = int(os.getenv("GEMMA_MAIN_GPU", "0"))
OFFLOAD_KQV: Final[bool] = os.getenv("GEMMA_OFFLOAD_KQV", "1") != "0"
MMPROJ_USE_GPU: Final[bool] = os.getenv("GEMMA_MMPROJ_USE_GPU", "1") != "0"

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


def build_prompt(user_prompt: str | None = None) -> str:
    base_prompt = (
        "너는 사용자가 사진과 함께 올릴 짧은 블로그 글 초안을 대신 작성하는 한국어 작가다.\n"
        "중요: 이미지 내용을 설명하는 해설문을 쓰지 마라.\n"
        "중요: 사용자가 자신의 순간을 기록하듯 자연스럽게 써라.\n"
        "중요: '이 사진은', '사진에는', '보인다', '배경에는' 같은 표현은 절대 사용하지 마라.\n"
        "반드시 한국어로만 작성하고, 정확히 2줄만 출력해라.\n"
        "각 줄은 실제 블로그나 SNS에 올릴 법한 자연스러운 초안이어야 한다.\n"
        "너무 분석적이거나 객관적인 묘사는 피하고, 일상 기록처럼 부드럽게 작성해라.\n"
        "제목, 번호, 기호, 따옴표 없이 결과만 출력해라.\n"
    )

    if user_prompt and user_prompt.strip():
        base_prompt += f"\n사용자 요청:\n{user_prompt.strip()}\n"

    base_prompt += "\n이 사진을 바탕으로 사용자가 직접 작성한 것 같은 블로그 초안 2줄을 작성해줘."
    return base_prompt


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
