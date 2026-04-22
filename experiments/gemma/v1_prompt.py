import base64
import mimetypes
import re
import time
from pathlib import Path
from typing import Final

import llama_cpp
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, suppress_stdout_stderr


BASE_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = BASE_DIR.parent.parent
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"

MODEL_PATH: Final[str] = str(MODEL_DIR / "gemma-4-E2B-it-Q4_K_M.gguf")
MMPROJ_PATH: Final[str] = str(MODEL_DIR / "mmproj-F16.gguf")
IMAGE_PATH: Final[str] = str(BASE_DIR / "images" / "running.jpg")

N_CTX: Final[int] = 1024
MAX_TOKENS: Final[int] = 64
TEMPERATURE: Final[float] = 1.0
TOP_P: Final[float] = 0.95
TOP_K: Final[int] = 64
REPEAT_PENALTY: Final[float] = 1.2
STOP_TOKENS: Final[list[str]] = ["<end_of_turn>"]


def dedupe_sentences(text: str) -> str:
    """중복 문장을 제거하고 줄바꿈 기준으로 정리한다."""
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
    seen: set[str] = set()
    kept: list[str] = []

    for part in parts:
        sentence = part.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            kept.append(sentence)

    return "\n".join(kept)


def image_to_data_uri(image_path: str) -> str:
    """로컬 이미지를 data URI로 변환한다."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def build_prompt() -> str:
    """블로그 초안 생성을 위한 사용자 프롬프트를 반환한다."""
    return (
        "너는 사용자가 사진과 함께 올릴 짧은 블로그 글 초안을 대신 작성하는 한국어 작가다.\n"
        "중요: 이미지 내용을 설명하는 해설문을 쓰지 마라.\n"
        "중요: 사용자가 자신의 순간을 기록하듯 자연스럽게 써라.\n"
        "중요: '이 사진은', '사진에는', '보인다', '배경에는' 같은 표현은 절대 사용하지 마라.\n"
        "반드시 한국어로만 작성하고, 정확히 2줄만 출력해라.\n"
        "각 줄은 실제 블로그나 SNS에 올릴 법한 자연스러운 초안이어야 한다.\n"
        "너무 분석적이거나 객관적인 묘사는 피하고, 일상 기록처럼 부드럽게 작성해라.\n"
        "제목, 번호, 기호, 따옴표 없이 결과만 출력해라.\n\n"
        "이 사진을 바탕으로 사용자가 직접 작성한 것 같은 블로그 초안 2줄을 작성해줘."
    )


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
        """멀티모달 컨텍스트를 초기화한다."""
        if self.mtmd_ctx is not None:
            return

        with suppress_stdout_stderr(disable=self.verbose):
            ctx_params = self._mtmd_cpp.mtmd_context_params_default()
            ctx_params.use_gpu = True
            ctx_params.print_timings = self.verbose
            ctx_params.n_threads = llama_model.n_threads
            ctx_params.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_DISABLED

            self.mtmd_ctx = self._mtmd_cpp.mtmd_init_from_file(
                self.clip_model_path.encode(),
                llama_model.model,
                ctx_params,
            )

            if self.mtmd_ctx is None:
                raise ValueError(
                    f"Failed to load mtmd context from: {self.clip_model_path}"
                )

            if not self._mtmd_cpp.mtmd_support_vision(self.mtmd_ctx):
                raise ValueError("Vision is not supported by this model")


def create_chat_handler(mmproj_path: str, verbose: bool = True) -> Gemma4VisionChatHandler:
    """비전 채팅 핸들러를 생성한다."""
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
    """Llama 모델 인스턴스를 생성한다."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    return Llama(
        model_path=model_path,
        chat_handler=chat_handler,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        offload_kqv=True,
        main_gpu=0,
        verbose=verbose,
    )


def generate_blog_draft(llm: Llama, image_path: str) -> str:
    """이미지를 기반으로 블로그 초안 2줄을 생성한다."""
    image_data_uri = image_to_data_uri(image_path)
    prompt_text = build_prompt()

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


def main() -> None:
    try:
        chat_handler = create_chat_handler(MMPROJ_PATH, verbose=True)
        llm = create_llm(MODEL_PATH, chat_handler, verbose=True)

        start_time = time.perf_counter()
        draft = generate_blog_draft(llm, IMAGE_PATH)
        elapsed = time.perf_counter() - start_time

        print(draft)
        print(f"\n총 추론 시간: {elapsed:.2f}초")

    except Exception as exc:
        print(f"오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    main()