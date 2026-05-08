from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Final

from llama_cpp import Llama


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.gemma_config import GEMMA_CONFIG, required_path_env
from app.config.gemma_runtime import (
    Gemma4VisionChatHandler,
    create_chat_handler,
    create_llm,
)
from app.prompts.gemma_formatter import (
    generate_text_from_image_uri,
    image_file_to_data_uri,
)
from app.prompts.gemma_prompt import (
    build_prompt as render_prompt,
)


IMAGE_PATH: Final[str] = required_path_env("GEMMA_SAMPLE_IMAGE_PATH")
PROMPT_PATH: Final[str] = GEMMA_CONFIG.paths.prompt_path


def image_to_data_uri(image_path: str) -> str:
    """로컬 이미지를 data URI로 변환한다."""
    return image_file_to_data_uri(image_path)


def build_prompt() -> str:
    """로컬 프롬프트 파일에서 사용자 프롬프트를 읽는다."""
    return render_prompt(PROMPT_PATH)


def generate_blog_draft(llm: Llama, image_path: str) -> str:
    """이미지를 기반으로 블로그 초안 2줄을 생성한다."""
    image_data_uri = image_to_data_uri(image_path)
    prompt_text = build_prompt()

    return generate_text_from_image_uri(
        llm=llm,
        image_data_uri=image_data_uri,
        prompt_text=prompt_text,
        completion_kwargs=GEMMA_CONFIG.generation.as_chat_completion_kwargs(),
    )


def main() -> None:
    try:
        chat_handler: Gemma4VisionChatHandler = create_chat_handler(
            GEMMA_CONFIG.paths.mmproj_path,
            verbose=True,
        )
        llm = create_llm(GEMMA_CONFIG.paths.model_path, chat_handler, verbose=True)

        start_time = time.perf_counter()
        draft = generate_blog_draft(llm, IMAGE_PATH)
        elapsed = time.perf_counter() - start_time

        print(draft)
        print(f"\n총 추론 시간: {elapsed:.2f}초")

    except Exception as exc:
        print(f"오류가 발생했습니다: {exc}")


if __name__ == "__main__":
    main()
