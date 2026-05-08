from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from app.prompts.gemma_prompt import dedupe_sentences


def bytes_to_data_uri(image_bytes: bytes, filename: str = "upload.jpg") -> str:
    mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_file_to_data_uri(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

    return bytes_to_data_uri(path.read_bytes(), path.name)


def build_vision_messages(image_data_uri: str, prompt_text: str) -> list[dict[str, Any]]:
    return [
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
    ]


def extract_chat_message_text(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"].strip()


def generate_text_from_image_uri(
    llm: Any,
    image_data_uri: str,
    prompt_text: str,
    completion_kwargs: dict[str, Any],
) -> str:
    response = llm.create_chat_completion(
        messages=build_vision_messages(image_data_uri, prompt_text),
        **completion_kwargs,
    )
    return dedupe_sentences(extract_chat_message_text(response))
