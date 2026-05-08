from __future__ import annotations

import re
from pathlib import Path


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def dedupe_sentences(text: str) -> str:
    parts = SENTENCE_SPLIT_PATTERN.split(text.strip())
    seen: set[str] = set()
    kept: list[str] = []

    for part in parts:
        sentence = part.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            kept.append(sentence)

    return "\n".join(kept)


def load_prompt_template(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_path}")

    prompt_template = path.read_text(encoding="utf-8").strip()
    if not prompt_template:
        raise ValueError("프롬프트 파일이 비어 있습니다.")
    return prompt_template


def build_prompt(prompt_path: str, user_prompt: str | None = None) -> str:
    prompt_template = load_prompt_template(prompt_path)
    safe_user_prompt = user_prompt.strip() if user_prompt else ""

    if "{user_prompt}" in prompt_template:
        return prompt_template.replace("{user_prompt}", safe_user_prompt)
    if safe_user_prompt:
        return f"{prompt_template}\n\n{safe_user_prompt}"
    return prompt_template
