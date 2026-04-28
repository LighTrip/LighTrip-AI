from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSONL 파싱 실패: {exc}") from exc
    return rows


def load_text_label_dataset(
    path: Path,
    text_field: str = "generated_text",
    label_field: str = "label",
) -> tuple[list[str], list[str]]:
    rows = read_jsonl(path)
    texts: list[str] = []
    labels: list[str] = []

    for index, row in enumerate(rows, start=1):
        text = row.get(text_field)
        label = row.get(label_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{index} '{text_field}' 값이 비어 있습니다.")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{path}:{index} '{label_field}' 값이 비어 있습니다.")
        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError(f"학습 가능한 데이터가 없습니다: {path}")

    return texts, labels


def load_texts_from_jsonl(path: Path, text_field: str = "generated_text") -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for index, row in enumerate(rows, start=1):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{index} '{text_field}' 값이 비어 있습니다.")
    return rows
