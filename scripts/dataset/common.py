from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_categories.json"


def load_categories(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"카테고리 설정이 비어 있습니다: {config_path}")

    return categories


def labels_by_slug(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    return {
        str(category["slug"]): str(category["label"])
        for category in load_categories(config_path)
    }


def slugs_by_label(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    return {
        str(category["label"]): str(category["slug"])
        for category in load_categories(config_path)
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
