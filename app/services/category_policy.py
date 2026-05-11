from __future__ import annotations

from typing import Any, Final


ALLOWED_CATEGORIES: Final[tuple[str, ...]] = (
    "카페",
    "식당",
    "술집",
    "문화",
    "운동",
    "쇼핑",
    "공원",
    "기타",
)
ALLOWED_CATEGORY_SET: Final[set[str]] = set(ALLOWED_CATEGORIES)


def normalize_category(category: Any) -> str | None:
    if not isinstance(category, str):
        return None

    normalized = category.strip().strip('"').strip("'").strip()
    return normalized or None


def category_fallback_reason(category: Any) -> str | None:
    if category is None:
        return "missing_category"
    if not isinstance(category, str):
        return "invalid_category_type"
    if not category.strip():
        return "empty_category"

    normalized = normalize_category(category)
    if normalized not in ALLOWED_CATEGORY_SET:
        return "category_outside_allowed_set"
    return None
