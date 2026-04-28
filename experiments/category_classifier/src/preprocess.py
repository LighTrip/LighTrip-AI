from __future__ import annotations

import re
from pathlib import Path


DEFAULT_STOPWORDS = {
    "가",
    "같아",
    "같아요",
    "게",
    "곳",
    "그",
    "그리고",
    "기분",
    "나",
    "너무",
    "오늘",
    "이",
    "이런",
    "있는",
    "있어",
    "있어요",
    "정말",
    "좀",
    "참",
    "하",
    "하고",
    "하면서",
}

TOKEN_PATTERN = re.compile(r"[^0-9a-zA-Z가-힣\s]+")


def load_stopwords(path: Path | None = None) -> set[str]:
    if path is None:
        return set(DEFAULT_STOPWORDS)
    if not path.exists():
        raise FileNotFoundError(f"불용어 파일을 찾을 수 없습니다: {path}")

    words: set[str] = set(DEFAULT_STOPWORDS)
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            word = line.strip()
            if word and not word.startswith("#"):
                words.add(word)
    return words


def normalize_text(text: str) -> str:
    text = text.lower()
    text = TOKEN_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_ko(text: str, stopwords: set[str] | None = None) -> list[str]:
    stopword_set = DEFAULT_STOPWORDS if stopwords is None else stopwords
    normalized = normalize_text(text)
    return [
        token
        for token in normalized.split()
        if len(token) > 1 and token not in stopword_set
    ]
