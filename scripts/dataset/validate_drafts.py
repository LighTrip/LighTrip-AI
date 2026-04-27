from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import read_jsonl, write_jsonl


BANNED_PHRASES = ("이 사진", "사진에는", "사진은", "이미지", "보인다", "배경에는")
KOREAN_RE = re.compile(r"[가-힣]")
LETTER_RE = re.compile(r"[A-Za-z가-힣]")
DATASET_FIELDS = ("id", "image", "generated_text", "label")


def korean_ratio(text: str) -> float:
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    korean = KOREAN_RE.findall(text)
    return len(korean) / len(letters)


def validate_text(text: str) -> list[str]:
    stripped = text.strip()
    reasons: list[str] = []
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]

    if not stripped:
        reasons.append("empty_text")
    if len(stripped) < 20:
        reasons.append("too_short")
    if len(lines) < 2:
        reasons.append("too_few_lines")
    if len(lines) > 3:
        reasons.append("too_many_lines")
    if any(phrase in stripped for phrase in BANNED_PHRASES):
        reasons.append("contains_banned_phrase")
    if korean_ratio(stripped) < 0.75:
        reasons.append("low_korean_ratio")
    if len(set(lines)) != len(lines):
        reasons.append("duplicate_lines")

    return reasons


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("generated_text", ""))
    reasons = validate_text(text)

    row["quality_status"] = "rejected" if reasons else "accepted"
    row["quality_reasons"] = reasons
    return row


def strip_to_dataset_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def main() -> None:
    parser = argparse.ArgumentParser(description="생성된 초안 JSONL을 품질 검증합니다.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/generated_drafts.jsonl"))
    parser.add_argument("--accepted-output", type=Path, default=Path("data/processed/accepted_drafts.jsonl"))
    parser.add_argument("--rejected-output", type=Path, default=Path("data/interim/rejected_drafts.jsonl"))
    parser.add_argument("--include-validation-metadata", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    validated = [validate_row(row) for row in rows]
    accepted = [row for row in validated if row["quality_status"] == "accepted"]
    rejected = [row for row in validated if row["quality_status"] == "rejected"]

    accepted_output_rows = (
        accepted
        if args.include_validation_metadata
        else [strip_to_dataset_fields(row) for row in accepted]
    )

    write_jsonl(args.accepted_output, accepted_output_rows)
    write_jsonl(args.rejected_output, rejected)

    print(f"accepted: {len(accepted)} -> {args.accepted_output}")
    print(f"rejected: {len(rejected)} -> {args.rejected_output}")


if __name__ == "__main__":
    main()
