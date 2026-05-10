from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import read_jsonl, write_jsonl


DATASET_FIELDS = ("id", "image", "generated_text", "label")
DEFAULT_INPUT = Path("data_places365_2/interim/places365_service_prompt_drafts.jsonl")
DEFAULT_OUTPUT_DIR = Path("data_places365_2/semantic_filter")


STRONG_CUES: dict[str, tuple[str, ...]] = {
    "카페": (
        "커피",
        "라떼",
        "아메리카노",
        "에스프레소",
        "카페",
        "디저트",
        "케이크",
        "티라미수",
    ),
    "식당": (
        "식사",
        "한 끼",
        "밥",
        "메뉴",
        "주문",
        "요리",
        "레스토랑",
        "식당",
        "피자",
        "햄버거",
        "치킨",
    ),
    "술집": (
        "맥주",
        "소주",
        "와인",
        "칵테일",
        "위스키",
        "술집",
        "펍",
        "안주",
        "건배",
        "잔을 기울",
        "취기",
        "취해서",
    ),
    "문화": (
        "전시",
        "미술관",
        "박물관",
        "갤러리",
        "공연",
        "영화",
        "극장",
        "무대",
        "관람",
        "조형물",
        "유물",
    ),
    "쇼핑": (
        "쇼핑",
        "진열대",
        "상품",
        "매장",
        "계산대",
        "구매",
        "백화점",
        "마트",
        "슈퍼마켓",
        "가게",
        "아이템",
        "소품",
        "신발",
        "보석",
        "장난감",
    ),
    "운동": (
        "운동",
        "경기",
        "스포츠",
        "축구",
        "야구",
        "농구",
        "골프",
        "스키",
        "복싱",
        "볼링",
        "체육관",
        "선수",
        "경기장",
        "운동장",
        "구장",
        "트랙",
        "땀",
    ),
    "공원": (
        "공원",
        "정원",
        "산책",
        "잔디밭",
        "피크닉",
        "놀이터",
        "숲",
        "벤치",
        "자연",
        "수풀",
        "꽃밭",
        "나무 그늘",
    ),
}

WEAK_CUES: dict[str, tuple[str, ...]] = {
    "카페": ("음료", "빵", "달콤", "차 한 잔"),
    "식당": ("음식", "맛있는", "먹", "토핑", "치즈", "고기"),
    "술집": ("밤", "분위기 좋은 곳"),
    "문화": ("작품", "예술", "그림", "역사", "감상", "구경"),
    "쇼핑": ("옷", "고르", "구경", "마음에 드는"),
    "운동": ("공", "활기", "에너지", "땀 흘"),
    "공원": ("나무", "꽃", "잔디", "물가", "야외"),
}

FALSE_POSITIVE_PATTERNS = {
    "바": ("바람", "바닥", "바깥", "바삭", "바구니", "바다"),
    "공": ("공간", "공기", "공유", "공원"),
    "고르": ("숨을 고르", "숨 고르"),
}

PAIR_ACTIONS: dict[tuple[str, str], str] = {
    ("카페", "술집"): "reject",
    ("술집", "카페"): "reject",
    ("카페", "식당"): "review",
    ("식당", "카페"): "review",
    ("식당", "술집"): "review",
    ("술집", "식당"): "review",
    ("쇼핑", "문화"): "reject",
    ("쇼핑", "운동"): "reject",
    ("쇼핑", "공원"): "review",
    ("쇼핑", "식당"): "review",
    ("쇼핑", "카페"): "review",
    ("쇼핑", "술집"): "reject",
    ("운동", "문화"): "reject",
    ("운동", "쇼핑"): "reject",
    ("운동", "술집"): "reject",
    ("운동", "카페"): "review",
    ("운동", "식당"): "review",
    ("운동", "공원"): "review",
    ("공원", "쇼핑"): "review",
    ("공원", "운동"): "review",
    ("공원", "식당"): "review",
    ("공원", "술집"): "review",
    ("문화", "식당"): "reject",
    ("문화", "술집"): "reject",
    ("문화", "쇼핑"): "review",
    ("문화", "운동"): "reject",
    ("문화", "공원"): "review",
}

AMBIGUOUS_SOURCE_LABELS = {
    "food_court",
    "restaurant_patio",
    "beer_garden",
    "amphitheater",
    "golf_course",
    "ski_slope",
}

BANNED_PHRASES = ("이 사진", "사진에는", "사진은", "이미지", "보인다", "배경에는")
KOREAN_RE = re.compile(r"[가-힣]")
LETTER_RE = re.compile(r"[A-Za-z가-힣]")

LEAKED_PROMPT_PATTERNS = (
    "작성해라",
    "사용자님",
    "요청하신",
    "이미지 분석",
    "사진을 바탕",
    "블로그 초안",
    "블로그 글 초안",
    "생성해야 합니다",
    "결과를 출력",
    "카테고리는",
    "여기서 사용자는",
    "<start_of_turn>",
    "<end_of_turn>",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def korean_ratio(text: str) -> float:
    letters = LETTER_RE.findall(text)
    if not letters:
        return 0.0
    korean = KOREAN_RE.findall(text)
    return len(korean) / len(letters)


def validate_text_quality(text: str) -> list[str]:
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    reasons: list[str] = []

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


def find_keywords(text: str, keywords: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for keyword in keywords:
        if keyword not in text:
            continue
        blocked_patterns = FALSE_POSITIVE_PATTERNS.get(keyword, ())
        if any(pattern in text for pattern in blocked_patterns):
            continue
        hits.append(keyword)
    return hits


def collect_category_hits(text: str) -> dict[str, dict[str, list[str]]]:
    hits: dict[str, dict[str, list[str]]] = {}
    for category, keywords in STRONG_CUES.items():
        strong = find_keywords(text, keywords)
        weak = find_keywords(text, WEAK_CUES.get(category, ()))
        if strong or weak:
            hits[category] = {"strong": strong, "weak": weak}
    return hits


def has_strong_own_signal(label: str, hits: dict[str, dict[str, list[str]]]) -> bool:
    return bool(hits.get(label, {}).get("strong"))


def infer_status(
    label: str,
    source_label: str,
    text: str,
    hits: dict[str, dict[str, list[str]]],
) -> tuple[str, list[str], list[dict[str, Any]], int]:
    reasons: list[str] = []
    conflicts: list[dict[str, Any]] = []
    risk_score = 0

    quality_reasons = validate_text_quality(text)
    if quality_reasons:
        reasons.extend(quality_reasons)
        risk_score += 5

    if any(pattern in text for pattern in LEAKED_PROMPT_PATTERNS):
        reasons.append("prompt_or_control_token_leak")
        risk_score += 5

    own_strong = has_strong_own_signal(label, hits)
    for category, category_hits in sorted(hits.items()):
        if category == label:
            continue
        strong_hits = category_hits["strong"]
        weak_hits = category_hits["weak"]
        if not strong_hits:
            continue

        action = PAIR_ACTIONS.get((label, category), "review")
        reason = {
            "conflict_category": category,
            "strong_keywords": strong_hits,
            "weak_keywords": weak_hits,
            "action": action,
        }
        conflicts.append(reason)

        if own_strong:
            reasons.append(f"mixed_signal:{label}_with_{category}")
            risk_score += 1
        elif action == "reject":
            reasons.append(f"strong_conflict:{label}_to_{category}")
            risk_score += 4
        else:
            reasons.append(f"review_conflict:{label}_to_{category}")
            risk_score += 2

    if source_label in AMBIGUOUS_SOURCE_LABELS:
        reasons.append(f"ambiguous_source_label:{source_label}")
        risk_score += 1

    if not reasons:
        return "accepted", reasons, conflicts, risk_score

    should_reject = bool(quality_reasons)
    should_reject = should_reject or any(reason.startswith("strong_conflict:") for reason in reasons)
    should_reject = should_reject or "prompt_or_control_token_leak" in reasons
    status = "rejected" if should_reject else "review_required"
    return status, sorted(set(reasons)), conflicts, risk_score


def strip_to_dataset_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def annotate_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("generated_text", ""))
    label = str(row.get("label", ""))
    source_label = str(row.get("source_label", ""))
    hits = collect_category_hits(text)
    status, reasons, conflicts, risk_score = infer_status(label, source_label, text, hits)

    output = dict(row)
    output["semantic_status"] = status
    output["semantic_reasons"] = reasons
    output["semantic_conflicts"] = conflicts
    output["semantic_risk_score"] = risk_score
    output["semantic_keyword_hits"] = hits
    return output


def write_review_queue(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "semantic_status",
        "semantic_risk_score",
        "id",
        "label",
        "source_label",
        "split",
        "semantic_reasons",
        "generated_text",
        "image",
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "semantic_status": row.get("semantic_status", ""),
                    "semantic_risk_score": row.get("semantic_risk_score", 0),
                    "id": row.get("id", ""),
                    "label": row.get("label", ""),
                    "source_label": row.get("source_label", ""),
                    "split": row.get("split", ""),
                    "semantic_reasons": ";".join(row.get("semantic_reasons", [])),
                    "generated_text": normalize_text(str(row.get("generated_text", ""))),
                    "image": row.get("image", ""),
                }
            )


def counter_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def build_summary(
    rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    review_required: list[dict[str, Any]],
) -> dict[str, Any]:
    reason_counts = Counter(
        reason
        for row in rejected + review_required
        for reason in row.get("semantic_reasons", [])
    )
    pair_counts = Counter(
        (
            str(row.get("label", "")),
            str(conflict.get("conflict_category", "")),
        )
        for row in rejected + review_required
        for conflict in row.get("semantic_conflicts", [])
    )
    return {
        "input_rows": len(rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "review_required_rows": len(review_required),
        "status_counts": {
            "accepted": len(accepted),
            "rejected": len(rejected),
            "review_required": len(review_required),
        },
        "input_label_counts": counter_by(rows, "label"),
        "accepted_label_counts": counter_by(accepted, "label"),
        "rejected_label_counts": counter_by(rejected, "label"),
        "review_required_label_counts": counter_by(review_required, "label"),
        "reason_counts": dict(reason_counts.most_common()),
        "conflict_pair_counts": {
            f"{label}->{category}": count
            for (label, category), count in pair_counts.most_common()
        },
    }


def write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    def table_from_counts(title: str, counts: dict[str, int]) -> list[str]:
        lines = [f"## {title}", "", "| key | rows |", "| --- | ---: |"]
        lines.extend(f"| {key} | {value} |" for key, value in counts.items())
        lines.append("")
        return lines

    lines = [
        "# Places365 Semantic Draft Filter",
        "",
        "기존 noisy draft로 학습한 SVM은 hard filter에 사용하지 않고, 보수적인 keyword rule만 적용했습니다.",
        "accepted만 학습 데이터로 사용하고 rejected/review_required는 제외 또는 수동 검수 대상으로 둡니다.",
        "",
        "## Status",
        "",
        "| status | rows |",
        "| --- | ---: |",
    ]
    for status, count in summary["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.extend(table_from_counts("Accepted Label Counts", summary["accepted_label_counts"]))
    lines.extend(table_from_counts("Rejected Label Counts", summary["rejected_label_counts"]))
    lines.extend(table_from_counts("Review Required Label Counts", summary["review_required_label_counts"]))
    lines.extend(table_from_counts("Top Reasons", summary["reason_counts"]))
    lines.extend(table_from_counts("Conflict Pairs", summary["conflict_pair_counts"]))
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Places365 Gemma draft의 label/text 의미 충돌 후보를 보수적으로 필터링합니다."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-accepted-metadata",
        action="store_true",
        help="accepted_drafts.jsonl에도 검수 메타데이터를 포함합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    annotated = [annotate_row(row) for row in rows]

    accepted = [row for row in annotated if row["semantic_status"] == "accepted"]
    rejected = [row for row in annotated if row["semantic_status"] == "rejected"]
    review_required = [
        row for row in annotated
        if row["semantic_status"] == "review_required"
    ]

    accepted_output = (
        accepted
        if args.include_accepted_metadata
        else [strip_to_dataset_fields(row) for row in accepted]
    )

    output_dir = args.output_dir
    write_jsonl(output_dir / "accepted_drafts.jsonl", accepted_output)
    write_jsonl(output_dir / "accepted_drafts_with_metadata.jsonl", accepted)
    write_jsonl(output_dir / "rejected_drafts.jsonl", rejected)
    write_jsonl(output_dir / "review_required_drafts.jsonl", review_required)

    queue = sorted(
        rejected + review_required,
        key=lambda row: (
            -int(row.get("semantic_risk_score", 0)),
            str(row.get("label", "")),
            str(row.get("id", "")),
        ),
    )
    write_review_queue(output_dir / "review_queue.csv", queue)

    summary = build_summary(rows, accepted, rejected, review_required)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_markdown(output_dir / "summary.md", summary)

    print(f"input: {len(rows)}")
    print(f"accepted: {len(accepted)} -> {output_dir / 'accepted_drafts.jsonl'}")
    print(f"rejected: {len(rejected)} -> {output_dir / 'rejected_drafts.jsonl'}")
    print(
        "review_required: "
        f"{len(review_required)} -> {output_dir / 'review_required_drafts.jsonl'}"
    )
    print(f"review_queue: {len(queue)} -> {output_dir / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
