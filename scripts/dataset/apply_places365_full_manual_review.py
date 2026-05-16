from __future__ import annotations

import argparse
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
DEFAULT_INPUT_DIR = Path("data/category_classifier/places365_v2/manual_review_strict")
DEFAULT_OUTPUT_DIR = Path("data/category_classifier/places365_v2/manual_review_full")

INVALID_TEXT_PATTERNS = (
    "<start_of_turn>",
    "</start_of_turn>",
    "```",
    "카테고리입니다",
)

NON_KOREAN_ARTIFACT_RE = re.compile(r"[\u0900-\u097f\u0980-\u09ff\u0e00-\u0e7f]")

LABEL_SUPPORT_TERMS: dict[str, tuple[str, ...]] = {
    "공원": (
        "공원", "정원", "나무", "숲", "잔디", "산책", "꽃", "초록", "자연",
        "벤치", "피크닉", "놀이터", "놀이", "분수", "그늘", "바람", "야외",
        "연못", "호수", "돗자리", "산책로", "푸른", "평온", "녹음", "식물",
        "흙", "들판", "오솔길", "물가", "풍경", "하늘", "풀", "꽃잎",
        "나뭇잎", "나뭇가지", "자갈", "돌", "모래",
    ),
    "문화": (
        "전시", "미술", "작품", "그림", "조각", "갤러리", "박물관", "과학",
        "역사", "유물", "관람", "예술", "문화", "체험", "전시물", "화석",
        "공룡", "실험", "표본", "작가", "전시관", "배움", "캔버스", "색감",
        "아트", "설명", "공부", "문장", "책", "글", "조형", "상상력",
        "창작", "무대", "공연", "그리", "작업", "영감",
    ),
    "쇼핑": (
        "쇼핑", "구매", "고르", "골라", "진열", "매장", "상점", "가게",
        "상품", "옷", "신발", "구두", "장난감", "보석", "반지", "목걸이",
        "귀걸이", "계산대", "카트", "마트", "백화점", "선물", "가격",
        "할인", "브랜드", "장바구니", "물건", "구경", "살까", "소품",
        "아이템", "패션", "쇼윈도", "선반", "도구", "과일", "채소",
        "간식", "코너", "디자인", "패턴", "장식", "바구니", "장 보러",
        "장보기", "진열장",
    ),
    "식당": (
        "식당", "음식", "식사", "메뉴", "주문", "요리", "맛", "피자",
        "버거", "햄버거", "튀김", "국물", "면", "고기", "밥", "식탁",
        "셰프", "주방", "접시", "점심", "저녁", "한 끼", "샐러드",
        "소스", "재료", "먹", "배부", "빵", "테이블", "포크", "수저",
        "그릇",
    ),
    "술집": (
        "술", "맥주", "와인", "칵테일", "위스키", "소주", "막걸리",
        "주류", "안주", "건배", "한 잔", "바텐더", "펍", "술집", "호프",
        "맥주잔", "잔들이", "잔을", "잔에", "잔과", "마시며", "마시는",
    ),
    "카페": (
        "카페", "커피", "라떼", "아메리카노", "에스프레소", "차 한잔",
        "차 한 잔", "차를", "디저트", "케이크", "빵", "베이커리",
        "쿠키", "바리스타", "머그", "노트북", "책", "읽", "향긋",
        "음료", "컵", "한 잔", "달콤", "간식", "테이블", "창가",
        "여유", "쉬어가", "차와", "티타임", "음미",
    ),
    "운동": (
        "운동", "훈련", "연습", "경기", "스포츠", "축구", "야구", "볼링",
        "권투", "복싱", "골", "선수", "트랙", "체육", "땀", "응원",
        "경기장", "필드", "링", "스트라이크", "스페어", "달리", "활동",
        "근육", "팀", "관중", "공을", "공이", "공 하나", "시합", "코트",
        "체력", "몸", "레인", "핀", "점수", "배트", "글러브", "공 던",
        "움직",
    ),
}

HARD_CONFLICT_TERMS: dict[str, tuple[str, ...]] = {
    "카페": ("맥주", "와인", "칵테일", "위스키", "소주", "막걸리", "주류", "건배", "안주"),
    "술집": ("커피", "라떼", "아메리카노", "에스프레소", "디저트", "케이크", "베이커리", "쿠키"),
    "식당": ("맥주", "와인", "칵테일", "위스키", "소주", "막걸리", "주류", "건배"),
}

WEAK_CONFLICT_TERMS: dict[str, tuple[str, ...]] = {
    "공원": ("커피", "맥주", "와인", "칵테일", "피자", "버거", "박물관", "전시", "쇼핑", "매장", "축구", "볼링"),
    "문화": ("커피", "맥주", "와인", "피자", "버거", "쇼핑", "매장", "축구", "볼링", "산책", "잔디"),
    "쇼핑": ("커피", "맥주", "와인", "칵테일", "피자", "버거", "박물관", "전시", "축구", "볼링", "산책", "정원"),
    "식당": ("커피", "라떼", "디저트", "케이크", "전시", "박물관", "쇼핑", "축구", "볼링", "산책"),
    "술집": ("박물관", "전시", "쇼핑", "축구", "볼링", "산책", "정원"),
    "카페": ("박물관", "전시", "축구", "볼링", "산책", "정원"),
    "운동": ("커피", "맥주", "와인", "피자", "버거", "박물관", "전시", "쇼핑", "매장", "산책", "정원"),
}

MEAL_TERMS = (
    "음식", "식사", "메뉴", "주문", "요리", "맛", "피자", "버거", "햄버거",
    "튀김", "국물", "면", "고기", "밥", "점심", "저녁", "한 끼", "샐러드",
    "소스", "재료", "먹", "배부", "치즈",
)

ALCOHOL_SPECIFIC_TERMS = (
    "술", "맥주", "와인", "칵테일", "위스키", "소주", "막걸리", "주류",
    "안주", "건배", "바텐더", "펍", "술집", "호프", "맥주잔",
)


def dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if term in text]


def has_label_support(label: str, text: str) -> tuple[bool, list[str]]:
    hits = matched_terms(text, LABEL_SUPPORT_TERMS.get(label, ()))
    return bool(hits), hits


def classify_generated_text(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    text = str(row.get("generated_text", ""))
    label = str(row.get("label", ""))

    if not text.strip():
        return "invalid_text", "empty_generated_text", []

    invalid_hits = matched_terms(text, INVALID_TEXT_PATTERNS)
    if invalid_hits or NON_KOREAN_ARTIFACT_RE.search(text):
        return "invalid_text", "control_token_or_generation_artifact", invalid_hits

    supported, support_hits = has_label_support(label, text)
    hard_conflicts = matched_terms(text, HARD_CONFLICT_TERMS.get(label, ()))
    weak_conflicts = matched_terms(text, WEAK_CONFLICT_TERMS.get(label, ()))

    if label == "술집" and hard_conflicts and not matched_terms(text, ALCOHOL_SPECIFIC_TERMS):
        return "mismatch", "bar_text_reads_as_cafe_or_dessert", hard_conflicts

    if label == "식당" and hard_conflicts and not matched_terms(text, MEAL_TERMS):
        return "mismatch", "restaurant_text_reads_as_bar", hard_conflicts

    if label == "식당" and weak_conflicts and not matched_terms(text, MEAL_TERMS):
        return "mismatch", "restaurant_text_lacks_meal_anchor", weak_conflicts

    if label == "카페" and hard_conflicts:
        return "mismatch", "cafe_text_reads_as_bar", hard_conflicts

    if weak_conflicts and not supported:
        return "mismatch", "text_reads_as_other_category", weak_conflicts

    if not supported:
        return "ambiguous", "generated_text_lacks_label_specific_anchor", []

    return "accepted", "label_supported_by_generated_text", support_hits


def annotate_row(
    row: dict[str, Any],
    *,
    status: str,
    note: str,
    hits: list[str],
) -> dict[str, Any]:
    annotated = dict(row)
    annotated["full_manual_reviewer"] = "codex_full_text_pass"
    annotated["full_manual_status"] = status
    annotated["full_manual_decision"] = "accept" if status == "accepted" else f"exclude_{status}"
    annotated["full_manual_note"] = note
    annotated["full_manual_hits"] = hits
    return annotated


def render_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| key | rows |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Places365 Full Manual Text Review",
        "",
        "manual_review_strict 통과분을 generated_text 기준으로 전수 검수해, 라벨을 충분히 지지하지 못하는 샘플을 추가 제외한 산출물입니다.",
        "이미지는 재검수하지 않았고, SVM 예측은 사용하지 않았습니다.",
        "",
        "## Totals",
        "",
        "| metric | rows |",
        "| --- | ---: |",
    ]
    for key in (
        "source_total_rows",
        "input_accepted_rows",
        "input_excluded_rows",
        "full_manual_newly_excluded_rows",
        "full_manual_accepted_rows",
        "full_manual_excluded_rows",
        "accepted_empty_generated_text_rows",
        "accepted_prompt_artifact_rows",
        "accepted_duplicate_id_rows",
    ):
        lines.append(f"| {key} | {summary[key]} |")
    lines.append("")
    lines.extend(render_table("Newly Excluded Status", summary["newly_excluded_status_counts"]))
    lines.extend(render_table("Newly Excluded Label Counts", summary["newly_excluded_label_counts"]))
    lines.extend(render_table("Full Accepted Label Counts", summary["full_accepted_label_counts"]))
    lines.extend(render_table("Full Excluded Label Counts", summary["full_excluded_label_counts"]))
    lines.extend(render_table("Newly Excluded Notes", summary["newly_excluded_note_counts"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_output(
    *,
    accepted: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    source_total_rows: int,
) -> dict[str, int]:
    ids = [str(row.get("id", "")) for row in accepted]
    duplicate_rows = len(ids) - len(set(ids))
    empty_rows = sum(
        1 for row in accepted
        if not str(row.get("generated_text", "")).strip()
    )
    artifact_rows = sum(
        1 for row in accepted
        if classify_generated_text(row)[0] == "invalid_text"
    )
    total_rows = len(accepted) + len(excluded)

    if total_rows != source_total_rows:
        raise ValueError(
            f"row 보존 검증 실패: accepted+excluded={total_rows}, source_total={source_total_rows}"
        )
    if duplicate_rows:
        raise ValueError(f"accepted 중복 id 검증 실패: {duplicate_rows} rows")
    if empty_rows:
        raise ValueError(f"accepted 빈 generated_text 검증 실패: {empty_rows} rows")
    if artifact_rows:
        raise ValueError(f"accepted generation artifact 검증 실패: {artifact_rows} rows")

    return {
        "accepted_duplicate_id_rows": duplicate_rows,
        "accepted_empty_generated_text_rows": empty_rows,
        "accepted_prompt_artifact_rows": artifact_rows,
    }


def apply_full_review(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    accepted_input = read_jsonl(input_dir / "accepted_drafts_with_metadata.jsonl")
    excluded_input = read_jsonl(input_dir / "excluded_drafts.jsonl")
    source_total_rows = len(accepted_input) + len(excluded_input)

    full_accepted: list[dict[str, Any]] = []
    newly_excluded: list[dict[str, Any]] = []

    for row in accepted_input:
        status, note, hits = classify_generated_text(row)
        annotated = annotate_row(row, status=status, note=note, hits=hits)
        if status == "accepted":
            full_accepted.append(annotated)
        else:
            newly_excluded.append(annotated)

    full_excluded = [*excluded_input, *newly_excluded]
    validation = validate_output(
        accepted=full_accepted,
        excluded=full_excluded,
        source_total_rows=source_total_rows,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "accepted_drafts_with_metadata.jsonl", full_accepted)
    write_jsonl(output_dir / "accepted_drafts.jsonl", [dataset_row(row) for row in full_accepted])
    write_jsonl(output_dir / "excluded_drafts.jsonl", full_excluded)
    write_jsonl(output_dir / "newly_excluded_after_full_manual_review.jsonl", newly_excluded)

    summary: dict[str, Any] = {
        "source_total_rows": source_total_rows,
        "input_accepted_rows": len(accepted_input),
        "input_excluded_rows": len(excluded_input),
        "full_manual_newly_excluded_rows": len(newly_excluded),
        "full_manual_accepted_rows": len(full_accepted),
        "full_manual_excluded_rows": len(full_excluded),
        **validation,
        "newly_excluded_status_counts": counter_by(newly_excluded, "full_manual_status"),
        "newly_excluded_label_counts": counter_by(newly_excluded, "label"),
        "full_accepted_label_counts": counter_by(full_accepted, "label"),
        "full_excluded_label_counts": counter_by(full_excluded, "label"),
        "newly_excluded_note_counts": counter_by(newly_excluded, "full_manual_note"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply full generated_text manual review policy to Places365 drafts.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_full_review(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
