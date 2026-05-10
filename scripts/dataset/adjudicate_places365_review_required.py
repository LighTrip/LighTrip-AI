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
DEFAULT_SEMANTIC_DIR = Path("data_places365_2/semantic_filter")
DEFAULT_OUTPUT_DIR = Path("data_places365_2/final_filtered")


ALIGNMENT_CUES: dict[str, tuple[str, ...]] = {
    "카페": (
        "커피",
        "카페",
        "라떼",
        "아메리카노",
        "에스프레소",
        "디저트",
        "케이크",
        "티라미수",
        "음료",
        "차 한 잔",
    ),
    "식당": (
        "음식",
        "식사",
        "한 끼",
        "요리",
        "메뉴",
        "피자",
        "햄버거",
        "치킨",
        "치즈",
        "고기",
        "재료",
        "토핑",
        "바삭",
        "풍미",
    ),
    "술집": (
        "맥주",
        "소주",
        "와인",
        "칵테일",
        "위스키",
        "술집",
        "안주",
        "건배",
        "잔을 기울",
        "취기",
        "취해서",
    ),
    "문화": (
        "전시",
        "관람",
        "박물관",
        "미술관",
        "갤러리",
        "작품",
        "조형물",
        "예술",
        "역사",
        "유물",
        "공룡",
        "화석",
        "뼈",
        "감상",
    ),
    "쇼핑": (
        "쇼핑",
        "구매",
        "사버렸",
        "샀",
        "진열",
        "상품",
        "매장",
        "옷",
        "신발",
        "슈즈",
        "보석",
        "액세서리",
        "소품",
        "아이템",
        "계산대",
        "마트",
    ),
    "운동": (
        "운동",
        "경기",
        "경기장",
        "운동장",
        "축구",
        "야구",
        "농구",
        "골프",
        "스키",
        "복싱",
        "볼링",
        "스포츠",
        "공을 차",
        "땀",
        "선수",
        "응원",
        "아드레날린",
        "체육",
        "구장",
    ),
    "공원": (
        "공원",
        "산책",
        "자연",
        "정원",
        "잔디",
        "잔디밭",
        "놀이터",
        "나무 그늘",
        "수풀",
        "숲",
        "피크닉",
        "벤치",
        "물가",
    ),
}

CORE_ALIGNMENT_CUES: dict[str, set[str]] = {
    "카페": {
        "커피",
        "카페",
        "라떼",
        "아메리카노",
        "에스프레소",
        "디저트",
        "케이크",
        "티라미수",
        "음료",
        "차 한 잔",
    },
    "식당": {
        "음식",
        "식사",
        "한 끼",
        "요리",
        "메뉴",
        "피자",
        "햄버거",
        "치킨",
        "고기",
    },
    "쇼핑": {
        "쇼핑",
        "구매",
        "사버렸",
        "샀",
        "진열",
        "상품",
        "매장",
        "옷",
        "신발",
        "슈즈",
        "보석",
        "액세서리",
        "소품",
        "아이템",
        "계산대",
        "마트",
    },
}

PLAYGROUND_CUES = ("아이", "어린", "놀", "웃음소리", "장난감")
GENERIC_RELAX_CUES = ("여유", "힐링", "쉬어", "숨을 고르", "마무리", "기분")
PROMPT_LEAK_CUES = (
    "여기서 사용자는",
    "가능성이 높아 보이며",
    "카테고리는",
    "요청하신 내용",
    "결과를 출력",
    "블로그 글 초안",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def cue_hits(text: str, label: str) -> list[str]:
    return [cue for cue in ALIGNMENT_CUES.get(label, ()) if cue in text]


def core_hits(text: str, label: str) -> list[str]:
    return [
        cue for cue in ALIGNMENT_CUES.get(label, ())
        if cue in CORE_ALIGNMENT_CUES.get(label, set()) and cue in text
    ]


def conflict_categories(row: dict[str, Any]) -> list[str]:
    return [
        str(conflict.get("conflict_category", ""))
        for conflict in row.get("semantic_conflicts", [])
        if conflict.get("conflict_category")
    ]


def strip_to_dataset_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def is_mixed_signal(row: dict[str, Any]) -> bool:
    return any(
        str(reason).startswith("mixed_signal:")
        for reason in row.get("semantic_reasons", [])
    )


def classify_review_row(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    label = str(row.get("label", ""))
    source_label = str(row.get("source_label", ""))
    text = normalize_text(str(row.get("generated_text", "")))
    own_hits = cue_hits(text, label)
    own_core_hits = core_hits(text, label)
    conflicts = conflict_categories(row)
    conflict_hit_count = sum(
        len(cue_hits(text, category))
        for category in conflicts
        if category != label
    )

    if any(cue in text for cue in PROMPT_LEAK_CUES):
        return "reject_after_review", "prompt_or_classifier_explanation_leak", own_hits

    if label == "운동" and "공원" in conflicts:
        if own_hits:
            return "accept_after_review", "sports_field_text_contains_sports_signal", own_hits
        return "ambiguous", "sports_field_described_as_rest_or_lawn", own_hits

    if label == "식당" and {"카페", "술집"}.intersection(conflicts):
        food_hits = [
            hit for hit in own_hits
            if hit not in {"치즈", "바삭", "풍미", "재료", "토핑"}
        ]
        if food_hits and len(food_hits) >= conflict_hit_count:
            return "accept_after_review", "restaurant_food_signal_outweighs_beverage_signal", own_hits
        return "reject_after_review", "restaurant_label_text_reads_as_cafe_or_bar", own_hits

    if label == "문화" and "공원" in conflicts:
        cultural_hits = [
            hit for hit in own_hits
            if hit in {"전시", "관람", "박물관", "미술관", "갤러리", "역사", "유물", "공룡", "화석", "뼈"}
        ]
        if cultural_hits:
            return "accept_after_review", "museum_or_exhibition_signal_present", own_hits
        return "ambiguous", "culture_label_text_reads_as_nature_or_scenery", own_hits

    if label == "공원":
        if source_label == "playground" and any(cue in text for cue in PLAYGROUND_CUES):
            return "accept_after_review", "playground_activity_signal_present", own_hits
        if own_hits and not {"쇼핑", "식당", "술집"}.intersection(conflicts):
            return "accept_after_review", "park_signal_present", own_hits
        return "ambiguous", "park_label_boundary_unclear", own_hits

    if label == "쇼핑":
        food_or_drink_conflict = {"카페", "식당", "술집"}.intersection(conflicts)
        if source_label == "supermarket" and "진열" in own_hits:
            return "accept_after_review", "supermarket_display_signal_present", own_hits
        if own_core_hits and not food_or_drink_conflict and "운동" not in conflicts:
            return "accept_after_review", "shopping_product_signal_present", own_hits
        if {"카페", "술집", "운동", "문화", "식당"}.intersection(conflicts):
            return "reject_after_review", "shopping_label_text_reads_as_other_service", own_hits
        return "ambiguous", "shopping_label_boundary_unclear", own_hits

    if label == "카페":
        if own_core_hits and "술집" not in conflicts and len(own_core_hits) >= conflict_hit_count:
            return "accept_after_review", "cafe_signal_outweighs_secondary_conflict", own_hits
        if {"술집", "식당", "쇼핑"}.intersection(conflicts):
            return "reject_after_review", "cafe_label_text_reads_as_other_service", own_hits
        return "ambiguous", "cafe_label_boundary_unclear", own_hits

    if label == "술집":
        if own_hits:
            return "accept_after_review", "bar_signal_present", own_hits
        if {"카페", "식당", "문화"}.intersection(conflicts):
            return "reject_after_review", "bar_label_text_reads_as_other_service", own_hits
        return "ambiguous", "bar_label_boundary_unclear", own_hits

    if own_hits and len(own_hits) >= conflict_hit_count:
        return "accept_after_review", "own_label_signal_outweighs_conflict", own_hits

    if any(cue in text for cue in GENERIC_RELAX_CUES):
        return "ambiguous", "generic_relaxation_text_without_label_anchor", own_hits

    return "reject_after_review", "other_category_signal_without_label_anchor", own_hits


def annotate_review_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        decision, note, own_hits = classify_review_row(row)
        output = dict(row)
        output["review_decision"] = decision
        output["review_note"] = note
        output["review_label_cues"] = own_hits
        annotated.append(output)
    return annotated


def write_decision_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "review_decision",
        "review_note",
        "semantic_status",
        "semantic_risk_score",
        "id",
        "label",
        "source_label",
        "split",
        "review_label_cues",
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
                    "review_decision": row.get("review_decision", ""),
                    "review_note": row.get("review_note", ""),
                    "semantic_status": row.get("semantic_status", ""),
                    "semantic_risk_score": row.get("semantic_risk_score", 0),
                    "id": row.get("id", ""),
                    "label": row.get("label", ""),
                    "source_label": row.get("source_label", ""),
                    "split": row.get("split", ""),
                    "review_label_cues": ";".join(row.get("review_label_cues", [])),
                    "semantic_reasons": ";".join(row.get("semantic_reasons", [])),
                    "generated_text": normalize_text(str(row.get("generated_text", ""))),
                    "image": row.get("image", ""),
                }
            )


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field, "")) for row in rows).items()))


def build_summary(
    semantic_accepted: list[dict[str, Any]],
    semantic_rejected: list[dict[str, Any]],
    reviewed: list[dict[str, Any]],
    final_accepted: list[dict[str, Any]],
    final_excluded: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "semantic_accepted_rows": len(semantic_accepted),
        "semantic_rejected_rows": len(semantic_rejected),
        "review_required_rows": len(reviewed),
        "review_decision_counts": count_by(reviewed, "review_decision"),
        "final_accepted_rows": len(final_accepted),
        "final_excluded_rows": len(final_excluded),
        "final_accepted_label_counts": count_by(final_accepted, "label"),
        "final_excluded_label_counts": count_by(final_excluded, "label"),
        "review_note_counts": count_by(reviewed, "review_note"),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    def append_counts(lines: list[str], title: str, counts: dict[str, int]) -> None:
        lines.extend([f"## {title}", "", "| key | rows |", "| --- | ---: |"])
        lines.extend(f"| {key} | {value} |" for key, value in counts.items())
        lines.append("")

    lines = [
        "# Places365 Final Filtered Drafts",
        "",
        "2차 검수는 이미지 재판독 없이 generated_text와 label의 의미 정합성만 기준으로 수행했습니다.",
        "최종 학습 데이터에는 semantic accepted와 accept_after_review만 포함했습니다.",
        "",
        "## Totals",
        "",
        "| metric | rows |",
        "| --- | ---: |",
        f"| semantic accepted | {summary['semantic_accepted_rows']} |",
        f"| semantic rejected | {summary['semantic_rejected_rows']} |",
        f"| review required | {summary['review_required_rows']} |",
        f"| final accepted | {summary['final_accepted_rows']} |",
        f"| final excluded | {summary['final_excluded_rows']} |",
        "",
    ]
    append_counts(lines, "Review Decisions", summary["review_decision_counts"])
    append_counts(lines, "Final Accepted Label Counts", summary["final_accepted_label_counts"])
    append_counts(lines, "Final Excluded Label Counts", summary["final_excluded_label_counts"])
    append_counts(lines, "Review Notes", summary["review_note_counts"])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="semantic_filter review_required 샘플을 2차 판정하고 최종 filtered dataset을 생성합니다."
    )
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic_accepted = read_jsonl(args.semantic_dir / "accepted_drafts_with_metadata.jsonl")
    semantic_rejected = read_jsonl(args.semantic_dir / "rejected_drafts.jsonl")
    review_required = read_jsonl(args.semantic_dir / "review_required_drafts.jsonl")

    reviewed = annotate_review_rows(review_required)
    accept_after_review = [
        row for row in reviewed
        if row["review_decision"] == "accept_after_review"
    ]
    final_accepted_full = semantic_accepted + accept_after_review
    review_excluded = [
        row for row in reviewed
        if row["review_decision"] != "accept_after_review"
    ]
    final_excluded = semantic_rejected + review_excluded

    output_dir = args.output_dir
    write_jsonl(
        output_dir / "accepted_drafts.jsonl",
        [strip_to_dataset_fields(row) for row in final_accepted_full],
    )
    write_jsonl(output_dir / "accepted_drafts_with_metadata.jsonl", final_accepted_full)
    write_jsonl(output_dir / "excluded_drafts.jsonl", final_excluded)
    write_jsonl(output_dir / "review_decisions.jsonl", reviewed)
    write_decision_csv(output_dir / "review_decisions.csv", reviewed)

    summary = build_summary(
        semantic_accepted,
        semantic_rejected,
        reviewed,
        final_accepted_full,
        final_excluded,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary_md(output_dir / "summary.md", summary)

    print(f"semantic accepted: {len(semantic_accepted)}")
    print(f"review accept: {len(accept_after_review)}")
    print(f"final accepted: {len(final_accepted_full)} -> {output_dir / 'accepted_drafts.jsonl'}")
    print(f"final excluded: {len(final_excluded)} -> {output_dir / 'excluded_drafts.jsonl'}")
    print(f"review decisions: {len(reviewed)} -> {output_dir / 'review_decisions.csv'}")


if __name__ == "__main__":
    main()
