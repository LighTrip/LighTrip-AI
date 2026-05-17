from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import read_jsonl, write_jsonl


DATASET_FIELDS = ("id", "image", "generated_text", "label")
DEFAULT_INPUT_DIR = Path("data/category_classifier/places365_v2/manual_review_high_risk")
DEFAULT_OUTPUT_DIR = Path("data/category_classifier/places365_v2/manual_review_strict")


SECOND_PASS_EXCLUDES: dict[str, tuple[str, str]] = {
    "shopping_supermarket_00076": ("ambiguous", "shopping_text_reads_as_restaurant"),
    "shopping_clothing_store_00013": ("mismatch", "shopping_text_reads_as_bar"),
    "shopping_clothing_store_00078": ("ambiguous", "shopping_text_reads_as_culture_or_workshop"),
    "shopping_department_store_00124": ("mismatch", "shopping_text_reads_as_culture"),
    "shopping_gift_shop_00030": ("ambiguous", "shopping_product_without_shop_context"),
    "shopping_gift_shop_00038": ("ambiguous", "shopping_text_reads_as_culture"),
    "shopping_supermarket_00018": ("mismatch", "shopping_text_reads_as_street_food"),
    "shopping_department_store_00012": ("ambiguous", "shopping_generic_without_product_anchor"),
    "shopping_department_store_00046": ("ambiguous", "shopping_generic_without_product_anchor"),
    "shopping_department_store_00055": ("ambiguous", "shopping_product_without_shop_context"),
    "shopping_department_store_00102": ("ambiguous", "shopping_generic_without_product_anchor"),
    "shopping_jewelry_shop_00111": ("mismatch", "shopping_text_reads_as_park_or_greenhouse"),
    "restaurant_fastfood_restaurant_00115": ("ambiguous", "restaurant_beverage_only_without_food_anchor"),
    "restaurant_fastfood_restaurant_00098": ("mismatch", "restaurant_text_reads_as_cafe_or_bar"),
    "bar_bar_00039": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_beer_hall_00013": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00046": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00069": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00097": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00103": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00129": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00266": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00309": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00311": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_beer_hall_00331": ("mismatch", "bar_text_reads_as_restaurant"),
    "bar_bar_00000": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00021": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00032": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00044": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00090": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00126": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00136": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00232": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00308": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_bar_00311": ("mismatch", "bar_text_reads_as_outdoor_or_culture"),
    "bar_bar_00312": ("ambiguous", "bar_generic_night_view_without_drink_anchor"),
    "bar_beer_hall_00047": ("ambiguous", "bar_generic_interior_without_drink_anchor"),
    "bar_beer_hall_00063": ("ambiguous", "bar_generic_atmosphere_without_drink_anchor"),
    "bar_beer_hall_00147": ("ambiguous", "bar_generic_crowded_without_drink_anchor"),
    "bar_beer_hall_00190": ("ambiguous", "bar_friends_night_without_drink_anchor"),
}


def dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in DATASET_FIELDS}


def counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "")) for row in rows))


def render_table(title: str, counts: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| key | rows |", "| --- | ---: |"]
    for key, value in counts.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return lines


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Places365 Strict Manual Review",
        "",
        "manual_review_high_risk 결과에서 직접 재검수 중 애매하게 유지된 샘플을 추가 제외한 보수적 최종본입니다.",
        "",
        "## Totals",
        "",
        "| metric | rows |",
        "| --- | ---: |",
    ]
    for key in (
        "input_accepted_rows",
        "input_excluded_rows",
        "second_pass_newly_excluded_rows",
        "strict_accepted_rows",
        "strict_excluded_rows",
    ):
        lines.append(f"| {key} | {summary[key]} |")
    lines.append("")
    lines.extend(render_table("Second Pass Status", summary["second_pass_status_counts"]))
    lines.extend(render_table("Second Pass Label Counts", summary["second_pass_label_counts"]))
    lines.extend(render_table("Strict Accepted Label Counts", summary["strict_accepted_label_counts"]))
    lines.extend(render_table("Strict Excluded Label Counts", summary["strict_excluded_label_counts"]))
    lines.extend(render_table("Second Pass Notes", summary["second_pass_note_counts"]))
    path.write_text("\n".join(lines), encoding="utf-8")


def apply_strict_review(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    accepted = read_jsonl(input_dir / "accepted_drafts_with_metadata.jsonl")
    excluded = read_jsonl(input_dir / "excluded_drafts.jsonl")
    exclude_ids = set(SECOND_PASS_EXCLUDES)
    accepted_ids = {str(row.get("id", "")) for row in accepted}
    missing_ids = sorted(exclude_ids - accepted_ids)
    if missing_ids:
        raise ValueError(f"second-pass exclude id가 accepted 입력에 없습니다: {missing_ids}")

    strict_accepted: list[dict[str, Any]] = []
    newly_excluded: list[dict[str, Any]] = []
    for row in accepted:
        row_id = str(row.get("id", ""))
        if row_id not in exclude_ids:
            strict_accepted.append(row)
            continue

        status, note = SECOND_PASS_EXCLUDES[row_id]
        updated = dict(row)
        updated["strict_manual_reviewer"] = "codex_second_pass"
        updated["strict_manual_status"] = status
        updated["strict_manual_decision"] = f"exclude_{status}"
        updated["strict_manual_note"] = note
        newly_excluded.append(updated)

    strict_excluded = [*excluded, *newly_excluded]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "accepted_drafts_with_metadata.jsonl", strict_accepted)
    write_jsonl(output_dir / "accepted_drafts.jsonl", [dataset_row(row) for row in strict_accepted])
    write_jsonl(output_dir / "excluded_drafts.jsonl", strict_excluded)
    write_jsonl(output_dir / "newly_excluded_after_second_pass.jsonl", newly_excluded)

    status_counts = Counter(row["strict_manual_status"] for row in newly_excluded)
    note_counts = Counter(row["strict_manual_note"] for row in newly_excluded)
    summary: dict[str, Any] = {
        "input_accepted_rows": len(accepted),
        "input_excluded_rows": len(excluded),
        "second_pass_newly_excluded_rows": len(newly_excluded),
        "strict_accepted_rows": len(strict_accepted),
        "strict_excluded_rows": len(strict_excluded),
        "second_pass_status_counts": dict(status_counts),
        "second_pass_label_counts": counter_by(newly_excluded, "label"),
        "strict_accepted_label_counts": counter_by(strict_accepted, "label"),
        "strict_excluded_label_counts": counter_by(strict_excluded, "label"),
        "second_pass_note_counts": dict(note_counts),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "summary.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply strict second-pass manual exclusions to Places365 drafts.",
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = apply_strict_review(args.input_dir, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
