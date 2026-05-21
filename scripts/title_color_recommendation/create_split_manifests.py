#!/usr/bin/env python3
"""Create image-level train/val/test manifests for title color recommendation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

try:
    from scripts.title_color_recommendation.common import (
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        TITLE_OUTPUT_ROOT,
        bootstrap_project_imports,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_config_path,
        resolve_output_path,
        resolve_project_path,
        write_csv_rows,
        write_json_file,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        TITLE_OUTPUT_ROOT,
        bootstrap_project_imports,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_config_path,
        resolve_output_path,
        resolve_project_path,
        write_csv_rows,
        write_json_file,
    )

bootstrap_project_imports()

from src.title_color_recommendation.data.split_manifest import (  # noqa: E402
    SPLIT_NAMES,
    SplitRatios,
    apply_split_to_rows,
    category_distribution,
    image_counts_by_split,
    stratified_image_split,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs/title_color_recommendation/default.yaml"
DEFAULT_CLEAN_METADATA = TITLE_DATA_ROOT / "processed/clean_metadata.csv"
DEFAULT_ROI_METADATA = TITLE_DATA_ROOT / "processed/roi_metadata.csv"
DEFAULT_LABELS_INDEX = TITLE_DATA_ROOT / "processed/labels/labels_index.csv"
DEFAULT_SPLIT_DIR = TITLE_DATA_ROOT / "splits"
DEFAULT_REPORT = TITLE_OUTPUT_ROOT / "reports/split_report.md"
DEFAULT_SUMMARY = TITLE_OUTPUT_ROOT / "reports/split_summary.json"

MANIFEST_FIELDS = [
    "id",
    "split",
    "label",
    "category_slug",
    "places365_id",
    "places365_label",
    "places365_slug",
    "source_dataset",
    "source_split",
    "source_index",
    "image_path",
    "clean_path",
    "roi_path",
    "mask_path",
    "label_matrix_index",
    "top1_palette_id",
    "top1_color_name",
    "top1_color_hex",
    "top1_color_group",
    "top1_probability",
    "top1_wcag_pass",
    "background_brightness_bin",
    "background_luminance_mean",
    "entropy_normalized",
    "clean_width",
    "clean_height",
    "brightness",
    "sha256",
    "dhash",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "processed clean/ROI/mask/label metadata를 image id 단위로 묶어 "
            "train/val/test manifest를 생성합니다."
        )
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clean-metadata", type=Path, default=None)
    parser.add_argument("--roi-metadata", type=Path, default=None)
    parser.add_argument("--labels-index", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--image-key", default="id")
    parser.add_argument("--category-key", default="category_slug")
    parser.add_argument(
        "--skip-missing-artifacts",
        action="store_true",
        help="clean/ROI/mask/label artifact가 누락된 행을 오류 대신 제외합니다.",
    )
    return parser.parse_args()


def path_from_config(
    config: dict[str, Any],
    key: str,
    *,
    default: Path,
    output: bool,
    description: str,
) -> Path:
    value = (config.get("paths") or {}).get(key)
    candidate = Path(value) if value else default
    if output:
        return resolve_output_path(candidate, description=description)
    return resolve_project_path(
        candidate,
        allowed_roots=(TITLE_DATA_ROOT,),
        description=description,
        must_exist=True,
    )


def configure(args: argparse.Namespace) -> dict[str, Any]:
    args.config_path = resolve_config_path(args.config_path)
    config = load_config(args.config_path)
    paths = config.get("paths") or {}
    training = config.get("training") or {}
    preprocessing = config.get("preprocessing") or {}

    args.clean_metadata = resolve_project_path(
        args.clean_metadata or DEFAULT_CLEAN_METADATA,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="clean metadata",
        must_exist=True,
    )
    args.roi_metadata = resolve_project_path(
        args.roi_metadata or DEFAULT_ROI_METADATA,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="ROI metadata",
        must_exist=True,
    )
    args.labels_index = resolve_project_path(
        args.labels_index or DEFAULT_LABELS_INDEX,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="labels index",
        must_exist=True,
    )
    args.split_dir = (
        path_from_config(
            config,
            "split_dir",
            default=DEFAULT_SPLIT_DIR,
            output=True,
            description="split directory",
        )
        if args.split_dir is None
        else resolve_output_path(args.split_dir, description="split directory")
    )

    default_report_dir = Path(paths.get("report_dir") or DEFAULT_REPORT.parent)
    report_default = default_report_dir / DEFAULT_REPORT.name
    summary_default = default_report_dir / DEFAULT_SUMMARY.name
    args.report = resolve_output_path(
        args.report or report_default,
        description="split report",
    )
    args.summary = resolve_output_path(
        args.summary or summary_default,
        description="split summary",
    )
    args.seed = int(
        args.seed
        if args.seed is not None
        else training.get("seed", preprocessing.get("crop_seed", 42))
    )
    return config


def index_by_id(
    rows: list[dict[str, str]],
    *,
    description: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in rows:
        row_id = (row.get("id") or "").strip()
        if not row_id:
            raise ValueError(f"{description} contains a row without id")
        if row_id in indexed:
            duplicates.append(row_id)
            continue
        indexed[row_id] = row
    if duplicates:
        examples = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"{description} contains duplicate ids: {examples}")
    return indexed


def existing_relative_path(value: str, *, description: str) -> str:
    path = resolve_project_path(
        value,
        allowed_roots=(TITLE_DATA_ROOT,),
        description=description,
        must_exist=True,
    )
    return project_relative(path)


def build_manifest_records(
    *,
    clean_rows: list[dict[str, str]],
    roi_by_id: dict[str, dict[str, str]],
    label_by_id: dict[str, dict[str, str]],
    skip_missing_artifacts: bool,
) -> tuple[list[dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    missing_examples: list[dict[str, Any]] = []

    for clean in clean_rows:
        image_id = (clean.get("id") or "").strip()
        if not image_id:
            raise ValueError("clean metadata contains a row without id")
        if clean.get("quality_status") and clean.get("quality_status") != "accepted":
            continue

        missing: list[str] = []
        roi = roi_by_id.get(image_id)
        label = label_by_id.get(image_id)
        if roi is None:
            missing.append("roi_metadata")
        if label is None:
            missing.append("label_index")

        clean_path = clean.get("clean_path") or ""
        roi_path = (roi or {}).get("roi_path") or ""
        mask_path = (roi or {}).get("mask_path") or ""
        matrix_index = (label or {}).get("matrix_index") or ""

        for key, value, description in (
            ("clean_path", clean_path, "clean image"),
            ("roi_path", roi_path, "ROI image"),
            ("mask_path", mask_path, "mask image"),
        ):
            if not value:
                missing.append(key)
                continue
            try:
                if key == "clean_path":
                    clean_path = existing_relative_path(value, description=description)
                elif key == "roi_path":
                    roi_path = existing_relative_path(value, description=description)
                else:
                    mask_path = existing_relative_path(value, description=description)
            except FileNotFoundError:
                missing.append(key)

        if not matrix_index:
            missing.append("label_matrix_index")

        if missing:
            missing_counts.update(missing)
            if len(missing_examples) < 20:
                missing_examples.append({"id": image_id, "missing": sorted(set(missing))})
            if skip_missing_artifacts:
                continue

        record = {
            "id": image_id,
            "split": "",
            "label": clean.get("label", ""),
            "category_slug": clean.get("category_slug") or clean.get("label", ""),
            "places365_id": clean.get("places365_id", ""),
            "places365_label": clean.get("places365_label", ""),
            "places365_slug": clean.get("places365_slug", ""),
            "source_dataset": clean.get("source_dataset", ""),
            "source_split": clean.get("source_split", ""),
            "source_index": clean.get("source_index", ""),
            "image_path": clean.get("image_path", ""),
            "clean_path": clean_path,
            "roi_path": roi_path,
            "mask_path": mask_path,
            "label_matrix_index": matrix_index,
            "top1_palette_id": (label or {}).get("top1_palette_id", ""),
            "top1_color_name": (label or {}).get("top1_color_name", ""),
            "top1_color_hex": (label or {}).get("top1_color_hex", ""),
            "top1_color_group": (label or {}).get("top1_color_group", ""),
            "top1_probability": (label or {}).get("top1_probability", ""),
            "top1_wcag_pass": (label or {}).get("top1_wcag_pass", ""),
            "background_brightness_bin": (label or {}).get("background_brightness_bin", ""),
            "background_luminance_mean": (label or {}).get("background_luminance_mean", ""),
            "entropy_normalized": (label or {}).get("entropy_normalized", ""),
            "clean_width": clean.get("clean_width", ""),
            "clean_height": clean.get("clean_height", ""),
            "brightness": clean.get("brightness", ""),
            "sha256": clean.get("sha256", ""),
            "dhash": clean.get("dhash", ""),
        }
        records.append(record)

    if missing_counts and not skip_missing_artifacts:
        examples = json.dumps(missing_examples, ensure_ascii=False)
        raise ValueError(f"artifact 누락으로 split을 중단합니다: {examples}")

    return records, missing_counts, missing_examples


def category_report_rows(
    distribution: dict[str, dict[str, int]],
    image_counts: dict[str, int],
) -> list[dict[str, Any]]:
    total_images = sum(image_counts.values()) or 1
    rows: list[dict[str, Any]] = []
    for category, counts in distribution.items():
        total = counts["total"]
        overall_ratio = total / total_images
        split_ratios = {
            split: counts[split] / image_counts[split]
            if image_counts[split]
            else 0.0
            for split in SPLIT_NAMES
        }
        max_delta_pp = max(
            abs(split_ratios[split] - overall_ratio) * 100.0
            for split in SPLIT_NAMES
        )
        row = {
            "category": category,
            "total": total,
            "overall_ratio": overall_ratio,
            "max_delta_pp": max_delta_pp,
        }
        for split in SPLIT_NAMES:
            row[f"{split}_count"] = counts[split]
            row[f"{split}_ratio"] = split_ratios[split]
        rows.append(row)
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    *,
    ratios: SplitRatios,
    seed: int,
    manifest_paths: dict[str, Path],
    image_counts: dict[str, int],
    row_counts: dict[str, int],
    distribution_rows: list[dict[str, Any]],
    missing_counts: Counter[str],
    missing_examples: list[dict[str, Any]],
) -> str:
    total_images = sum(image_counts.values())
    total_rows = sum(row_counts.values())
    split_rows = []
    for split in SPLIT_NAMES:
        split_rows.append(
            [
                split,
                image_counts[split],
                row_counts[split],
                pct(image_counts[split] / total_images if total_images else 0.0),
                project_relative(manifest_paths[split]),
            ]
        )

    category_rows = []
    for row in distribution_rows:
        category_rows.append(
            [
                row["category"],
                row["total"],
                pct(row["overall_ratio"]),
                row["train_count"],
                pct(row["train_ratio"]),
                row["val_count"],
                pct(row["val_ratio"]),
                row["test_count"],
                pct(row["test_ratio"]),
                f"{row['max_delta_pp']:.3f}pp",
            ]
        )

    max_delta = max((row["max_delta_pp"] for row in distribution_rows), default=0.0)
    artifacts_status = "PASS" if not missing_counts else "WARN"

    lines = [
        "# Title Color Recommendation Split Report",
        "",
        f"- Seed: `{seed}`",
        (
            "- Target ratio: "
            f"train={ratios.train:.2f}, val={ratios.val:.2f}, test={ratios.test:.2f}"
        ),
        f"- Total images: `{total_images}`",
        f"- Total manifest rows: `{total_rows}`",
        f"- Artifact consistency: `{artifacts_status}`",
        f"- Max category distribution delta: `{max_delta:.3f}pp`",
        "",
        "## Split Summary",
        "",
        markdown_table(
            ["split", "images", "rows", "image_ratio", "manifest"],
            split_rows,
        ),
        "",
        "## Category Distribution",
        "",
        markdown_table(
            [
                "category",
                "total",
                "overall",
                "train",
                "train_ratio",
                "val",
                "val_ratio",
                "test",
                "test_ratio",
                "max_delta",
            ],
            category_rows,
        ),
        "",
        "## Artifact Check",
        "",
        (
            "All manifest rows include clean image, ROI, mask, and label matrix index."
            if not missing_counts
            else "Some rows had missing artifacts and were skipped."
        ),
    ]
    if missing_counts:
        lines.extend(
            [
                "",
                "Missing artifact counts:",
                "",
                markdown_table(
                    ["artifact", "count"],
                    [[key, count] for key, count in sorted(missing_counts.items())],
                ),
                "",
                "Examples:",
                "",
                "```json",
                json.dumps(missing_examples, ensure_ascii=False, indent=2),
                "```",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def write_text_file(path: Path, text: str) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"심볼릭 링크 출력 파일은 허용하지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure(args)

    ratios = SplitRatios(
        train=args.train_ratio,
        val=args.val_ratio,
        test=args.test_ratio,
    )
    ratios.validate()

    clean_rows, _ = read_csv_rows(args.clean_metadata)
    roi_rows, _ = read_csv_rows(args.roi_metadata)
    label_rows, _ = read_csv_rows(args.labels_index)
    roi_by_id = index_by_id(roi_rows, description="ROI metadata")
    label_by_id = index_by_id(label_rows, description="labels index")

    records, missing_counts, missing_examples = build_manifest_records(
        clean_rows=clean_rows,
        roi_by_id=roi_by_id,
        label_by_id=label_by_id,
        skip_missing_artifacts=args.skip_missing_artifacts,
    )
    assignments = stratified_image_split(
        records,
        seed=args.seed,
        ratios=ratios,
        image_key=args.image_key,
        category_key=args.category_key,
    )
    split_rows = apply_split_to_rows(
        records,
        assignments,
        seed=args.seed,
        image_key=args.image_key,
        category_key=args.category_key,
    )

    ensure_output_dir(args.split_dir)
    manifest_paths: dict[str, Path] = {}
    for split in SPLIT_NAMES:
        path = args.split_dir / f"{split}.csv"
        write_csv_rows(path, split_rows[split], MANIFEST_FIELDS)
        manifest_paths[split] = path

    image_counts = image_counts_by_split(assignments)
    row_counts = {split: len(split_rows[split]) for split in SPLIT_NAMES}
    distribution = category_distribution(
        records,
        assignments,
        image_key=args.image_key,
        category_key=args.category_key,
    )
    distribution_rows = category_report_rows(distribution, image_counts)

    report = build_report(
        ratios=ratios,
        seed=args.seed,
        manifest_paths=manifest_paths,
        image_counts=image_counts,
        row_counts=row_counts,
        distribution_rows=distribution_rows,
        missing_counts=missing_counts,
        missing_examples=missing_examples,
    )
    write_text_file(args.report, report)

    summary = {
        "seed": args.seed,
        "ratios": ratios.as_dict(),
        "source": {
            "clean_metadata": project_relative(args.clean_metadata),
            "roi_metadata": project_relative(args.roi_metadata),
            "labels_index": project_relative(args.labels_index),
        },
        "split_dir": project_relative(args.split_dir),
        "report": project_relative(args.report),
        "manifests": {
            split: project_relative(path) for split, path in manifest_paths.items()
        },
        "image_counts": image_counts,
        "row_counts": row_counts,
        "category_distribution": distribution,
        "category_distribution_delta_pp": {
            row["category"]: round(row["max_delta_pp"], 6)
            for row in distribution_rows
        },
        "missing_artifact_counts": dict(sorted(missing_counts.items())),
        "missing_artifact_examples": missing_examples,
    }
    write_json_file(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
