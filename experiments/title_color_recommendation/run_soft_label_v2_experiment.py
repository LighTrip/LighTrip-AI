from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.title_color_recommendation.labeling.soft_labels import (
    SoftLabelConfig,
    compute_pseudo_score,
    normalized_distribution_entropy,
    soft_label_config_from_mapping,
    softmax,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/title_color_recommendation/soft_label_v2.yaml")
DEFAULT_LABEL_DIR = Path("data/title_color_recommendation/processed/labels")
DEFAULT_V1_LABELS_SOFT = DEFAULT_LABEL_DIR / "labels_soft.csv"
DEFAULT_V1_LABELS_MATRIX = DEFAULT_LABEL_DIR / "labels_matrix.npy"
DEFAULT_V2_LABELS_SOFT = DEFAULT_LABEL_DIR / "labels_soft_v2.csv"
DEFAULT_V2_LABELS_MATRIX = DEFAULT_LABEL_DIR / "labels_matrix_v2.npy"
DEFAULT_PREVIEW_MANIFEST = Path("data/title_color_recommendation/splits/test.csv")
DEFAULT_SUMMARY_PATH = Path("outputs/reports/soft_label_v2_summary.json")
DEFAULT_ANALYSIS_REPORT = Path("outputs/reports/soft_label_v2_analysis.md")
DEFAULT_COMPARISON_REPORT = Path("outputs/reports/soft_label_v1_vs_v2_comparison.md")
DEFAULT_PREVIEW_DIR = Path("outputs/reports/soft_label_v2_preview")
LABEL_COLUMNS = (
    "id",
    "split",
    "palette_id",
    "color_name",
    "color_hex",
    "color_group",
    "readability_score",
    "aesthetic_prior",
    "tone_match_score",
    "simplicity_score",
    "fail_penalty",
    "pseudo_score",
    "target_probability",
    "temperature",
    "rank",
    "wcag_pass",
)
REQUIRED_LABEL_COLUMNS = frozenset(LABEL_COLUMNS)
COMPONENT_COLUMNS = (
    "readability_score",
    "aesthetic_prior",
    "tone_match_score",
    "simplicity_score",
    "fail_penalty",
    "pseudo_score",
    "target_probability",
)
SUMMARY_METRIC_KEYS = (
    "top1_probability_mean",
    "top5_mass_mean",
    "entropy_normalized_mean",
    "effective_colors_mean",
    "top1_wcag_pass_rate",
    "top5_any_wcag_pass_rate",
    "wcag_probability_mass_mean",
    "max_color_share",
    "top1_fail_when_pass_available_ratio",
)
ANALYSIS_METRIC_KEYS = (
    "images",
    "row_sum_max_error",
    *SUMMARY_METRIC_KEYS[:-1],
)


@dataclass(frozen=True)
class VariantGenerationResult:
    labels_soft_path: Path
    labels_matrix_path: Path
    processed_images: int
    num_classes: int
    probability_sum_max_error: float


@dataclass(frozen=True)
class SoftLabelExperimentResult:
    v1_summary: dict[str, Any]
    v2_summary: dict[str, Any]
    generation_result: VariantGenerationResult | None
    summary_path: Path
    analysis_report_path: Path
    comparison_report_path: Path
    preview_manifest_path: Path | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and compare soft label v2 from existing component scores."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v1-labels-soft", type=Path, default=DEFAULT_V1_LABELS_SOFT)
    parser.add_argument(
        "--v1-labels-matrix",
        type=Path,
        default=DEFAULT_V1_LABELS_MATRIX,
    )
    parser.add_argument("--v2-labels-soft", type=Path, default=DEFAULT_V2_LABELS_SOFT)
    parser.add_argument(
        "--v2-labels-matrix",
        type=Path,
        default=DEFAULT_V2_LABELS_MATRIX,
    )
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument(
        "--analysis-report-path",
        type=Path,
        default=DEFAULT_ANALYSIS_REPORT,
    )
    parser.add_argument(
        "--comparison-report-path",
        type=Path,
        default=DEFAULT_COMPARISON_REPORT,
    )
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_DIR)
    parser.add_argument(
        "--preview-manifest",
        type=Path,
        default=DEFAULT_PREVIEW_MANIFEST,
    )
    parser.add_argument("--preview-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--skip-generation", action="store_true")
    return parser.parse_args(argv)


def is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_project_path(
    value: str | Path,
    *,
    must_exist: bool = False,
    description: str = "path",
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=False)
    project_root = PROJECT_ROOT.resolve()
    if not is_relative_to(resolved, project_root):
        raise ValueError(f"{description} must be inside project root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {value}")
    return resolved


def ensure_writable_file(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"refusing to write symlink file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_writable_dir(path: Path) -> None:
    if path.exists() and path.is_symlink():
        raise ValueError(f"refusing to write symlink directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def format_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def parse_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def row_float(row: Mapping[str, str], column: str) -> float:
    value = str(row.get(column, "")).strip()
    if not value:
        raise ValueError(f"{column} is required for id={row.get('id', '')}")
    return float(value)


def label_fieldnames(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = list(reader.fieldnames or [])
    missing = sorted(REQUIRED_LABEL_COLUMNS.difference(fieldnames))
    if missing:
        raise ValueError(f"labels csv missing columns: {missing}")
    return fieldnames


def iter_image_groups(path: Path) -> Iterator[list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        missing = sorted(REQUIRED_LABEL_COLUMNS.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"labels csv missing columns: {missing}")

        current_id: str | None = None
        current_rows: list[dict[str, str]] = []
        completed_ids: set[str] = set()
        for row in reader:
            image_id = str(row["id"])
            if current_id is None:
                current_id = image_id
            if image_id != current_id:
                completed_ids.add(current_id)
                yield current_rows
                if image_id in completed_ids:
                    raise ValueError(
                        "labels csv must be grouped by id; repeated id="
                        f"{image_id}"
                    )
                current_id = image_id
                current_rows = []
            current_rows.append(dict(row))

        if current_rows:
            yield current_rows


def ordered_rows_and_arrays(
    rows: list[dict[str, str]],
    *,
    num_classes: int,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    by_palette: list[dict[str, str] | None] = [None] * num_classes
    for row in rows:
        palette_id = int(row["palette_id"])
        if not 0 <= palette_id < num_classes:
            raise ValueError(
                f"palette_id out of range for id={row.get('id', '')}: "
                f"{palette_id}"
            )
        if by_palette[palette_id] is not None:
            raise ValueError(
                f"duplicate palette_id for id={row.get('id', '')}: {palette_id}"
            )
        by_palette[palette_id] = row

    if any(row is None for row in by_palette):
        image_id = rows[0].get("id", "") if rows else ""
        raise ValueError(f"incomplete palette rows for id={image_id}")

    ordered_rows = [row for row in by_palette if row is not None]
    arrays = {
        column: np.asarray(
            [row_float(row, column) for row in ordered_rows],
            dtype=np.float32,
        )
        for column in COMPONENT_COLUMNS
    }
    arrays["wcag_pass"] = np.asarray(
        [float(parse_bool(row["wcag_pass"])) for row in ordered_rows],
        dtype=np.float32,
    )
    return ordered_rows, arrays


def update_variant_rows(
    rows: list[dict[str, str]],
    *,
    config: SoftLabelConfig,
    num_classes: int,
) -> tuple[list[dict[str, str]], np.ndarray]:
    ordered_rows, arrays = ordered_rows_and_arrays(rows, num_classes=num_classes)
    pseudo_score = compute_pseudo_score(
        readability_score=arrays["readability_score"],
        aesthetic_prior=arrays["aesthetic_prior"],
        tone_match_score=arrays["tone_match_score"],
        simplicity_score=arrays["simplicity_score"],
        fail_penalty=arrays["fail_penalty"],
        config=config,
    )
    probabilities = softmax(pseudo_score, temperature=config.temperature)
    ranks = np.empty(num_classes, dtype=np.int32)
    ranks[np.argsort(-probabilities)] = np.arange(1, num_classes + 1)

    updated_rows: list[dict[str, str]] = []
    for palette_index, row in enumerate(ordered_rows):
        updated = dict(row)
        updated["pseudo_score"] = format_float(pseudo_score[palette_index])
        updated["target_probability"] = format_float(
            probabilities[palette_index],
            digits=8,
        )
        updated["temperature"] = format_float(config.temperature, digits=3)
        updated["rank"] = str(int(ranks[palette_index]))
        updated_rows.append(updated)
    return updated_rows, probabilities.astype(np.float32)


def generate_soft_label_variant(
    *,
    source_labels_soft: Path,
    output_labels_soft: Path,
    output_labels_matrix: Path,
    config: SoftLabelConfig,
    progress_every: int = 0,
) -> VariantGenerationResult:
    fieldnames = label_fieldnames(source_labels_soft)
    ensure_writable_file(output_labels_soft)
    ensure_writable_file(output_labels_matrix)

    processed = 0
    num_classes: int | None = None
    matrix_rows: list[np.ndarray] = []
    probability_sum_max_error = 0.0

    with output_labels_soft.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rows in iter_image_groups(source_labels_soft):
            if num_classes is None:
                num_classes = len(rows)
            if len(rows) != num_classes:
                raise ValueError(
                    f"unexpected class count for id={rows[0].get('id', '')}: "
                    f"{len(rows)} != {num_classes}"
                )
            updated_rows, probabilities = update_variant_rows(
                rows,
                config=config,
                num_classes=num_classes,
            )
            for row in updated_rows:
                writer.writerow(row)
            matrix_rows.append(probabilities)
            probability_sum_max_error = max(
                probability_sum_max_error,
                abs(float(np.sum(probabilities)) - 1.0),
            )
            processed += 1
            if progress_every and processed % progress_every == 0:
                LOGGER.info("[PROGRESS] generated v2 labels for %s images", processed)

    final_num_classes = int(num_classes or 0)
    matrix = (
        np.vstack(matrix_rows).astype(np.float32)
        if matrix_rows
        else np.empty((0, final_num_classes), dtype=np.float32)
    )
    np.save(output_labels_matrix, matrix)
    return VariantGenerationResult(
        labels_soft_path=output_labels_soft,
        labels_matrix_path=output_labels_matrix,
        processed_images=processed,
        num_classes=final_num_classes,
        probability_sum_max_error=probability_sum_max_error,
    )


def matrix_distribution_stats(matrix_path: Path) -> dict[str, Any]:
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"labels matrix must be 2D: shape={matrix.shape}")
    if matrix.shape[1] <= 0:
        raise ValueError(f"labels matrix has no class dimension: {matrix.shape}")

    values = np.asarray(matrix, dtype=np.float32)
    row_sums = values.sum(axis=1)
    entropy = -(values * np.log(np.clip(values, 1e-12, 1.0))).sum(axis=1)
    top_probs = np.sort(values, axis=1)
    return {
        "matrix_shape": [int(values.shape[0]), int(values.shape[1])],
        "row_sum_min": float(row_sums.min()) if row_sums.size else 0.0,
        "row_sum_max": float(row_sums.max()) if row_sums.size else 0.0,
        "row_sum_max_error": (
            float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else 0.0
        ),
        "top1_probability_mean": float(values.max(axis=1).mean())
        if values.size
        else 0.0,
        "top1_probability_p95": float(np.percentile(values.max(axis=1), 95))
        if values.size
        else 0.0,
        "entropy_mean": float(entropy.mean()) if entropy.size else 0.0,
        "entropy_normalized_mean": float((entropy / np.log(values.shape[1])).mean())
        if entropy.size
        else 0.0,
        "effective_colors_mean": float(np.exp(entropy).mean())
        if entropy.size
        else 0.0,
        "top5_mass_mean": float(top_probs[:, -5:].sum(axis=1).mean())
        if values.size
        else 0.0,
    }


def empty_component_stats() -> dict[str, dict[str, float]]:
    return {
        column: {"min": math.inf, "mean": 0.0, "max": -math.inf}
        for column in COMPONENT_COLUMNS
    }


def update_component_stats(
    stats: dict[str, dict[str, float]],
    arrays: Mapping[str, np.ndarray],
    *,
    row_count: int,
) -> int:
    next_count = row_count
    for column in COMPONENT_COLUMNS:
        values = arrays[column]
        stats[column]["min"] = min(stats[column]["min"], float(values.min()))
        stats[column]["max"] = max(stats[column]["max"], float(values.max()))
        stats[column]["mean"] += float(values.sum())
    return next_count + len(arrays[COMPONENT_COLUMNS[0]])


def finalize_component_stats(
    stats: dict[str, dict[str, float]],
    *,
    row_count: int,
) -> dict[str, dict[str, float]]:
    if row_count <= 0:
        return {
            column: {"min": 0.0, "mean": 0.0, "max": 0.0}
            for column in COMPONENT_COLUMNS
        }
    return {
        column: {
            "min": values["min"],
            "mean": values["mean"] / row_count,
            "max": values["max"],
        }
        for column, values in stats.items()
    }


def summarize_counter(
    counter: Counter[int],
    metadata: Mapping[int, Mapping[str, str]],
    *,
    total: int,
    limit: int = 32,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_or_one = total or 1
    for palette_id, count in counter.most_common(limit):
        item = metadata.get(palette_id, {})
        rows.append(
            {
                "palette_id": palette_id,
                "name": item.get("name", ""),
                "hex": item.get("hex", ""),
                "group": item.get("group", ""),
                "count": count,
                "ratio": count / total_or_one,
            }
        )
    return rows


def summarize_soft_labels(
    *,
    labels_soft_path: Path,
    labels_matrix_path: Path,
) -> dict[str, Any]:
    matrix_stats = matrix_distribution_stats(labels_matrix_path)
    num_classes = int(matrix_stats["matrix_shape"][1])
    image_count = 0
    top1_wcag_pass_count = 0
    any_wcag_pass_count = 0
    top1_fail_when_pass_available_count = 0
    top5_any_wcag_pass_count = 0
    wcag_probability_mass_total = 0.0
    top1_color_counts: Counter[int] = Counter()
    top1_group_counts: Counter[str] = Counter()
    group_probability_mass: dict[str, float] = defaultdict(float)
    palette_metadata: dict[int, dict[str, str]] = {}
    component_stats = empty_component_stats()
    component_row_count = 0

    for rows in iter_image_groups(labels_soft_path):
        ordered_rows, arrays = ordered_rows_and_arrays(rows, num_classes=num_classes)
        probabilities = arrays["target_probability"]
        wcag_pass = arrays["wcag_pass"].astype(bool)
        top1_index = int(np.argmax(probabilities))
        top1_row = ordered_rows[top1_index]
        top1_palette_id = int(top1_row["palette_id"])
        top1_group = str(top1_row["color_group"])
        top5_indices = np.argsort(-probabilities)[:5]
        has_any_wcag_pass = bool(np.any(wcag_pass))

        for row in ordered_rows:
            palette_id = int(row["palette_id"])
            palette_metadata.setdefault(
                palette_id,
                {
                    "name": str(row["color_name"]),
                    "hex": str(row["color_hex"]),
                    "group": str(row["color_group"]),
                },
            )

        image_count += 1
        top1_color_counts[top1_palette_id] += 1
        top1_group_counts[top1_group] += 1
        top1_wcag_pass_count += int(bool(wcag_pass[top1_index]))
        any_wcag_pass_count += int(has_any_wcag_pass)
        top1_fail_when_pass_available_count += int(
            has_any_wcag_pass and not bool(wcag_pass[top1_index])
        )
        top5_any_wcag_pass_count += int(bool(np.any(wcag_pass[top5_indices])))
        wcag_probability_mass_total += float(probabilities[wcag_pass].sum())
        for row, probability in zip(ordered_rows, probabilities):
            group_probability_mass[str(row["color_group"])] += float(probability)
        component_row_count = update_component_stats(
            component_stats,
            arrays,
            row_count=component_row_count,
        )

    total_or_one = image_count or 1
    any_pass_or_one = any_wcag_pass_count or 1
    summary = dict(matrix_stats)
    summary.update(
        {
            "labels_soft": str(labels_soft_path),
            "labels_matrix": str(labels_matrix_path),
            "images": image_count,
            "top1_wcag_pass_rate": top1_wcag_pass_count / total_or_one,
            "any_wcag_pass_ratio": any_wcag_pass_count / total_or_one,
            "top1_fail_when_pass_available_ratio": (
                top1_fail_when_pass_available_count / any_pass_or_one
            ),
            "top5_any_wcag_pass_rate": top5_any_wcag_pass_count / total_or_one,
            "wcag_probability_mass_mean": (
                wcag_probability_mass_total / total_or_one
            ),
            "max_color_share": (
                max(top1_color_counts.values()) / total_or_one
                if top1_color_counts
                else 0.0
            ),
            "top1_color_distribution": summarize_counter(
                top1_color_counts,
                palette_metadata,
                total=image_count,
            ),
            "top1_group_distribution": {
                group: {
                    "count": count,
                    "ratio": count / total_or_one,
                }
                for group, count in top1_group_counts.most_common()
            },
            "target_probability_mass_by_group": {
                group: mass / total_or_one
                for group, mass in sorted(
                    group_probability_mass.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            },
            "component_stats": finalize_component_stats(
                component_stats,
                row_count=component_row_count,
            ),
        }
    )
    return summary


def metric_value(summary: Mapping[str, Any], key: str) -> float:
    value = summary.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def metric_table(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> list[str]:
    rows = [
        "| metric | v1 | v2 | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in SUMMARY_METRIC_KEYS:
        before = metric_value(v1, key)
        after = metric_value(v2, key)
        rows.append(f"| `{key}` | {before:.6f} | {after:.6f} | {after - before:+.6f} |")
    return rows


def top_color_table(summary: Mapping[str, Any], *, title: str) -> list[str]:
    rows = [f"### {title}", "", "| rank | palette_id | color | group | share |"]
    rows.append("| ---: | ---: | --- | --- | ---: |")
    colors = list(summary.get("top1_color_distribution", []))[:10]
    for rank, item in enumerate(colors, start=1):
        rows.append(
            f"| {rank} | {item['palette_id']} | {item['name']} | "
            f"{item['group']} | {float(item['ratio']):.4f} |"
        )
    return rows


def write_v2_analysis_report(
    path: Path,
    *,
    config: SoftLabelConfig,
    summary: Mapping[str, Any],
    preview_dir: Path,
) -> None:
    ensure_writable_file(path)
    lines = [
        "# Soft Label v2 Analysis",
        "",
        "## Configuration",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| temperature | {config.temperature:.4f} |",
        f"| readability_weight | {config.readability_weight:.4f} |",
        f"| aesthetic_weight | {config.aesthetic_weight:.4f} |",
        f"| tone_match_weight | {config.tone_match_weight:.4f} |",
        f"| simplicity_weight | {config.simplicity_weight:.4f} |",
        f"| fail_penalty_weight | {config.fail_penalty_weight:.4f} |",
        "",
        "## Distribution Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ANALYSIS_METRIC_KEYS:
        value = summary.get(key, "")
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:.6f} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    lines.extend(["", *top_color_table(summary, title="Top1 Color Distribution")])
    lines.extend(
        [
            "",
            "## Preview",
            "",
            f"- preview_dir: `{preview_dir.as_posix()}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(
    path: Path,
    *,
    v1_summary: Mapping[str, Any],
    v2_summary: Mapping[str, Any],
    preview_dir: Path,
) -> None:
    ensure_writable_file(path)
    lines = [
        "# Soft Label v1 vs v2 Comparison",
        "",
        "## Summary",
        "",
        *metric_table(v1_summary, v2_summary),
        "",
        "## Top Colors",
        "",
        *top_color_table(v1_summary, title="v1 Top1 Colors"),
        "",
        *top_color_table(v2_summary, title="v2 Top1 Colors"),
        "",
        "## Training Commands",
        "",
        "Use the same model and hyperparameters while changing only label inputs.",
        "",
        "```bash",
        "python experiments/title_color_recommendation/run_full_training.py \\",
        "  --config configs/title_color_recommendation/full_training.yaml \\",
        "  --learning-rate 0.0005 \\",
        "  --weight-decay 0.0001 \\",
        "  --batch-size 64 \\",
        "  --epochs 20 \\",
        "  --scheduler cosine \\",
        "  --labels-matrix data/title_color_recommendation/processed/labels/labels_matrix_v2.npy \\",
        "  --labels-soft data/title_color_recommendation/processed/labels/labels_soft_v2.csv \\",
        "  --checkpoint-dir outputs/checkpoints/soft_label_v2 \\",
        "  --log-path outputs/logs/soft_label_v2_training_metrics.jsonl \\",
        "  --report-path outputs/reports/soft_label_v2_full_training_report.md \\",
        "  --loss-plot-path outputs/reports/soft_label_v2_loss_curve.png \\",
        "  --ndcg-plot-path outputs/reports/soft_label_v2_ndcg5_curve.png \\",
        "  --color-plot-path outputs/reports/soft_label_v2_color_distribution.png",
        "```",
        "",
        "## Preview",
        "",
        f"- preview_dir: `{preview_dir.as_posix()}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def select_preview_items(
    manifest_path: Path,
    *,
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    if count <= 0:
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"id", "roi_path", "mask_path"}
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"preview manifest missing columns: {missing}")
        rows = [dict(row) for row in reader]
    rows.sort(key=lambda row: stable_hash(seed, str(row.get("id", ""))))
    return rows[:count]


def top_entries_for_ids(
    labels_soft_path: Path,
    image_ids: Iterable[str],
    *,
    top_k: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    wanted = {str(image_id) for image_id in image_ids}
    found: dict[str, list[dict[str, Any]]] = {}
    if not wanted:
        return found
    for rows in iter_image_groups(labels_soft_path):
        image_id = str(rows[0]["id"])
        if image_id not in wanted:
            continue
        entries = sorted(
            rows,
            key=lambda row: float(row["target_probability"]),
            reverse=True,
        )[:top_k]
        found[image_id] = [
            {
                "palette_id": int(row["palette_id"]),
                "name": row["color_name"],
                "hex": row["color_hex"],
                "group": row["color_group"],
                "target_probability": float(row["target_probability"]),
                "wcag_pass": parse_bool(row["wcag_pass"]),
            }
            for row in entries
        ]
        if len(found) == len(wanted):
            break
    return found


def sanitize_filename(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe)[:96] or "sample"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"invalid hex color: {value}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16))


def overlay_mask_text(
    roi_image: Image.Image,
    mask_image: Image.Image,
    color_hex: str,
) -> Image.Image:
    base = roi_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, hex_to_rgb(color_hex) + (0,))
    overlay.putalpha(mask_image.convert("L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def scale_to_height(image: Image.Image, *, height: int) -> Image.Image:
    if image.height == height:
        return image.copy()
    width = max(1, round(image.width * (height / image.height)))
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def preview_panel(
    *,
    image_id: str,
    roi_image: Image.Image,
    mask_image: Image.Image,
    v1_top: Mapping[str, Any],
    v2_top: Mapping[str, Any],
) -> Image.Image:
    panels = [
        ("original", roi_image.convert("RGB")),
        (
            f"v1 {v1_top['name']} {v1_top['target_probability']:.3f}",
            overlay_mask_text(roi_image, mask_image, str(v1_top["hex"])),
        ),
        (
            f"v2 {v2_top['name']} {v2_top['target_probability']:.3f}",
            overlay_mask_text(roi_image, mask_image, str(v2_top["hex"])),
        ),
    ]
    height = 144
    scaled = [(label, scale_to_height(image, height=height)) for label, image in panels]
    margin = 14
    gap = 12
    header_height = 24
    label_height = 22
    canvas_width = (
        (margin * 2)
        + sum(image.width for _, image in scaled)
        + (gap * (len(scaled) - 1))
    )
    canvas_height = margin + header_height + label_height + height + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), image_id, fill=(17, 24, 39))
    x = margin
    y = margin + header_height
    for label, image in scaled:
        draw.text((x, y), label, fill=(31, 41, 55))
        canvas.paste(image, (x, y + label_height))
        x += image.width + gap
    return canvas


def write_preview_images(
    *,
    preview_dir: Path,
    manifest_path: Path,
    v1_labels_soft: Path,
    v2_labels_soft: Path,
    preview_count: int,
    seed: int,
) -> Path | None:
    items = select_preview_items(manifest_path, count=preview_count, seed=seed)
    if not items:
        return None
    ensure_writable_dir(preview_dir)
    image_ids = [str(item["id"]) for item in items]
    v1_top_entries = top_entries_for_ids(v1_labels_soft, image_ids)
    v2_top_entries = top_entries_for_ids(v2_labels_soft, image_ids)
    manifest_rows: list[dict[str, Any]] = []

    for order, item in enumerate(items, start=1):
        image_id = str(item["id"])
        if image_id not in v1_top_entries or image_id not in v2_top_entries:
            continue
        roi_path = resolve_project_path(
            item["roi_path"],
            must_exist=True,
            description="preview ROI",
        )
        mask_path = resolve_project_path(
            item["mask_path"],
            must_exist=True,
            description="preview mask",
        )
        with Image.open(roi_path) as roi_opened, Image.open(mask_path) as mask_opened:
            roi_opened.load()
            mask_opened.load()
            panel = preview_panel(
                image_id=image_id,
                roi_image=roi_opened.convert("RGB"),
                mask_image=mask_opened.convert("L"),
                v1_top=v1_top_entries[image_id][0],
                v2_top=v2_top_entries[image_id][0],
            )
        filename = f"{order:02d}_{sanitize_filename(image_id)}.png"
        output_path = preview_dir / filename
        if output_path.exists() and output_path.is_symlink():
            raise ValueError(f"refusing to write symlink file: {output_path}")
        panel.save(output_path)
        manifest_rows.append(
            {
                "image_id": image_id,
                "preview_path": output_path.as_posix(),
                "v1_top3": v1_top_entries[image_id],
                "v2_top3": v2_top_entries[image_id],
            }
        )

    manifest_output = preview_dir / "manifest.json"
    ensure_writable_file(manifest_output)
    manifest_output.write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_output


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.config_path = resolve_project_path(
        args.config_path,
        must_exist=True,
        description="config path",
    )
    args.v1_labels_soft = resolve_project_path(
        args.v1_labels_soft,
        must_exist=True,
        description="v1 labels soft",
    )
    args.v1_labels_matrix = resolve_project_path(
        args.v1_labels_matrix,
        must_exist=True,
        description="v1 labels matrix",
    )
    args.v2_labels_soft = resolve_project_path(
        args.v2_labels_soft,
        description="v2 labels soft",
    )
    args.v2_labels_matrix = resolve_project_path(
        args.v2_labels_matrix,
        description="v2 labels matrix",
    )
    args.summary_path = resolve_project_path(
        args.summary_path,
        description="summary path",
    )
    args.analysis_report_path = resolve_project_path(
        args.analysis_report_path,
        description="analysis report path",
    )
    args.comparison_report_path = resolve_project_path(
        args.comparison_report_path,
        description="comparison report path",
    )
    args.preview_dir = resolve_project_path(args.preview_dir, description="preview dir")
    args.preview_manifest = resolve_project_path(
        args.preview_manifest,
        must_exist=True,
        description="preview manifest",
    )
    if args.preview_count < 0:
        raise ValueError("--preview-count must be non-negative")
    if args.progress_every < 0:
        raise ValueError("--progress-every must be non-negative")
    return args


def write_summary_json(
    path: Path,
    *,
    config: SoftLabelConfig,
    v1_summary: Mapping[str, Any],
    v2_summary: Mapping[str, Any],
    generation_result: VariantGenerationResult | None,
) -> None:
    ensure_writable_file(path)
    payload = {
        "config": {
            "temperature": config.temperature,
            "readability_weight": config.readability_weight,
            "aesthetic_weight": config.aesthetic_weight,
            "tone_match_weight": config.tone_match_weight,
            "simplicity_weight": config.simplicity_weight,
            "fail_penalty_weight": config.fail_penalty_weight,
        },
        "generation": (
            None
            if generation_result is None
            else {
                "processed_images": generation_result.processed_images,
                "num_classes": generation_result.num_classes,
                "probability_sum_max_error": (
                    generation_result.probability_sum_max_error
                ),
                "labels_soft": generation_result.labels_soft_path.as_posix(),
                "labels_matrix": generation_result.labels_matrix_path.as_posix(),
            }
        ),
        "v1": v1_summary,
        "v2": v2_summary,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> SoftLabelExperimentResult:
    args = resolve_args(args)
    config = soft_label_config_from_mapping(load_config(args.config_path))

    generation_result = None
    if not args.skip_generation:
        generation_result = generate_soft_label_variant(
            source_labels_soft=args.v1_labels_soft,
            output_labels_soft=args.v2_labels_soft,
            output_labels_matrix=args.v2_labels_matrix,
            config=config,
            progress_every=args.progress_every,
        )
    elif not args.v2_labels_soft.exists() or not args.v2_labels_matrix.exists():
        raise FileNotFoundError(
            "v2 label files do not exist; remove --skip-generation or create them first"
        )

    v1_summary = summarize_soft_labels(
        labels_soft_path=args.v1_labels_soft,
        labels_matrix_path=args.v1_labels_matrix,
    )
    v2_summary = summarize_soft_labels(
        labels_soft_path=args.v2_labels_soft,
        labels_matrix_path=args.v2_labels_matrix,
    )
    preview_manifest = write_preview_images(
        preview_dir=args.preview_dir,
        manifest_path=args.preview_manifest,
        v1_labels_soft=args.v1_labels_soft,
        v2_labels_soft=args.v2_labels_soft,
        preview_count=args.preview_count,
        seed=args.seed,
    )
    write_v2_analysis_report(
        args.analysis_report_path,
        config=config,
        summary=v2_summary,
        preview_dir=args.preview_dir,
    )
    write_comparison_report(
        args.comparison_report_path,
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        preview_dir=args.preview_dir,
    )
    write_summary_json(
        args.summary_path,
        config=config,
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        generation_result=generation_result,
    )
    return SoftLabelExperimentResult(
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        generation_result=generation_result,
        summary_path=args.summary_path,
        analysis_report_path=args.analysis_report_path,
        comparison_report_path=args.comparison_report_path,
        preview_manifest_path=preview_manifest,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run(parse_args())
    LOGGER.info(
        json.dumps(
            {
                "summary_path": result.summary_path.as_posix(),
                "analysis_report_path": result.analysis_report_path.as_posix(),
                "comparison_report_path": result.comparison_report_path.as_posix(),
                "preview_manifest_path": (
                    result.preview_manifest_path.as_posix()
                    if result.preview_manifest_path is not None
                    else None
                ),
                "v2_top1_probability_mean": result.v2_summary[
                    "top1_probability_mean"
                ],
                "v2_entropy_normalized_mean": result.v2_summary[
                    "entropy_normalized_mean"
                ],
                "v2_top1_wcag_pass_rate": result.v2_summary[
                    "top1_wcag_pass_rate"
                ],
                "v2_max_color_share": result.v2_summary["max_color_share"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
