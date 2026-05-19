#!/usr/bin/env python3
"""Generate pseudo-scores and soft labels for title color recommendation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

try:
    from scripts.title_color_recommendation.common import (
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        TITLE_OUTPUT_ROOT,
        bootstrap_project_imports,
        clear_output_dir,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_config_path,
        resolve_output_path,
        resolve_project_path,
        safe_child_path,
        utc_now,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        TITLE_OUTPUT_ROOT,
        bootstrap_project_imports,
        clear_output_dir,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_config_path,
        resolve_output_path,
        resolve_project_path,
        safe_child_path,
        utc_now,
    )

bootstrap_project_imports()

from src.title_color_recommendation.labeling.soft_labels import (
    ImageSoftLabelResult,
    PaletteColor,
    SoftLabelConfig,
    compute_image_soft_labels,
    load_palette,
    normalized_distribution_entropy,
    soft_label_config_from_mapping,
    softmax,
)
from src.title_color_recommendation.data.roi_preprocessing import resampling_lanczos


DEFAULT_CONFIG = PROJECT_ROOT / "configs/title_color_recommendation/default.yaml"
DEFAULT_ROI_METADATA = TITLE_DATA_ROOT / "processed/roi_metadata.csv"
DEFAULT_PALETTE = TITLE_DATA_ROOT / "processed/palette.json"
DEFAULT_LABEL_DIR = TITLE_DATA_ROOT / "processed/labels"
DEFAULT_SUMMARY = TITLE_OUTPUT_ROOT / "reports/soft_label_summary.json"
DEFAULT_PREVIEW_DIR = TITLE_OUTPUT_ROOT / "previews/soft_label_samples"

CONTRAST_FIELDS = [
    "id",
    "split",
    "roi_path",
    "mask_path",
    "palette_id",
    "color_name",
    "color_hex",
    "color_group",
    "contrast_p05",
    "contrast_p10",
    "contrast_mean",
    "wcag_pass",
    "background_luminance_mean",
    "background_luminance_std",
    "background_luminance_p05",
    "background_luminance_p95",
    "background_chroma",
    "background_colorfulness",
    "background_complexity",
    "text_pixel_count",
    "roi_pixel_count",
]

LABEL_FIELDS = [
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
    "contrast_p05",
    "contrast_p10",
    "contrast_mean",
    "wcag_pass",
]

INDEX_FIELDS = [
    "matrix_index",
    "id",
    "split",
    "roi_path",
    "mask_path",
    "top1_palette_id",
    "top1_color_name",
    "top1_color_hex",
    "top1_color_group",
    "top1_probability",
    "top1_wcag_pass",
    "background_brightness_bin",
    "background_luminance_mean",
    "entropy_normalized",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def parse_temperature_list(value: str) -> list[float]:
    temperatures: list[float] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        temperature = float(raw)
        if temperature <= 0:
            raise ValueError(f"temperature must be positive: {temperature}")
        temperatures.append(temperature)
    if not temperatures:
        raise ValueError("at least one temperature is required")
    return temperatures


def temperature_suffix(temperature: float) -> str:
    return f"t{temperature:g}".replace(".", "_")


def format_float(value: float, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROI/mask와 32색 palette에서 contrast feature, pseudo-score, soft label을 생성합니다."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--roi-metadata", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=None)
    parser.add_argument("--label-dir", type=Path, default=None)
    parser.add_argument("--contrast-features", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-index", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-count", type=int, default=None)
    parser.add_argument("--temperatures", default="")
    parser.add_argument("--include-comparison-temperatures", action="store_true")
    parser.add_argument("--split", default="")
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--clear-preview", action="store_true")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> tuple[dict[str, Any], SoftLabelConfig, list[float]]:
    args.config_path = resolve_config_path(args.config_path)
    config = load_config(args.config_path)
    labeling = config.get("labeling") or {}
    preprocessing = config.get("preprocessing") or {}

    label_dir = args.label_dir or (config.get("paths") or {}).get("label_dir") or DEFAULT_LABEL_DIR
    args.label_dir = resolve_output_path(label_dir, description="label directory")
    args.roi_metadata = resolve_project_path(
        args.roi_metadata or DEFAULT_ROI_METADATA,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="ROI metadata",
        must_exist=True,
    )
    args.palette = resolve_project_path(
        args.palette or DEFAULT_PALETTE,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="palette",
        must_exist=True,
    )
    args.contrast_features = resolve_output_path(
        args.contrast_features or args.label_dir / "contrast_features.csv",
        description="contrast features",
    )
    args.labels_soft = resolve_output_path(
        args.labels_soft or args.label_dir / "labels_soft.csv",
        description="soft label csv",
    )
    args.labels_matrix = resolve_output_path(
        args.labels_matrix or args.label_dir / "labels_matrix.npy",
        description="soft label matrix",
    )
    args.labels_index = resolve_output_path(
        args.labels_index or args.label_dir / "labels_index.csv",
        description="soft label matrix index",
    )
    args.summary = resolve_output_path(args.summary, description="summary")

    preview_root = (config.get("paths") or {}).get("preview_dir") or TITLE_OUTPUT_ROOT / "previews"
    args.preview_dir = resolve_output_path(
        args.preview_dir or Path(preview_root) / "soft_label_samples",
        description="preview directory",
    )

    args.seed = int(args.seed if args.seed is not None else preprocessing.get("crop_seed", 42))
    args.preview_count = int(
        args.preview_count
        if args.preview_count is not None
        else labeling.get("preview_sample_count", 50)
    )

    soft_config = soft_label_config_from_mapping(config)
    if args.temperatures:
        temperatures = parse_temperature_list(args.temperatures)
    else:
        temperatures = [soft_config.temperature]
        if args.include_comparison_temperatures:
            temperatures.extend(float(value) for value in labeling.get("comparison_temperatures", []))

    deduped: list[float] = []
    for temperature in temperatures:
        if temperature <= 0:
            raise ValueError(f"temperature must be positive: {temperature}")
        if temperature not in deduped:
            deduped.append(temperature)
    return config, soft_config, deduped


def validate_args(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("--limit는 0 이상이어야 합니다.")
    if args.sample_count < 0:
        raise ValueError("--sample-count는 0 이상이어야 합니다.")
    if args.preview_count < 0:
        raise ValueError("--preview-count는 0 이상이어야 합니다.")
    if args.progress_every < 0:
        raise ValueError("--progress-every는 0 이상이어야 합니다.")


def row_path(row: dict[str, str], key: str, *, description: str) -> Path:
    value = row.get(key) or ""
    if not value.strip():
        raise ValueError(f"{key}가 비어 있습니다: id={row.get('id', '')}")
    return resolve_project_path(
        value,
        allowed_roots=(TITLE_DATA_ROOT,),
        description=description,
        must_exist=True,
    )


def brightness_bin(mean_luminance: float) -> str:
    if mean_luminance < 0.33:
        return "dark"
    if mean_luminance > 0.67:
        return "bright"
    return "mid"


def neutral_bias_group(color: PaletteColor) -> bool:
    return color.group in {"neutral_light", "neutral_dark", "muted"}


def open_csv_writer(
    stack: ExitStack,
    path: Path,
    fieldnames: list[str],
) -> csv.DictWriter:
    if path.exists() and path.is_symlink():
        raise ValueError(f"심볼릭 링크 출력 파일은 허용하지 않습니다: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    file = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
    writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return writer


def contrast_row(
    *,
    row: dict[str, str],
    roi_path: Path,
    mask_path: Path,
    color: PaletteColor,
    result: ImageSoftLabelResult,
    index: int,
) -> dict[str, Any]:
    background = result.background
    return {
        "id": row.get("id", ""),
        "split": row.get("split", ""),
        "roi_path": project_relative(roi_path),
        "mask_path": project_relative(mask_path),
        "palette_id": color.id,
        "color_name": color.name,
        "color_hex": color.hex,
        "color_group": color.group,
        "contrast_p05": format_float(result.contrast_p05[index]),
        "contrast_p10": format_float(result.contrast_p10[index]),
        "contrast_mean": format_float(result.contrast_mean[index]),
        "wcag_pass": int(bool(result.wcag_pass[index])),
        "background_luminance_mean": format_float(background.mean_luminance),
        "background_luminance_std": format_float(background.std_luminance),
        "background_luminance_p05": format_float(background.p05_luminance),
        "background_luminance_p95": format_float(background.p95_luminance),
        "background_chroma": format_float(background.chroma),
        "background_colorfulness": format_float(background.colorfulness),
        "background_complexity": format_float(background.complexity),
        "text_pixel_count": background.text_pixel_count,
        "roi_pixel_count": background.roi_pixel_count,
    }


def label_row(
    *,
    row: dict[str, str],
    color: PaletteColor,
    result: ImageSoftLabelResult,
    probabilities: np.ndarray,
    ranks: np.ndarray,
    index: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "id": row.get("id", ""),
        "split": row.get("split", ""),
        "palette_id": color.id,
        "color_name": color.name,
        "color_hex": color.hex,
        "color_group": color.group,
        "readability_score": format_float(result.readability_score[index]),
        "aesthetic_prior": format_float(result.aesthetic_prior[index]),
        "tone_match_score": format_float(result.tone_match_score[index]),
        "simplicity_score": format_float(result.simplicity_score[index]),
        "fail_penalty": format_float(result.fail_penalty[index]),
        "pseudo_score": format_float(result.pseudo_score[index]),
        "target_probability": format_float(probabilities[index], digits=8),
        "temperature": format_float(temperature, digits=3),
        "rank": int(ranks[index]),
        "contrast_p05": format_float(result.contrast_p05[index]),
        "contrast_p10": format_float(result.contrast_p10[index]),
        "contrast_mean": format_float(result.contrast_mean[index]),
        "wcag_pass": int(bool(result.wcag_pass[index])),
    }


def overlay_mask_text(roi_image: Image.Image, mask_image: Image.Image, color: PaletteColor) -> Image.Image:
    base = roi_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, tuple(color.rgb) + (0,))
    overlay.putalpha(mask_image.convert("L"))
    return Image.alpha_composite(base, overlay).convert("RGB")


def scale_panel(image: Image.Image, *, height: int) -> Image.Image:
    if image.height == height:
        return image.copy()
    width = max(1, round(image.width * (height / image.height)))
    return image.resize((width, height), resample=resampling_lanczos())


def save_preview(
    path: Path,
    *,
    image_id: str,
    roi_image: Image.Image,
    mask_image: Image.Image,
    palette: list[PaletteColor],
    probabilities: np.ndarray,
    result: ImageSoftLabelResult,
) -> None:
    top_indices = list(np.argsort(-probabilities)[:3])
    original = roi_image.convert("RGB")
    panels = [original]
    labels = ["original"]
    for rank, color_index in enumerate(top_indices, start=1):
        panels.append(overlay_mask_text(roi_image, mask_image, palette[color_index]))
        labels.append(
            f"top{rank} {palette[color_index].name} {probabilities[color_index]:.3f}"
        )

    panel_height = 144
    scaled = [scale_panel(panel, height=panel_height) for panel in panels]
    gap = 12
    margin = 14
    header_height = 26
    label_height = 24
    footer_height = 24
    width = (margin * 2) + sum(panel.width for panel in scaled) + (gap * (len(scaled) - 1))
    height = margin + header_height + label_height + panel_height + footer_height + margin
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), image_id, fill=(17, 24, 39))

    x = margin
    y = margin + header_height
    for label, panel in zip(labels, scaled):
        draw.text((x, y), label, fill=(31, 41, 55))
        canvas.paste(panel, (x, y + label_height))
        x += panel.width + gap

    footer = (
        f"bg={brightness_bin(result.background.mean_luminance)} "
        f"L={result.background.mean_luminance:.3f} "
        f"complexity={result.background.complexity:.3f}"
    )
    draw.text((margin, height - margin - 16), footer, fill=(75, 85, 99))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=92)


def candidate_hash(seed: int, image_id: str) -> str:
    return hashlib.sha256(f"{seed}:soft-label-preview:{image_id}".encode("utf-8")).hexdigest()


def row_sample_hash(seed: int, row: dict[str, str]) -> str:
    key = row.get("id") or row.get("roi_path") or row.get("mask_path") or ""
    return hashlib.sha256(f"{seed}:soft-label-row:{key}".encode("utf-8")).hexdigest()


def maybe_add_preview_candidate(
    candidates: dict[str, list[tuple[str, dict[str, Any]]]],
    *,
    bucket: str,
    max_per_bucket: int,
    digest: str,
    payload: dict[str, Any],
) -> None:
    items = candidates[bucket]
    if len(items) >= max_per_bucket and digest >= items[-1][0]:
        return
    items.append((digest, payload))
    items.sort(key=lambda item: item[0])
    del items[max_per_bucket:]


def write_preview_candidates(
    candidates: dict[str, list[tuple[str, dict[str, Any]]]],
    *,
    preview_dir: Path,
    preview_count: int,
    palette: list[PaletteColor],
) -> int:
    selected: list[tuple[str, dict[str, Any]]] = []
    bucket_order = ["dark", "mid", "bright"]
    for bucket in bucket_order:
        selected.extend(candidates.get(bucket, []))
    selected.sort(key=lambda item: item[0])
    selected = selected[:preview_count]

    saved = 0
    for _, payload in selected:
        path = safe_child_path(preview_dir, f"{payload['image_id']}_soft_label.jpg")
        save_preview(path, palette=palette, **payload)
        saved += 1
    return saved


def summarize_counter(counter: Counter[int], palette: list[PaletteColor]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total = sum(counter.values()) or 1
    for color in palette:
        count = counter.get(color.id, 0)
        rows.append(
            {
                "palette_id": color.id,
                "name": color.name,
                "hex": color.hex,
                "group": color.group,
                "count": count,
                "ratio": round(count / total, 6),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    _, soft_config, temperatures = configure(args)
    validate_args(args)

    ensure_output_dir(args.label_dir)
    if args.clear_preview:
        clear_output_dir(args.preview_dir)
    else:
        ensure_output_dir(args.preview_dir)

    rows, _ = read_csv_rows(args.roi_metadata)
    if args.split:
        wanted = args.split.strip().lower()
        rows = [row for row in rows if (row.get("split") or "").strip().lower() == wanted]
    if args.sample_count:
        rows = sorted(rows, key=lambda row: row_sample_hash(args.seed, row))[: args.sample_count]
    if args.limit:
        rows = rows[: args.limit]

    palette = load_palette(args.palette)
    primary_temperature = temperatures[0]
    writers: dict[float, csv.DictWriter] = {}
    matrix_rows: dict[float, list[np.ndarray]] = {temperature: [] for temperature in temperatures}
    entropy_values: dict[float, list[float]] = {temperature: [] for temperature in temperatures}
    max_probability_values: dict[float, list[float]] = {temperature: [] for temperature in temperatures}

    index_rows: list[dict[str, Any]] = []
    top1_color_counts: Counter[int] = Counter()
    top1_group_counts: Counter[str] = Counter()
    top1_brightness_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    top1_fail_count = 0
    images_with_any_wcag_pass = 0
    top1_fail_when_pass_available_count = 0
    top1_neutral_count = 0
    probability_sum_max_error = 0.0
    skipped = 0
    preview_candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    max_preview_per_bucket = max(1, math.ceil(max(args.preview_count, 1) / 3) * 2)

    with ExitStack() as stack:
        contrast_writer = open_csv_writer(stack, args.contrast_features, CONTRAST_FIELDS)
        for temperature in temperatures:
            label_path = args.labels_soft
            if temperature != primary_temperature:
                label_path = args.label_dir / f"labels_soft_{temperature_suffix(temperature)}.csv"
            writers[temperature] = open_csv_writer(
                stack,
                resolve_output_path(label_path, description="soft label csv"),
                LABEL_FIELDS,
            )

        for row_index, row in enumerate(rows, start=1):
            roi_path: Path | None = None
            mask_path: Path | None = None
            try:
                image_id = row.get("id") or f"row_{row_index}"
                roi_path = row_path(row, "roi_path", description="ROI image")
                mask_path = row_path(row, "mask_path", description="mask image")
                with Image.open(roi_path) as opened_roi, Image.open(mask_path) as opened_mask:
                    opened_roi.load()
                    opened_mask.load()
                    roi_image = ImageOps.exif_transpose(opened_roi).convert("RGB")
                    mask_image = opened_mask.convert("L")

                result = compute_image_soft_labels(
                    roi_image,
                    mask_image,
                    palette,
                    soft_config,
                    temperature=primary_temperature,
                )
                ranks = np.empty(len(palette), dtype=np.int32)
                ranks[np.argsort(-result.target_distribution)] = np.arange(1, len(palette) + 1)

                for color_index, color in enumerate(palette):
                    contrast_writer.writerow(
                        contrast_row(
                            row=row,
                            roi_path=roi_path,
                            mask_path=mask_path,
                            color=color,
                            result=result,
                            index=color_index,
                        )
                    )

                probability_by_temperature: dict[float, np.ndarray] = {}
                for temperature in temperatures:
                    probabilities = (
                        result.target_distribution
                        if temperature == primary_temperature
                        else softmax(result.pseudo_score, temperature=temperature)
                    )
                    probability_by_temperature[temperature] = probabilities
                    matrix_rows[temperature].append(probabilities.astype(np.float32))
                    entropy = normalized_distribution_entropy(probabilities)
                    entropy_values[temperature].append(entropy)
                    max_probability_values[temperature].append(float(np.max(probabilities)))
                    probability_sum_max_error = max(
                        probability_sum_max_error,
                        abs(float(np.sum(probabilities)) - 1.0),
                    )
                    temperature_ranks = np.empty(len(palette), dtype=np.int32)
                    temperature_ranks[np.argsort(-probabilities)] = np.arange(1, len(palette) + 1)
                    writer = writers[temperature]
                    for color_index, color in enumerate(palette):
                        writer.writerow(
                            label_row(
                                row=row,
                                color=color,
                                result=result,
                                probabilities=probabilities,
                                ranks=temperature_ranks,
                                index=color_index,
                                temperature=temperature,
                            )
                        )

                primary_probabilities = probability_by_temperature[primary_temperature]
                top1_index = int(np.argmax(primary_probabilities))
                top1_color = palette[top1_index]
                bucket = brightness_bin(result.background.mean_luminance)
                top1_color_counts[top1_color.id] += 1
                top1_group_counts[top1_color.group] += 1
                top1_brightness_group_counts[bucket][top1_color.group] += 1
                has_any_wcag_pass = bool(np.any(result.wcag_pass))
                top1_failed = not bool(result.wcag_pass[top1_index])
                images_with_any_wcag_pass += int(has_any_wcag_pass)
                top1_fail_count += int(top1_failed)
                top1_fail_when_pass_available_count += int(has_any_wcag_pass and top1_failed)
                top1_neutral_count += int(neutral_bias_group(top1_color))
                matrix_index = len(index_rows)
                index_rows.append(
                    {
                        "matrix_index": matrix_index,
                        "id": image_id,
                        "split": row.get("split", ""),
                        "roi_path": project_relative(roi_path),
                        "mask_path": project_relative(mask_path),
                        "top1_palette_id": top1_color.id,
                        "top1_color_name": top1_color.name,
                        "top1_color_hex": top1_color.hex,
                        "top1_color_group": top1_color.group,
                        "top1_probability": format_float(primary_probabilities[top1_index], digits=8),
                        "top1_wcag_pass": int(bool(result.wcag_pass[top1_index])),
                        "background_brightness_bin": bucket,
                        "background_luminance_mean": format_float(result.background.mean_luminance),
                        "entropy_normalized": format_float(
                            normalized_distribution_entropy(primary_probabilities)
                        ),
                    }
                )

                if args.preview_count > 0:
                    digest = candidate_hash(args.seed, image_id)
                    if (
                        len(preview_candidates[bucket]) < max_preview_per_bucket
                        or digest < preview_candidates[bucket][-1][0]
                    ):
                        maybe_add_preview_candidate(
                            preview_candidates,
                            bucket=bucket,
                            max_per_bucket=max_preview_per_bucket,
                            digest=digest,
                            payload={
                                "image_id": image_id,
                                "roi_image": roi_image.copy(),
                                "mask_image": mask_image.copy(),
                                "probabilities": primary_probabilities.copy(),
                                "result": result,
                            },
                        )
            except (OSError, ValueError, KeyError) as exc:
                skipped += 1
                print(
                    "[WARN] soft label 생성 실패: "
                    f"row={row_index} roi_path={project_relative(roi_path) if roi_path else ''} "
                    f"mask_path={project_relative(mask_path) if mask_path else ''} error={exc}"
                )

            if args.progress_every and row_index % args.progress_every == 0:
                print(
                    f"[PROGRESS] {row_index}/{len(rows)} "
                    f"processed={len(index_rows)} skipped={skipped}"
                )

    if matrix_rows[primary_temperature]:
        primary_matrix = np.vstack(matrix_rows[primary_temperature]).astype(np.float32)
    else:
        primary_matrix = np.empty((0, len(palette)), dtype=np.float32)
    args.labels_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.labels_matrix, primary_matrix)

    for temperature in temperatures:
        if temperature == primary_temperature:
            continue
        matrix = (
            np.vstack(matrix_rows[temperature]).astype(np.float32)
            if matrix_rows[temperature]
            else np.empty((0, len(palette)), dtype=np.float32)
        )
        np.save(args.label_dir / f"labels_matrix_{temperature_suffix(temperature)}.npy", matrix)

    with args.labels_index.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in index_rows:
            writer.writerow(row)

    preview_saved = write_preview_candidates(
        preview_candidates,
        preview_dir=args.preview_dir,
        preview_count=args.preview_count,
        palette=palette,
    )

    processed = len(index_rows)
    processed_or_one = processed or 1
    temperature_summary = {}
    for temperature in temperatures:
        entropy = entropy_values[temperature]
        max_probs = max_probability_values[temperature]
        temperature_summary[str(temperature)] = {
            "entropy_normalized_mean": round(float(np.mean(entropy)), 6) if entropy else 0.0,
            "entropy_normalized_p05": round(float(np.percentile(entropy, 5)), 6) if entropy else 0.0,
            "max_probability_mean": round(float(np.mean(max_probs)), 6) if max_probs else 0.0,
            "max_probability_p95": round(float(np.percentile(max_probs, 95)), 6) if max_probs else 0.0,
        }

    summary = {
        "roi_metadata": project_relative(args.roi_metadata),
        "palette": project_relative(args.palette),
        "contrast_features": project_relative(args.contrast_features),
        "labels_soft": project_relative(args.labels_soft),
        "labels_matrix": project_relative(args.labels_matrix),
        "labels_index": project_relative(args.labels_index),
        "preview_dir": project_relative(args.preview_dir),
        "temperatures": temperatures,
        "primary_temperature": primary_temperature,
        "total_rows": len(rows),
        "processed": processed,
        "skipped": skipped,
        "palette_size": len(palette),
        "matrix_shape": list(primary_matrix.shape),
        "probability_sum_max_error": probability_sum_max_error,
        "top1_wcag_fail_count": top1_fail_count,
        "top1_wcag_fail_ratio": round(top1_fail_count / processed_or_one, 6),
        "images_with_any_wcag_pass": images_with_any_wcag_pass,
        "images_with_any_wcag_pass_ratio": round(images_with_any_wcag_pass / processed_or_one, 6),
        "top1_wcag_fail_when_pass_available_count": top1_fail_when_pass_available_count,
        "top1_wcag_fail_when_pass_available_ratio": round(
            top1_fail_when_pass_available_count / (images_with_any_wcag_pass or 1),
            6,
        ),
        "top1_white_black_gray_count": top1_neutral_count,
        "top1_white_black_gray_ratio": round(top1_neutral_count / processed_or_one, 6),
        "top1_color_distribution": summarize_counter(top1_color_counts, palette),
        "top1_group_distribution": {
            group: {
                "count": count,
                "ratio": round(count / processed_or_one, 6),
            }
            for group, count in sorted(top1_group_counts.items())
        },
        "background_brightness_group_distribution": {
            bucket: {
                group: {
                    "count": count,
                    "ratio": round(count / (sum(counter.values()) or 1), 6),
                }
                for group, count in sorted(counter.items())
            }
            for bucket, counter in sorted(top1_brightness_group_counts.items())
        },
        "temperature_summary": temperature_summary,
        "preview_saved": preview_saved,
        "generated_at": utc_now(),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
