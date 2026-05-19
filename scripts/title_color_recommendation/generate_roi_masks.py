#!/usr/bin/env python3
"""Generate native-size title ROI crops and matching text masks."""

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any

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
        safe_path_segment,
        utc_now,
        write_csv_rows,
        write_json_file,
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
        safe_path_segment,
        utc_now,
        write_csv_rows,
        write_json_file,
    )

bootstrap_project_imports()

from src.title_color_recommendation.data.roi_preprocessing import (
    ImageSize,
    RelativeROI,
    TitleROIResult,
    TitleSpec,
    image_size_from_config,
    prepare_title_roi,
    relative_roi_from_config,
    resampling_lanczos,
    resampling_nearest,
    split_to_crop_mode,
    stable_random,
    title_spec_from_config,
)


Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

DEFAULT_CONFIG = PROJECT_ROOT / "configs/title_color_recommendation/default.yaml"
DEFAULT_METADATA = TITLE_DATA_ROOT / "processed/clean_metadata.csv"
DEFAULT_SUMMARY = TITLE_OUTPUT_ROOT / "reports/roi_mask_summary.json"
ROI_METADATA_FIELDS = [
    "id",
    "split",
    "clean_path",
    "roi_path",
    "mask_path",
    "crop_mode",
    "original_width",
    "original_height",
    "resized_width",
    "resized_height",
    "input_width",
    "input_height",
    "crop_x1",
    "crop_y1",
    "crop_x2",
    "crop_y2",
    "roi_x1",
    "roi_y1",
    "roi_x2",
    "roi_y2",
    "roi_width",
    "roi_height",
    "mask_width",
    "mask_height",
    "title_center_roi_x",
    "title_center_roi_y",
    "processed_at",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


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


def split_for_row(row: dict[str, str], default_split: str) -> str:
    return (row.get("split") or default_split).strip().lower()


def clean_path_for_row(row: dict[str, str]) -> Path:
    value = row.get("clean_path") or ""
    if not value.strip():
        raise ValueError(f"clean_path가 비어 있습니다: id={row.get('id', '')}")
    return resolve_project_path(
        value,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="clean image",
        must_exist=True,
    )


def image_id_for_row(row: dict[str, str], clean_path: Path) -> str:
    return safe_path_segment(row.get("id") or clean_path.stem, fallback="image")


def choose_preview_ids(
    rows: list[dict[str, str]],
    *,
    count: int,
    seed: int,
) -> set[str]:
    if count <= 0:
        return set()
    keyed: list[tuple[float, str]] = []
    sampler = random.Random(seed)
    for row in rows:
        row_id = str(row.get("id") or row.get("clean_path") or "")
        if row_id:
            keyed.append((sampler.random(), row_id))
    keyed.sort()
    return {row_id for _, row_id in keyed[:count]}


def draw_label(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
) -> None:
    draw.text(xy, text, fill=(31, 41, 55))


def scale_panel(image: Image.Image, *, height: int, nearest: bool = False) -> Image.Image:
    if image.height == height:
        return image.copy()
    width = max(1, round(image.width * (height / image.height)))
    return image.resize(
        (width, height),
        resample=resampling_nearest() if nearest else resampling_lanczos(),
    )


def save_preview(
    path: Path,
    *,
    image_id: str,
    result: TitleROIResult,
) -> None:
    crop_panel = result.cropped_image.copy()
    crop_draw = ImageDraw.Draw(crop_panel)
    crop_draw.rectangle(result.roi_box, outline=(239, 68, 68), width=2)

    panel_height = result.cropped_image.height
    roi_panel = scale_panel(result.roi_image, height=panel_height)
    mask_panel = scale_panel(result.text_mask.convert("RGB"), height=panel_height, nearest=True)
    overlay = result.roi_image.convert("RGBA")
    red = Image.new("RGBA", overlay.size, (239, 68, 68, 0))
    red.putalpha(result.text_mask.point(lambda value: 120 if value else 0))
    overlay = Image.alpha_composite(overlay, red).convert("RGB")
    overlay_panel = scale_panel(overlay, height=panel_height)

    labels = ("crop", "roi", "mask", "overlay")
    panels = (crop_panel, roi_panel, mask_panel, overlay_panel)
    gap = 12
    margin = 14
    label_height = 22
    header_height = 26
    width = (margin * 2) + sum(panel.width for panel in panels) + (gap * (len(panels) - 1))
    height = margin + header_height + label_height + panel_height + margin
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), image_id, fill=(17, 24, 39))

    x = margin
    y = margin + header_height
    for label, panel in zip(labels, panels):
        draw_label(canvas, draw, (x, y), label)
        canvas.paste(panel, (x, y + label_height))
        x += panel.width + gap

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, "JPEG", quality=92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "clean image에서 195x280 화면 crop을 만든 뒤 config ROI를 native size로 "
            "저장하고 동일 크기 text mask를 생성합니다."
        )
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--roi-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--roi-metadata", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--preview-count", type=int, default=None)
    parser.add_argument("--split", default="")
    parser.add_argument(
        "--crop-mode",
        choices=("auto", "random", "center"),
        default="auto",
        help="auto는 train=random, 그 외 split=center를 사용합니다.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--text", default="")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument("--clear-preview", action="store_true")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> tuple[dict[str, Any], ImageSize, RelativeROI, TitleSpec]:
    args.config_path = resolve_config_path(args.config_path)
    config = load_config(args.config_path)
    preprocessing = config.get("preprocessing") or {}
    args.seed = int(args.seed if args.seed is not None else preprocessing.get("crop_seed", 42))
    args.default_split = str(preprocessing.get("default_split") or "train")
    args.preview_count = int(
        args.preview_count
        if args.preview_count is not None
        else preprocessing.get("preview_sample_count", 50)
    )

    args.metadata = resolve_project_path(
        args.metadata or DEFAULT_METADATA,
        allowed_roots=(TITLE_DATA_ROOT,),
        description="metadata",
        must_exist=True,
    )
    args.roi_dir = path_from_config(
        config,
        "roi_dir",
        default=TITLE_DATA_ROOT / "processed/rois",
        output=True,
        description="ROI output directory",
    ) if args.roi_dir is None else resolve_output_path(args.roi_dir, description="ROI output directory")
    args.mask_dir = path_from_config(
        config,
        "mask_dir",
        default=TITLE_DATA_ROOT / "processed/masks",
        output=True,
        description="mask output directory",
    ) if args.mask_dir is None else resolve_output_path(args.mask_dir, description="mask output directory")
    args.roi_metadata = resolve_output_path(
        args.roi_metadata or TITLE_DATA_ROOT / "processed/roi_metadata.csv",
        description="ROI metadata",
    )
    args.summary = resolve_output_path(args.summary, description="summary")
    preview_root = path_from_config(
        config,
        "preview_dir",
        default=TITLE_OUTPUT_ROOT / "previews",
        output=True,
        description="preview directory",
    )
    args.preview_dir = resolve_output_path(
        args.preview_dir or preview_root / "roi_mask_samples",
        description="preview directory",
    )

    input_size = image_size_from_config(config["input_size"])
    roi = relative_roi_from_config(config["roi"])
    title = title_spec_from_config(config["title"])
    if args.text:
        title = TitleSpec(
            center_x=title.center_x,
            center_y=title.center_y,
            font_size=title.font_size,
            text=args.text,
            font_path=title.font_path,
        )
    return config, input_size, roi, title


def validate_args(args: argparse.Namespace) -> None:
    if args.preview_count < 0:
        raise ValueError("--preview-count는 0 이상이어야 합니다.")
    if args.limit < 0:
        raise ValueError("--limit는 0 이상이어야 합니다.")
    if args.progress_every < 0:
        raise ValueError("--progress-every는 0 이상이어야 합니다.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality는 1..100 범위여야 합니다.")


def main() -> None:
    args = parse_args()
    _, input_size, roi, title = configure(args)
    validate_args(args)

    if args.clear_output:
        clear_output_dir(args.roi_dir)
        clear_output_dir(args.mask_dir)
    else:
        ensure_output_dir(args.roi_dir)
        ensure_output_dir(args.mask_dir)

    if args.clear_preview:
        clear_output_dir(args.preview_dir)
    else:
        ensure_output_dir(args.preview_dir)

    rows, _ = read_csv_rows(args.metadata)
    if args.split:
        wanted = args.split.strip().lower()
        rows = [
            row
            for row in rows
            if split_for_row(row, args.default_split) == wanted
        ]
    if args.limit:
        rows = rows[: args.limit]

    preview_ids = choose_preview_ids(rows, count=args.preview_count, seed=args.seed)
    metadata_rows: list[dict[str, Any]] = []
    roi_size_counts: dict[str, int] = {}
    mask_size_counts: dict[str, int] = {}
    crop_mode_counts: dict[str, int] = {}
    skipped = 0
    preview_saved = 0

    for index, row in enumerate(rows, start=1):
        clean_path: Path | None = None
        try:
            clean_path = clean_path_for_row(row)
            image_id = image_id_for_row(row, clean_path)
            split = split_for_row(row, args.default_split)
            crop_mode = split_to_crop_mode(split) if args.crop_mode == "auto" else args.crop_mode
            rng = stable_random(args.seed, f"{split}:{image_id}") if crop_mode == "random" else None

            with Image.open(clean_path) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened).convert("RGB")
                original_width, original_height = image.size
                result = prepare_title_roi(
                    image,
                    input_size=input_size,
                    roi=roi,
                    title=title,
                    crop_mode=crop_mode,
                    rng=rng,
                )

            roi_path = safe_child_path(args.roi_dir, f"{image_id}.jpg")
            mask_path = safe_child_path(args.mask_dir, f"{image_id}.png")
            result.roi_image.save(roi_path, "JPEG", quality=args.jpeg_quality)
            result.text_mask.save(mask_path, "PNG")

            if (row.get("id") or "") in preview_ids:
                preview_path = safe_child_path(args.preview_dir, f"{image_id}_preview.jpg")
                save_preview(preview_path, image_id=image_id, result=result)
                preview_saved += 1

            roi_width, roi_height = result.roi_image.size
            mask_width, mask_height = result.text_mask.size
            roi_size_counts[f"{roi_width}x{roi_height}"] = (
                roi_size_counts.get(f"{roi_width}x{roi_height}", 0) + 1
            )
            mask_size_counts[f"{mask_width}x{mask_height}"] = (
                mask_size_counts.get(f"{mask_width}x{mask_height}", 0) + 1
            )
            crop_mode_counts[crop_mode] = crop_mode_counts.get(crop_mode, 0) + 1

            crop_x1, crop_y1, crop_x2, crop_y2 = result.crop_box
            roi_x1, roi_y1, roi_x2, roi_y2 = result.roi_box
            metadata_rows.append(
                {
                    "id": image_id,
                    "split": split,
                    "clean_path": project_relative(clean_path),
                    "roi_path": project_relative(roi_path),
                    "mask_path": project_relative(mask_path),
                    "crop_mode": crop_mode,
                    "original_width": original_width,
                    "original_height": original_height,
                    "resized_width": result.resized_size.width,
                    "resized_height": result.resized_size.height,
                    "input_width": input_size.width,
                    "input_height": input_size.height,
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "roi_x1": roi_x1,
                    "roi_y1": roi_y1,
                    "roi_x2": roi_x2,
                    "roi_y2": roi_y2,
                    "roi_width": roi_width,
                    "roi_height": roi_height,
                    "mask_width": mask_width,
                    "mask_height": mask_height,
                    "title_center_roi_x": round(result.title_center_in_roi[0], 3),
                    "title_center_roi_y": round(result.title_center_in_roi[1], 3),
                    "processed_at": utc_now(),
                }
            )
        except Exception as exc:
            skipped += 1
            print(
                "[WARN] ROI/mask 생성 실패: "
                f"row={index} clean_path={project_relative(clean_path) if clean_path else ''} "
                f"error={exc}"
            )

        if args.progress_every and index % args.progress_every == 0:
            print(
                f"[PROGRESS] {index}/{len(rows)} "
                f"processed={len(metadata_rows)} skipped={skipped}"
            )

    write_csv_rows(args.roi_metadata, metadata_rows, ROI_METADATA_FIELDS)
    summary = {
        "metadata": project_relative(args.metadata),
        "roi_metadata": project_relative(args.roi_metadata),
        "roi_dir": project_relative(args.roi_dir),
        "mask_dir": project_relative(args.mask_dir),
        "preview_dir": project_relative(args.preview_dir),
        "input_size": {"width": input_size.width, "height": input_size.height},
        "roi": {"x1": roi.x1, "y1": roi.y1, "x2": roi.x2, "y2": roi.y2},
        "title": {
            "center_x": title.center_x,
            "center_y": title.center_y,
            "font_size": title.font_size,
            "text": title.text,
            "font_path": title.font_path,
        },
        "total_rows": len(rows),
        "processed": len(metadata_rows),
        "skipped": skipped,
        "preview_saved": preview_saved,
        "roi_size_counts": dict(sorted(roi_size_counts.items())),
        "mask_size_counts": dict(sorted(mask_size_counts.items())),
        "crop_mode_counts": dict(sorted(crop_mode_counts.items())),
        "generated_at": utc_now(),
    }
    write_json_file(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
