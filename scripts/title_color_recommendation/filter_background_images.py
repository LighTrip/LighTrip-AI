from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageStat

try:
    from scripts.title_color_recommendation.common import (
        IMAGE_SUFFIXES,
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        clear_output_dir,
        dhash,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_input_image_path,
        resolve_output_path,
        resolve_project_path,
        safe_child_path,
        safe_path_segment,
        sha256_file,
        utc_now,
        write_csv_rows,
        write_json_file,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        IMAGE_SUFFIXES,
        PROJECT_ROOT,
        TITLE_DATA_ROOT,
        clear_output_dir,
        dhash,
        ensure_output_dir,
        project_relative,
        read_csv_rows,
        resolve_input_image_path,
        resolve_output_path,
        resolve_project_path,
        safe_child_path,
        safe_path_segment,
        sha256_file,
        utc_now,
        write_csv_rows,
        write_json_file,
    )


Image.MAX_IMAGE_PIXELS = 40_000_000
warnings.simplefilter("error", Image.DecompressionBombWarning)

DEFAULT_RAW_DIR = PROJECT_ROOT / "data/title_color_recommendation/raw/places365"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/title_color_recommendation/processed"
QUALITY_FIELDS = [
    "clean_path",
    "quality_status",
    "rejection_reasons",
    "rejection_error",
    "clean_width",
    "clean_height",
    "clean_mode",
    "brightness",
    "sha256",
    "dhash",
    "duplicate_of",
    "filtered_at",
]


def default_input_metadata(raw_dir: Path) -> Path:
    metadata = raw_dir / "metadata.csv"
    if metadata.exists():
        return metadata
    checkpoint = raw_dir / "metadata_checkpoint.csv"
    if checkpoint.exists():
        return checkpoint
    return metadata


def scan_image_rows(raw_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows: list[dict[str, str]] = []
    for image_path in sorted(raw_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            category = image_path.relative_to(raw_dir).parts[0]
        except IndexError:
            category = "unknown"
        image_id = f"{category}_{image_path.stem}"
        rows.append(
            {
                "id": image_id,
                "image_path": project_relative(image_path),
                "label": category,
                "category_slug": category,
                "places365_slug": image_path.parent.name,
            }
        )
    return rows, ["id", "image_path", "label", "category_slug", "places365_slug"]


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_similar_hash(
    image_hash: str,
    seen_dhash: dict[str, str],
    threshold: int,
) -> str:
    if threshold <= 0:
        return seen_dhash.get(image_hash, "")

    for known_hash, known_id in seen_dhash.items():
        if hamming_distance(image_hash, known_hash) <= threshold:
            return known_id
    return ""


def image_brightness(image: Image.Image) -> float:
    gray = ImageOps.grayscale(image)
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0])


def category_for(row: dict[str, str]) -> str:
    return (
        row.get("category_slug")
        or row.get("category")
        or row.get("label")
        or "unknown"
    )


def clean_path_for(row: dict[str, str], clean_image_dir: Path) -> Path:
    category = safe_path_segment(category_for(row))
    image_id = row.get("id") or Path(row.get("image_path", "image")).stem
    return safe_child_path(
        clean_image_dir,
        category,
        f"{safe_path_segment(image_id, fallback='image')}.jpg",
    )


def raw_path_for(row: dict[str, str], raw_dir: Path) -> Path | None:
    image_path = row.get("image_path") or row.get("raw_path") or ""
    if not image_path:
        return None
    return resolve_input_image_path(image_path, raw_dir=raw_dir)


def inspect_row(
    row: dict[str, str],
    *,
    clean_image_dir: Path,
    seen_sha: dict[str, str],
    seen_dhash: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output: dict[str, Any] = dict(row)
    output["filtered_at"] = utc_now()
    reasons: list[str] = []

    raw_path: Path | None = None
    try:
        raw_path = raw_path_for(row, args.raw_dir)
    except ValueError as exc:
        reasons.append("unsafe_image_path")
        output["rejection_error"] = str(exc)

    if raw_path is None and not reasons:
        reasons.append("missing_image_path")
    elif not raw_path.exists():
        reasons.append("file_missing")
    elif not raw_path.is_file():
        reasons.append("not_a_file")
    elif raw_path.suffix.lower() not in IMAGE_SUFFIXES:
        reasons.append("unsupported_image_suffix")

    converted: Image.Image | None = None
    digest = ""

    if not reasons and raw_path is not None:
        try:
            digest = sha256_file(raw_path)
            output["sha256"] = digest
            duplicate_of = seen_sha.get(digest)
            if duplicate_of:
                reasons.append("exact_duplicate")
                output["duplicate_of"] = duplicate_of

            with Image.open(raw_path) as opened:
                opened.load()
                image = ImageOps.exif_transpose(opened)
                width, height = image.size
                output["clean_width"] = width
                output["clean_height"] = height
                output["clean_mode"] = image.mode

                if width < args.min_width:
                    reasons.append("too_small_width")
                if height < args.min_height:
                    reasons.append("too_small_height")

                converted = image.convert("RGB")
                output["clean_mode"] = "RGB"

            if converted is not None:
                brightness = image_brightness(converted)
                output["brightness"] = round(brightness, 2)
                if brightness < args.dark_threshold:
                    reasons.append("too_dark")
                if brightness > args.bright_threshold:
                    reasons.append("too_bright")

                image_hash = dhash(converted)
                output["dhash"] = image_hash
                similar_to = find_similar_hash(
                    image_hash,
                    seen_dhash,
                    threshold=args.perceptual_hash_threshold,
                )
                if similar_to:
                    reasons.append("perceptual_duplicate")
                    output["duplicate_of"] = similar_to
        except Exception as exc:
            reasons.append("unreadable_or_rgb_convert_failed")
            output["rejection_error"] = str(exc)

    if reasons:
        output["quality_status"] = "rejected"
        output["rejection_reasons"] = "|".join(reasons)
        return output

    if converted is None:
        output["quality_status"] = "rejected"
        output["rejection_reasons"] = "rgb_convert_failed"
        return output

    clean_path = clean_path_for(row, clean_image_dir)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    converted.save(clean_path, "JPEG", quality=args.jpeg_quality)

    image_id = str(row.get("id") or project_relative(clean_path))
    if digest:
        seen_sha[digest] = image_id
    if output.get("dhash"):
        seen_dhash[str(output["dhash"])] = image_id

    output["clean_path"] = project_relative(clean_path)
    output["quality_status"] = "accepted"
    output["rejection_reasons"] = ""
    return output


def summarize(
    *,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    args: argparse.Namespace,
    input_metadata: Path | None,
) -> dict[str, Any]:
    category_counts: Counter[str] = Counter()
    category_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()

    for row in accepted:
        category = category_for(row)
        category_counts[category] += 1
        category_status_counts[category]["accepted"] += 1

    for row in rejected:
        category = category_for(row)
        category_status_counts[category]["rejected"] += 1
        for reason in str(row.get("rejection_reasons", "")).split("|"):
            if reason:
                reason_counts[reason] += 1

    total = len(accepted) + len(rejected)
    return {
        "total_input_images": total,
        "accepted_clean_images": len(accepted),
        "rejected_images": len(rejected),
        "acceptance_rate": round(len(accepted) / total, 4) if total else 0.0,
        "roi_crop_ready_images": len(accepted),
        "category_clean_counts": dict(sorted(category_counts.items())),
        "category_status_counts": {
            category: dict(sorted(counts.items()))
            for category, counts in sorted(category_status_counts.items())
        },
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "input_metadata": project_relative(input_metadata) if input_metadata else "",
        "raw_dir": project_relative(args.raw_dir),
        "clean_image_dir": project_relative(args.clean_image_dir),
        "clean_metadata": project_relative(args.clean_metadata),
        "rejected_metadata": project_relative(args.rejected_metadata),
        "summary": project_relative(args.summary),
        "thresholds": {
            "min_width": args.min_width,
            "min_height": args.min_height,
            "dark_threshold": args.dark_threshold,
            "bright_threshold": args.bright_threshold,
            "perceptual_hash_threshold": args.perceptual_hash_threshold,
        },
        "generated_at": utc_now(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="title_color_recommendation raw 배경 이미지를 최소 품질 기준으로 필터링합니다."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--input-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clean-image-dir", type=Path, default=None)
    parser.add_argument("--clean-metadata", type=Path, default=None)
    parser.add_argument("--rejected-metadata", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--dark-threshold", type=float, default=10.0)
    parser.add_argument("--bright-threshold", type=float, default=245.0)
    parser.add_argument(
        "--perceptual-hash-threshold",
        type=int,
        default=0,
        help=(
            "dHash 중복 판정 거리입니다. 기본 0은 동일 dHash만 제거합니다. "
            "1 이상은 더 느릴 수 있습니다."
        ),
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--clear-output", action="store_true")
    parser.add_argument(
        "--scan-without-metadata",
        action="store_true",
        help="metadata CSV가 없을 때 raw directory를 직접 스캔합니다.",
    )
    return parser.parse_args()


def configure_paths(args: argparse.Namespace) -> None:
    read_roots = (TITLE_DATA_ROOT,)
    args.raw_dir = resolve_project_path(
        args.raw_dir,
        allowed_roots=read_roots,
        description="raw directory",
    )
    args.output_dir = resolve_output_path(
        args.output_dir,
        description="output directory",
    )
    args.clean_image_dir = resolve_output_path(
        args.clean_image_dir or args.output_dir / "clean_images",
        description="clean image directory",
    )
    args.clean_metadata = resolve_output_path(
        args.clean_metadata or args.output_dir / "clean_metadata.csv",
        description="clean metadata",
    )
    args.rejected_metadata = resolve_output_path(
        args.rejected_metadata or args.output_dir / "rejected_metadata.csv",
        description="rejected metadata",
    )
    args.summary = resolve_output_path(
        args.summary or args.output_dir / "quality_summary.json",
        description="summary",
    )
    if args.input_metadata is None:
        args.input_metadata = default_input_metadata(args.raw_dir)
    else:
        args.input_metadata = resolve_project_path(
            args.input_metadata,
            allowed_roots=read_roots,
            description="input metadata",
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.min_width <= 0 or args.min_height <= 0:
        raise ValueError("--min-width와 --min-height는 1 이상이어야 합니다.")
    if not 0 <= args.dark_threshold <= 255:
        raise ValueError("--dark-threshold는 0..255 범위여야 합니다.")
    if not 0 <= args.bright_threshold <= 255:
        raise ValueError("--bright-threshold는 0..255 범위여야 합니다.")
    if args.dark_threshold >= args.bright_threshold:
        raise ValueError("--dark-threshold는 --bright-threshold보다 작아야 합니다.")
    if args.perceptual_hash_threshold < 0:
        raise ValueError("--perceptual-hash-threshold는 0 이상이어야 합니다.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality는 1..100 범위여야 합니다.")
    if args.limit < 0:
        raise ValueError("--limit는 0 이상이어야 합니다.")
    if args.progress_every < 0:
        raise ValueError("--progress-every는 0 이상이어야 합니다.")


def main() -> None:
    args = parse_args()
    configure_paths(args)
    validate_args(args)

    if args.clear_output:
        clear_output_dir(args.clean_image_dir)
    else:
        ensure_output_dir(args.clean_image_dir)

    input_metadata: Path | None = args.input_metadata
    if input_metadata.exists():
        rows, input_fields = read_csv_rows(input_metadata)
    elif args.scan_without_metadata:
        rows, input_fields = scan_image_rows(args.raw_dir)
        input_metadata = None
    else:
        raise FileNotFoundError(
            f"metadata CSV를 찾을 수 없습니다: {args.input_metadata}. "
            "--scan-without-metadata를 사용하면 이미지 파일을 직접 스캔합니다."
        )

    if args.limit:
        rows = rows[: args.limit]

    output_fields = [
        *input_fields,
        *[field for field in QUALITY_FIELDS if field not in input_fields],
    ]
    seen_sha: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        reviewed = inspect_row(
            row,
            clean_image_dir=args.clean_image_dir,
            seen_sha=seen_sha,
            seen_dhash=seen_dhash,
            args=args,
        )
        if reviewed["quality_status"] == "accepted":
            accepted.append(reviewed)
        else:
            rejected.append(reviewed)

        if args.progress_every and index % args.progress_every == 0:
            print(
                f"[PROGRESS] {index}/{len(rows)} "
                f"accepted={len(accepted)} rejected={len(rejected)}"
            )

    write_csv_rows(args.clean_metadata, accepted, output_fields)
    write_csv_rows(args.rejected_metadata, rejected, output_fields)

    summary = summarize(
        accepted=accepted,
        rejected=rejected,
        args=args,
        input_metadata=input_metadata,
    )
    write_json_file(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
