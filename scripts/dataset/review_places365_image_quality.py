from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageStat

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import write_jsonl


DEFAULT_METADATA = Path("data_places365_2/metadata.csv")
DEFAULT_MAPPING = Path("configs/places365_category_mapping_v2.json")
DEFAULT_OUTPUT_DIR = Path("data_places365_2/quality")
TRAINABLE_DECISIONS = {"keep", "keep_sample_review"}


def normalize_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def load_mapping(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    by_slug: dict[str, dict[str, Any]] = {}
    for category in payload.get("categories", []):
        for places in category.get("places365", []):
            slug = normalize_slug(places.get("slug") or places.get("label"))
            if not slug:
                continue
            by_slug[slug] = {
                "decision": str(places.get("decision", "")),
                "category_label": str(category.get("label", "")),
                "category_slug": str(category.get("slug", "")),
                "places365_label": str(places.get("label", "")),
            }

    for places in payload.get("excluded_labels", []):
        slug = normalize_slug(places.get("slug") or places.get("label"))
        if not slug:
            continue
        by_slug[slug] = {
            "decision": "exclude",
            "category_label": "",
            "category_slug": "",
            "places365_label": str(places.get("label", "")),
        }
    return by_slug


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    gray = ImageOps.grayscale(image)
    resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    bits: list[str] = []
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append("1" if pixels[offset + col] > pixels[offset + col + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def image_stats(image: Image.Image) -> tuple[float, float]:
    gray = ImageOps.grayscale(image)
    stat = ImageStat.Stat(gray)
    return float(stat.mean[0]), float(stat.stddev[0])


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "id",
        "image_path",
        "label",
        "category_slug",
        "places365_id",
        "places365_label",
        "places365_slug",
        "split",
    ]
    return {field: row.get(field, "") for field in fields}


def review_image_row(
    row: dict[str, str],
    *,
    mapping_by_slug: dict[str, dict[str, Any]],
    seen_sha: dict[str, str],
    seen_dhash: dict[str, str],
    min_dimension: int,
    min_aspect_ratio: float,
    max_aspect_ratio: float,
    dark_threshold: float,
    bright_threshold: float,
    low_contrast_threshold: float,
) -> dict[str, Any]:
    output = compact_row(row)
    image_path = Path(row.get("image_path", ""))
    slug = normalize_slug(row.get("places365_slug"))
    mapped = mapping_by_slug.get(slug)
    mapping_decision = mapped["decision"] if mapped is not None else "unmapped"

    hard_reasons: list[str] = []
    review_flags: list[str] = []
    metadata: dict[str, Any] = {
        "mapping_decision": mapping_decision,
    }

    if mapped is not None:
        metadata["mapping_category"] = mapped["category_label"]
        metadata["mapping_category_slug"] = mapped["category_slug"]

    if mapping_decision not in TRAINABLE_DECISIONS:
        hard_reasons.append(f"mapping_{mapping_decision}")
    elif mapping_decision == "keep_sample_review":
        review_flags.append("ambiguous_label")

    if not image_path.exists():
        hard_reasons.append("file_missing")
    else:
        try:
            digest = sha256_file(image_path)
            duplicate_of = seen_sha.get(digest)
            if duplicate_of is not None:
                hard_reasons.append("exact_duplicate")
                metadata["duplicate_of"] = duplicate_of
            else:
                seen_sha[digest] = str(row.get("id", ""))

            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
                aspect_ratio = width / height if height else 0
                brightness, contrast = image_stats(image)
                phash = dhash(image)

                metadata.update(
                    {
                        "width": width,
                        "height": height,
                        "aspect_ratio": round(aspect_ratio, 4),
                        "mode": image.mode,
                        "brightness": round(brightness, 2),
                        "contrast": round(contrast, 2),
                        "dhash": phash,
                    }
                )

                if width < min_dimension or height < min_dimension:
                    hard_reasons.append("too_small")
                if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                    hard_reasons.append("extreme_aspect_ratio")
                if image.mode not in {"RGB", "L"}:
                    review_flags.append("non_rgb_mode")
                if brightness < dark_threshold:
                    review_flags.append("very_dark")
                if brightness > bright_threshold:
                    review_flags.append("very_bright")
                if contrast < low_contrast_threshold:
                    review_flags.append("low_contrast")

                similar_to = seen_dhash.get(phash)
                if similar_to is not None and "exact_duplicate" not in hard_reasons:
                    review_flags.append("perceptual_duplicate_hash")
                    metadata["similar_to"] = similar_to
                else:
                    seen_dhash[phash] = str(row.get("id", ""))

        except Exception as exc:
            hard_reasons.append("unreadable_image")
            metadata["error"] = str(exc)

    if hard_reasons:
        status = "rejected"
    elif review_flags:
        status = "review_required"
    else:
        status = "accepted"

    output.update(metadata)
    output["quality_status"] = status
    output["quality_reasons"] = hard_reasons
    output["review_flags"] = review_flags
    output["draft_candidate"] = status == "accepted"
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_status = Counter(row["quality_status"] for row in rows)
    by_category = defaultdict(Counter)
    by_places = defaultdict(Counter)
    reason_counts: Counter[str] = Counter()
    review_flag_counts: Counter[str] = Counter()

    for row in rows:
        status = str(row["quality_status"])
        by_category[str(row.get("label", ""))][status] += 1
        by_places[str(row.get("places365_slug", ""))][status] += 1
        reason_counts.update(row.get("quality_reasons", []))
        review_flag_counts.update(row.get("review_flags", []))

    return {
        "total_images": len(rows),
        "status_counts": dict(sorted(by_status.items())),
        "category_status_counts": {
            label: dict(sorted(counts.items()))
            for label, counts in sorted(by_category.items())
        },
        "places365_status_counts": {
            slug: dict(sorted(counts.items()))
            for slug, counts in sorted(by_places.items())
        },
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "review_flag_counts": dict(sorted(review_flag_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="data_places365_2 이미지 파일을 초안 생성 전 품질 기준으로 검사합니다."
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-dimension", type=int, default=128)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.33)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.0)
    parser.add_argument("--dark-threshold", type=float, default=10.0)
    parser.add_argument("--bright-threshold", type=float, default=245.0)
    parser.add_argument("--low-contrast-threshold", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv_rows(args.metadata)
    mapping_by_slug = load_mapping(args.mapping)
    seen_sha: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}

    reviewed = [
        review_image_row(
            row,
            mapping_by_slug=mapping_by_slug,
            seen_sha=seen_sha,
            seen_dhash=seen_dhash,
            min_dimension=args.min_dimension,
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            dark_threshold=args.dark_threshold,
            bright_threshold=args.bright_threshold,
            low_contrast_threshold=args.low_contrast_threshold,
        )
        for row in rows
    ]

    accepted = [row for row in reviewed if row["quality_status"] == "accepted"]
    review_required = [
        row for row in reviewed
        if row["quality_status"] == "review_required"
    ]
    rejected = [row for row in reviewed if row["quality_status"] == "rejected"]
    summary = summarize(reviewed)
    summary.update(
        {
            "metadata": str(args.metadata),
            "mapping": str(args.mapping),
            "output_dir": str(args.output_dir),
            "thresholds": {
                "min_dimension": args.min_dimension,
                "min_aspect_ratio": args.min_aspect_ratio,
                "max_aspect_ratio": args.max_aspect_ratio,
                "dark_threshold": args.dark_threshold,
                "bright_threshold": args.bright_threshold,
                "low_contrast_threshold": args.low_contrast_threshold,
            },
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "quality_manifest.jsonl", reviewed)
    write_jsonl(args.output_dir / "accepted_images.jsonl", accepted)
    write_jsonl(args.output_dir / "review_required_images.jsonl", review_required)
    write_jsonl(args.output_dir / "rejected_images.jsonl", rejected)
    write_jsonl(args.output_dir / "draft_candidates.jsonl", accepted)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
