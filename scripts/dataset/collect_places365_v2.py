from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import PROJECT_ROOT, remove_tree_inside_root, stratified_split


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "places365_categories_v2.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_places365_2"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
METADATA_FIELDS = [
    "id",
    "image_path",
    "label",
    "category_slug",
    "places365_id",
    "places365_label",
    "places365_slug",
    "source_dataset",
    "source_split",
    "source_index",
    "source_image_file_path",
    "split",
]
CHECKPOINT_METADATA_NAME = "metadata_checkpoint.csv"


@dataclass(frozen=True)
class ClassMapping:
    category_label: str
    category_slug: str
    places365_id: str
    places365_label: str
    places365_slug: str


def normalize_places_slug(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_categories(
    categories: list[dict[str, Any]],
    category_filter: str,
) -> list[dict[str, Any]]:
    if not category_filter.strip():
        return categories

    requested = {item.strip() for item in category_filter.split(",") if item.strip()}
    by_label = {str(category["label"]): category for category in categories}
    by_slug = {str(category["slug"]): category for category in categories}

    selected: list[dict[str, Any]] = []
    for item in requested:
        category = by_label.get(item) or by_slug.get(item)
        if category is None:
            allowed = ", ".join([*by_label.keys(), *by_slug.keys()])
            raise ValueError(f"알 수 없는 카테고리입니다: {item}. 사용 가능: {allowed}")
        selected.append(category)

    return selected


def build_class_maps(
    categories: list[dict[str, Any]],
) -> tuple[dict[str, ClassMapping], dict[str, ClassMapping]]:
    by_slug: dict[str, ClassMapping] = {}
    by_id: dict[str, ClassMapping] = {}

    for category in categories:
        category_label = str(category["label"])
        category_slug = str(category["slug"])

        for places in category.get("places365", []):
            places365_id = str(places["id"])
            places365_label = str(places["label"])
            places365_slug = normalize_places_slug(
                str(places.get("slug") or places365_label)
            )

            if places365_slug in by_slug:
                raise ValueError(f"중복 Places365 slug 매핑: {places365_slug}")
            if places365_id in by_id:
                raise ValueError(f"중복 Places365 id 매핑: {places365_id}")

            mapping = ClassMapping(
                category_label=category_label,
                category_slug=category_slug,
                places365_id=places365_id,
                places365_label=places365_label,
                places365_slug=places365_slug,
            )
            by_slug[places365_slug] = mapping
            by_id[places365_id] = mapping

    return by_slug, by_id


def iter_selected_mappings(categories: list[dict[str, Any]]) -> Iterable[ClassMapping]:
    by_slug, _ = build_class_maps(categories)
    return by_slug.values()


def prepare_output_dirs(output_dir: Path, mappings: Iterable[ClassMapping]) -> None:
    for mapping in mappings:
        (output_dir / mapping.category_label / mapping.places365_slug).mkdir(
            parents=True,
            exist_ok=True,
        )
    (output_dir / "splits").mkdir(parents=True, exist_ok=True)


def read_metadata_csv(metadata_path: Path) -> list[dict[str, str]]:
    if not metadata_path.exists():
        return []

    with metadata_path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def dedupe_metadata_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}

    for row in rows:
        row_id = str(row.get("id", ""))
        image_path = str(row.get("image_path", ""))
        source_path = str(row.get("source_image_file_path", ""))
        key = row_id or image_path or source_path
        if not key:
            continue
        deduped[key] = row

    return list(deduped.values())


def write_metadata_csv(metadata_path: Path, rows: list[dict[str, Any]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})


def append_metadata_csv(metadata_path: Path, row: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not metadata_path.exists() or metadata_path.stat().st_size == 0

    with metadata_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in METADATA_FIELDS})
        file.flush()


def write_metadata_jsonl(metadata_path: Path, rows: list[dict[str, Any]]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [
        item
        for item in sorted(path.iterdir())
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    ]


def numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def initialize_counts(
    output_dir: Path,
    mappings_by_slug: dict[str, ClassMapping],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    category_counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    next_indices: dict[str, int] = defaultdict(int)

    for places365_slug, mapping in mappings_by_slug.items():
        source_dir = output_dir / mapping.category_label / places365_slug
        images = iter_image_files(source_dir)
        class_counts[places365_slug] = len(images)
        category_counts[mapping.category_label] += len(images)

        numeric_indices = [
            index
            for index in (numeric_stem(path) for path in images)
            if index is not None
        ]
        next_indices[places365_slug] = (
            max(numeric_indices) + 1 if numeric_indices else len(images)
        )

    return category_counts, class_counts, next_indices


def row_id_for(mapping: ClassMapping, image_index: int) -> str:
    return f"{mapping.category_slug}_{mapping.places365_slug}_{image_index:05d}"


def build_existing_file_rows(
    *,
    output_dir: Path,
    mappings_by_slug: dict[str, ClassMapping],
    source_dataset: str,
    source_split: str,
    existing_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    known_paths = {str(row.get("image_path", "")) for row in existing_rows}
    rebuilt_rows: list[dict[str, Any]] = []

    for places365_slug, mapping in mappings_by_slug.items():
        source_dir = output_dir / mapping.category_label / places365_slug
        for image_path in iter_image_files(source_dir):
            relative_image_path = project_relative(image_path)
            if relative_image_path in known_paths:
                continue

            image_index = numeric_stem(image_path)
            if image_index is None:
                image_index = len(rebuilt_rows)

            rebuilt_rows.append(
                {
                    "id": row_id_for(mapping, image_index),
                    "image_path": relative_image_path,
                    "label": mapping.category_label,
                    "category_slug": mapping.category_slug,
                    "places365_id": mapping.places365_id,
                    "places365_label": mapping.places365_label,
                    "places365_slug": mapping.places365_slug,
                    "source_dataset": source_dataset,
                    "source_split": source_split,
                    "source_index": "",
                    "source_image_file_path": "",
                    "split": "",
                }
            )

    return rebuilt_rows


def source_path_from_sample(sample: dict[str, Any]) -> str:
    for key in ("image_file_path", "image_path", "path"):
        value = sample.get(key)
        if value:
            return str(value)
    return ""


def resolve_mapping(
    sample: dict[str, Any],
    mappings_by_slug: dict[str, ClassMapping],
    mappings_by_id: dict[str, ClassMapping],
) -> ClassMapping | None:
    source_path = source_path_from_sample(sample)
    if source_path:
        places365_slug = Path(source_path).parent.name
        mapping = mappings_by_slug.get(places365_slug)
        if mapping is not None:
            return mapping

    for key in ("class_label", "label"):
        if key not in sample:
            continue

        value = sample[key]
        value_text = str(value).strip()
        if value_text.isdigit():
            mapping = mappings_by_id.get(value_text)
            if mapping is not None:
                return mapping

        mapping = mappings_by_slug.get(normalize_places_slug(value_text))
        if mapping is not None:
            return mapping

    return None


def save_image(sample: dict[str, Any], save_path: Path, quality: int) -> None:
    image = sample.get("image")
    if image is None:
        raise ValueError("sample에 image 필드가 없습니다.")

    if hasattr(image, "convert"):
        pil_image = image
    elif isinstance(image, bytes):
        from PIL import Image

        pil_image = Image.open(io.BytesIO(image))
    else:
        raise TypeError(f"지원하지 않는 image 타입입니다: {type(image)!r}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.convert("RGB").save(save_path, "JPEG", quality=quality)


def build_metadata_row(
    *,
    mapping: ClassMapping,
    image_path: Path,
    image_index: int,
    source_dataset: str,
    source_split: str,
    source_index: int,
    source_image_file_path: str,
) -> dict[str, Any]:
    return {
        "id": row_id_for(mapping, image_index),
        "image_path": project_relative(image_path),
        "label": mapping.category_label,
        "category_slug": mapping.category_slug,
        "places365_id": mapping.places365_id,
        "places365_label": mapping.places365_label,
        "places365_slug": mapping.places365_slug,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "source_index": source_index,
        "source_image_file_path": source_image_file_path,
        "split": "",
    }


def build_subcategory_limits(
    *,
    categories: list[dict[str, Any]],
    target_per_category: int,
    max_per_subcategory: int,
    balance_subcategories: bool,
) -> dict[str, int]:
    limits: dict[str, int] = {}

    for category in categories:
        places365_classes = category.get("places365", [])
        if not places365_classes:
            continue

        if balance_subcategories:
            if not target_per_category:
                raise ValueError("--balance-subcategories는 --target-per-category가 필요합니다.")
            limit = math.ceil(target_per_category / len(places365_classes))
        else:
            limit = max_per_subcategory

        if not limit:
            continue

        for places in places365_classes:
            places365_slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limits[places365_slug] = limit

    return limits


def category_done(
    *,
    category: dict[str, Any],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    target_per_category: int,
    subcategory_limits: dict[str, int],
) -> bool:
    category_label = str(category["label"])

    if target_per_category and category_counts[category_label] >= target_per_category:
        return True

    if not subcategory_limits:
        return False

    return all(
        class_counts[normalize_places_slug(str(places.get("slug") or places["label"]))]
        >= subcategory_limits.get(
            normalize_places_slug(str(places.get("slug") or places["label"])),
            0,
        )
        for places in category.get("places365", [])
    )


def all_done(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    target_per_category: int,
    subcategory_limits: dict[str, int],
) -> bool:
    return all(
        category_done(
            category=category,
            category_counts=category_counts,
            class_counts=class_counts,
            target_per_category=target_per_category,
            subcategory_limits=subcategory_limits,
        )
        for category in categories
    )


def warn_impossible_targets(
    *,
    categories: list[dict[str, Any]],
    target_per_category: int,
    subcategory_limits: dict[str, int],
) -> None:
    if not target_per_category or not subcategory_limits:
        return

    for category in categories:
        max_possible = sum(
            subcategory_limits.get(
                normalize_places_slug(str(places.get("slug") or places["label"])),
                0,
            )
            for places in category.get("places365", [])
        )
        if max_possible < target_per_category:
            print(
                "[WARN] "
                f"{category['label']} target_per_category={target_per_category}는 "
                f"현재 서브카테고리 제한 기준 최대 "
                f"{max_possible}장까지만 가능합니다."
            )


def assign_splits(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows_for_split = [dict(row) for row in rows]
    train, valid, test = stratified_split(
        rows_for_split,
        seed=seed,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
    )

    split_by_id: dict[str, str] = {}
    for split_name, split_rows in (
        ("train", train),
        ("valid", valid),
        ("test", test),
    ):
        for row in split_rows:
            split_by_id[str(row["id"])] = split_name
            row["split"] = split_name

    for row in rows:
        row["split"] = split_by_id.get(str(row["id"]), "")

    return train, valid, test


def write_split_files(
    *,
    output_dir: Path,
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> None:
    split_dir = output_dir / "splits"
    for split_name, rows in (("train", train), ("valid", valid), ("test", test)):
        write_metadata_csv(split_dir / f"{split_name}.csv", rows)
        write_metadata_jsonl(split_dir / f"{split_name}.jsonl", rows)


def build_summary(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    subcategory_limits: dict[str, int],
    output_dir: Path,
    source_dataset: str,
    source_split: str,
) -> dict[str, Any]:
    category_summaries = []
    for category in categories:
        class_summaries = []
        for places in category.get("places365", []):
            places365_slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            class_summaries.append(
                {
                    "places365_id": str(places["id"]),
                    "places365_label": str(places["label"]),
                    "places365_slug": places365_slug,
                    "count": class_counts[places365_slug],
                    "limit": subcategory_limits.get(places365_slug, ""),
                }
            )

        category_summaries.append(
            {
                "label": str(category["label"]),
                "slug": str(category["slug"]),
                "count": category_counts[str(category["label"])],
                "places365": class_summaries,
            }
        )

    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "output_dir": project_relative(output_dir),
        "total_images": sum(category_counts.values()),
        "categories": category_summaries,
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def print_progress(
    *,
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    subcategory_limits: dict[str, int],
    categories: list[dict[str, Any]],
    total_saved: int,
) -> None:
    print(f"\n[PROGRESS] 이번 실행 저장: {total_saved}")
    for category in categories:
        category_label = str(category["label"])
        print(f"- {category_label}: {category_counts[category_label]}")
        for places in category.get("places365", []):
            places365_slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limit = subcategory_limits.get(places365_slug)
            suffix = f"/{limit}" if limit else ""
            print(f"  - {places365_slug}: {class_counts[places365_slug]}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Places365 클래스를 LighTrip 7개 카테고리로 매핑해 data_places365_2 이미지 데이터셋을 구축합니다."
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--target-per-category", type=int, default=300)
    parser.add_argument("--max-per-subcategory", type=int, default=120)
    parser.add_argument("--balance-subcategories", action="store_true")
    parser.add_argument("--fill-shortfall", action="store_true")
    parser.add_argument("--max-scan", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--no-split", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def print_dry_run(
    *,
    config_path: Path,
    output_dir: Path,
    dataset_name: str,
    split: str,
    categories: list[dict[str, Any]],
    target_per_category: int,
    max_per_subcategory: int,
    subcategory_limits: dict[str, int],
    balance_subcategories: bool,
) -> None:
    print(f"config: {config_path}")
    print(f"dataset: {dataset_name}")
    print(f"split: {split}")
    print(f"output_dir: {output_dir}")
    print(f"target_per_category: {target_per_category}")
    print(f"max_per_subcategory: {max_per_subcategory}")
    print(f"balance_subcategories: {balance_subcategories}")
    print("\nselected categories:")
    for category in categories:
        print(f"- {category['label']} ({category['slug']}): {len(category.get('places365', []))} classes")
        for places in category.get("places365", []):
            places365_slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limit = subcategory_limits.get(places365_slug)
            if limit:
                print(f"  - {places365_slug}: limit={limit}")


def load_places365_dataset(
    *,
    dataset_name: str,
    split: str,
    shuffle_buffer_size: int,
    seed: int,
) -> Iterable[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return dataset


def build_shortfall_fill_limits(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    target_per_category: int,
) -> dict[str, int]:
    limits: dict[str, int] = {}
    if not target_per_category:
        return limits

    for category in categories:
        category_label = str(category["label"])
        if category_counts[category_label] >= target_per_category:
            continue

        available_slugs = [
            normalize_places_slug(str(places.get("slug") or places["label"]))
            for places in category.get("places365", [])
            if class_counts[
                normalize_places_slug(str(places.get("slug") or places["label"]))
            ]
            > 0
        ]
        if not available_slugs:
            continue

        limit = math.ceil(target_per_category / len(available_slugs))
        for places365_slug in available_slugs:
            limits[places365_slug] = max(limit, class_counts[places365_slug])

    return limits


def print_fill_shortfall_plan(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    fill_limits: dict[str, int],
    target_per_category: int,
) -> None:
    if not fill_limits:
        print("[INFO] 부족분 재분배 대상이 없습니다.")
        return

    print("\n=== 부족분 재분배 계획 ===")
    for category in categories:
        category_label = str(category["label"])
        if category_counts[category_label] >= target_per_category:
            continue

        planned = []
        for places in category.get("places365", []):
            places365_slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limit = fill_limits.get(places365_slug)
            if limit:
                planned.append(f"{places365_slug}:{limit}")

        if planned:
            print(
                f"- {category_label}: {category_counts[category_label]}/"
                f"{target_per_category} -> {', '.join(planned)}"
            )
    print("========================\n")


def collect_dataset_pass(
    *,
    args: argparse.Namespace,
    pass_name: str,
    dataset_name: str,
    source_split: str,
    output_dir: Path,
    categories: list[dict[str, Any]],
    mappings_by_slug: dict[str, ClassMapping],
    mappings_by_id: dict[str, ClassMapping],
    subcategory_limits: dict[str, int],
    seen_source_paths: set[str],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    next_indices: dict[str, int],
    new_rows: list[dict[str, Any]],
    checkpoint_metadata_path: Path,
) -> int:
    dataset = load_places365_dataset(
        dataset_name=dataset_name,
        split=source_split,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
    )
    saved = 0

    for source_index, sample in enumerate(dataset):
        if args.max_scan and source_index >= args.max_scan:
            print(f"[STOP] {pass_name}: max_scan={args.max_scan} 도달")
            break

        mapping = resolve_mapping(sample, mappings_by_slug, mappings_by_id)
        if mapping is None:
            continue

        source_image_file_path = source_path_from_sample(sample)
        if source_image_file_path and source_image_file_path in seen_source_paths:
            continue

        if (
            args.target_per_category
            and category_counts[mapping.category_label] >= args.target_per_category
        ):
            continue

        subcategory_limit = subcategory_limits.get(mapping.places365_slug)
        if subcategory_limit and class_counts[mapping.places365_slug] >= subcategory_limit:
            continue

        image_index = next_indices[mapping.places365_slug]
        save_path = (
            output_dir
            / mapping.category_label
            / mapping.places365_slug
            / f"{image_index:05d}.jpg"
        )

        try:
            save_image(sample, save_path, quality=args.quality)
        except Exception as exc:
            print(f"[ERROR] {pass_name}: idx={source_index} 저장 실패: {exc}")
            continue

        row = build_metadata_row(
            mapping=mapping,
            image_path=save_path,
            image_index=image_index,
            source_dataset=dataset_name,
            source_split=source_split,
            source_index=source_index,
            source_image_file_path=source_image_file_path,
        )
        new_rows.append(row)
        append_metadata_csv(checkpoint_metadata_path, row)

        if source_image_file_path:
            seen_source_paths.add(source_image_file_path)
        class_counts[mapping.places365_slug] += 1
        category_counts[mapping.category_label] += 1
        next_indices[mapping.places365_slug] += 1
        saved += 1

        print(
            f"[SAVE:{pass_name}] {mapping.category_label}/{mapping.places365_slug} "
            f"{class_counts[mapping.places365_slug]}장 "
            f"(category={category_counts[mapping.category_label]})"
        )

        if args.progress_every and saved % args.progress_every == 0:
            print_progress(
                category_counts=category_counts,
                class_counts=class_counts,
                subcategory_limits=subcategory_limits,
                categories=categories,
                total_saved=saved,
            )

        if all_done(
            categories=categories,
            category_counts=category_counts,
            class_counts=class_counts,
            target_per_category=args.target_per_category,
            subcategory_limits=subcategory_limits,
        ):
            print(f"[STOP] {pass_name}: 모든 선택 카테고리 목표를 채웠습니다.")
            break

    return saved


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    dataset_name = args.dataset_name or str(config["dataset"])
    source_split = args.split or str(config.get("split", "train"))
    output_dir = args.output_dir
    categories = select_categories(config["categories"], args.categories)
    mappings_by_slug, mappings_by_id = build_class_maps(categories)
    subcategory_limits = build_subcategory_limits(
        categories=categories,
        target_per_category=args.target_per_category,
        max_per_subcategory=args.max_per_subcategory,
        balance_subcategories=args.balance_subcategories,
    )

    if args.dry_run:
        print_dry_run(
            config_path=args.config_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            split=source_split,
            categories=categories,
            target_per_category=args.target_per_category,
            max_per_subcategory=args.max_per_subcategory,
            subcategory_limits=subcategory_limits,
            balance_subcategories=args.balance_subcategories,
        )
        return

    if args.overwrite and output_dir.exists():
        remove_tree_inside_root(output_dir, PROJECT_ROOT)

    prepare_output_dirs(output_dir, mappings_by_slug.values())
    metadata_csv = output_dir / "metadata.csv"
    checkpoint_metadata_csv = output_dir / CHECKPOINT_METADATA_NAME
    raw_existing_rows = [
        *read_metadata_csv(metadata_csv),
        *read_metadata_csv(checkpoint_metadata_csv),
    ]
    existing_rows: list[dict[str, Any]] = [
        row
        for row in dedupe_metadata_rows(raw_existing_rows)
        if str(row.get("places365_slug", "")) in mappings_by_slug
        and str(row.get("label", ""))
        == mappings_by_slug[str(row.get("places365_slug", ""))].category_label
    ]
    existing_rows.extend(
        build_existing_file_rows(
            output_dir=output_dir,
            mappings_by_slug=mappings_by_slug,
            source_dataset=dataset_name,
            source_split=source_split,
            existing_rows=existing_rows,
        )
    )

    seen_source_paths = {
        str(row.get("source_image_file_path", ""))
        for row in existing_rows
        if row.get("source_image_file_path")
    }
    category_counts, class_counts, next_indices = initialize_counts(
        output_dir,
        mappings_by_slug,
    )

    warn_impossible_targets(
        categories=categories,
        target_per_category=args.target_per_category,
        subcategory_limits=subcategory_limits,
    )

    new_rows: list[dict[str, Any]] = []
    total_saved = collect_dataset_pass(
        args=args,
        pass_name="balanced",
        dataset_name=dataset_name,
        source_split=source_split,
        output_dir=output_dir,
        categories=categories,
        mappings_by_slug=mappings_by_slug,
        mappings_by_id=mappings_by_id,
        subcategory_limits=subcategory_limits,
        seen_source_paths=seen_source_paths,
        category_counts=category_counts,
        class_counts=class_counts,
        next_indices=next_indices,
        new_rows=new_rows,
        checkpoint_metadata_path=checkpoint_metadata_csv,
    )
    effective_subcategory_limits = dict(subcategory_limits)

    if args.fill_shortfall and args.target_per_category:
        fill_limits = build_shortfall_fill_limits(
            categories=categories,
            category_counts=category_counts,
            class_counts=class_counts,
            target_per_category=args.target_per_category,
        )
        print_fill_shortfall_plan(
            categories=categories,
            category_counts=category_counts,
            fill_limits=fill_limits,
            target_per_category=args.target_per_category,
        )
        if fill_limits:
            effective_subcategory_limits.update(fill_limits)
            total_saved += collect_dataset_pass(
                args=args,
                pass_name="fill",
                dataset_name=dataset_name,
                source_split=source_split,
                output_dir=output_dir,
                categories=categories,
                mappings_by_slug=mappings_by_slug,
                mappings_by_id=mappings_by_id,
                subcategory_limits=effective_subcategory_limits,
                seen_source_paths=seen_source_paths,
                category_counts=category_counts,
                class_counts=class_counts,
                next_indices=next_indices,
                new_rows=new_rows,
                checkpoint_metadata_path=checkpoint_metadata_csv,
            )

    rows = dedupe_metadata_rows([*existing_rows, *new_rows])
    if not args.no_split:
        train, valid, test = assign_splits(
            rows,
            seed=args.seed,
            valid_ratio=args.valid_ratio,
            test_ratio=args.test_ratio,
        )
        write_split_files(output_dir=output_dir, train=train, valid=valid, test=test)
    else:
        train, valid, test = [], [], []

    write_metadata_csv(output_dir / "metadata.csv", rows)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    write_summary(
        output_dir,
        build_summary(
            categories=categories,
            category_counts=category_counts,
            class_counts=class_counts,
            subcategory_limits=effective_subcategory_limits,
            output_dir=output_dir,
            source_dataset=dataset_name,
            source_split=source_split,
        ),
    )

    print("\n=== 완료 ===")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"new_saved: {total_saved}")
    print(f"metadata: {(output_dir / 'metadata.csv').resolve()}")
    if not args.no_split:
        print(f"split train/valid/test: {len(train)}/{len(valid)}/{len(test)}")
    print_progress(
        category_counts=category_counts,
        class_counts=class_counts,
        subcategory_limits=effective_subcategory_limits,
        categories=categories,
        total_saved=total_saved,
    )


if __name__ == "__main__":
    main()
