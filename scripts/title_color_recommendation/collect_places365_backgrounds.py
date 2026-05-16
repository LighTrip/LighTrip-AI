from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.collect_places365_v2 import (  # noqa: E402
    CHECKPOINT_METADATA_NAME,
    build_class_maps,
    build_existing_file_rows,
    build_metadata_row,
    build_summary,
    dedupe_metadata_rows,
    initialize_counts,
    load_config,
    load_places365_dataset,
    normalize_places_slug,
    prepare_output_dirs,
    read_metadata_csv,
    remove_tree_inside_root,
    resolve_mapping,
    save_image,
    select_categories,
    source_path_from_sample,
    write_metadata_csv,
    write_metadata_jsonl,
    write_summary,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/title_color_recommendation/places365_background_categories.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/title_color_recommendation/raw/places365"


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def append_metadata_csv(metadata_path: Path, row: dict[str, Any]) -> None:
    from scripts.dataset.collect_places365_v2 import append_metadata_csv as append_row

    append_row(metadata_path, row)


def target_per_category(args: argparse.Namespace, category_count: int) -> int:
    if args.target_per_category:
        return args.target_per_category
    return math.ceil(args.target_total / category_count)


def build_balanced_limits(
    *,
    categories: list[dict[str, Any]],
    target: int,
    balance_subcategories: bool,
    max_per_subcategory: int,
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for category in categories:
        places365_classes = category.get("places365", [])
        if not places365_classes:
            continue

        limit = (
            math.ceil(target / len(places365_classes))
            if balance_subcategories
            else max_per_subcategory
        )
        if not limit:
            continue

        for places in places365_classes:
            slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limits[slug] = limit
    return limits


def category_finished(
    *,
    category: dict[str, Any],
    category_counts: dict[str, int],
    target: int,
) -> bool:
    return category_counts[str(category["label"])] >= target


def all_finished(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    target: int,
) -> bool:
    return all(
        category_finished(category=category, category_counts=category_counts, target=target)
        for category in categories
    )


def print_plan(
    *,
    config_path: Path,
    output_dir: Path,
    dataset_name: str,
    split: str,
    categories: list[dict[str, Any]],
    target: int,
    limits: dict[str, int],
    args: argparse.Namespace,
) -> None:
    payload = {
        "config": project_relative(config_path),
        "dataset": dataset_name,
        "split": split,
        "output_dir": project_relative(output_dir),
        "target_per_category": target,
        "target_total": target * len(categories),
        "balance_subcategories": args.balance_subcategories,
        "max_per_subcategory": args.max_per_subcategory,
        "categories": [
            {
                "label": category["label"],
                "slug": category["slug"],
                "places365_count": len(category.get("places365", [])),
                "subcategory_limits": {
                    normalize_places_slug(str(places.get("slug") or places["label"])): limits.get(
                        normalize_places_slug(str(places.get("slug") or places["label"])),
                        "",
                    )
                    for places in category.get("places365", [])
                },
            }
            for category in categories
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def print_progress(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    limits: dict[str, int],
    saved: int,
    target: int,
) -> None:
    print(f"\n[PROGRESS] 이번 실행 저장: {saved}")
    for category in categories:
        label = str(category["label"])
        print(f"- {label}: {category_counts[label]}/{target}")
        for places in category.get("places365", []):
            slug = normalize_places_slug(str(places.get("slug") or places["label"]))
            limit = limits.get(slug)
            suffix = f"/{limit}" if limit else ""
            print(f"  - {slug}: {class_counts[slug]}{suffix}")


def collect_pass(
    *,
    args: argparse.Namespace,
    pass_name: str,
    dataset_name: str,
    source_split: str,
    output_dir: Path,
    categories: list[dict[str, Any]],
    mappings_by_slug: dict[str, Any],
    mappings_by_id: dict[str, Any],
    limits: dict[str, int],
    seen_source_paths: set[str],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    next_indices: dict[str, int],
    new_rows: list[dict[str, Any]],
    checkpoint_metadata_path: Path,
    target: int,
) -> int:
    dataset = load_places365_dataset(
        dataset_name=dataset_name,
        split=source_split,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
    )
    saved = 0
    skipped_by_limit: dict[str, int] = defaultdict(int)

    for source_index, sample in enumerate(dataset):
        if args.max_scan and source_index >= args.max_scan:
            print(f"[STOP] {pass_name}: max_scan={args.max_scan} 도달")
            break

        mapping = resolve_mapping(sample, mappings_by_slug, mappings_by_id)
        if mapping is None:
            continue

        if category_counts[mapping.category_label] >= target:
            continue

        source_image_file_path = source_path_from_sample(sample)
        if source_image_file_path and source_image_file_path in seen_source_paths:
            continue

        subcategory_limit = limits.get(mapping.places365_slug)
        if subcategory_limit and class_counts[mapping.places365_slug] >= subcategory_limit:
            skipped_by_limit[mapping.places365_slug] += 1
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

        if args.progress_every and saved % args.progress_every == 0:
            print_progress(
                categories=categories,
                category_counts=category_counts,
                class_counts=class_counts,
                limits=limits,
                saved=saved,
                target=target,
            )

        if all_finished(categories=categories, category_counts=category_counts, target=target):
            print(f"[STOP] {pass_name}: 모든 카테고리 목표를 채웠습니다.")
            break

    if skipped_by_limit:
        print(
            f"[INFO] {pass_name}: subcategory limit으로 건너뛴 class "
            f"{len(skipped_by_limit)}개"
        )
    return saved


def build_fill_limits(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    class_counts: dict[str, int],
    target: int,
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for category in categories:
        label = str(category["label"])
        if category_counts[label] >= target:
            continue

        available = [
            normalize_places_slug(str(places.get("slug") or places["label"]))
            for places in category.get("places365", [])
            if class_counts[normalize_places_slug(str(places.get("slug") or places["label"]))]
            > 0
        ]
        if not available:
            continue

        limit = math.ceil(target / len(available))
        for slug in available:
            limits[slug] = max(limit, class_counts[slug])
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "title_color_recommendation용 Places365 배경 이미지 raw dataset을 수집합니다."
        )
    )
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--categories", default="")
    parser.add_argument("--target-total", type=int, default=30000)
    parser.add_argument("--target-per-category", type=int, default=0)
    parser.add_argument("--max-per-subcategory", type=int, default=0)
    parser.add_argument("--no-balance-subcategories", dest="balance_subcategories", action="store_false")
    parser.set_defaults(balance_subcategories=True)
    parser.add_argument("--no-fill-shortfall", dest="fill_shortfall", action="store_false")
    parser.set_defaults(fill_shortfall=True)
    parser.add_argument("--max-scan", type=int, default=0)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    dataset_name = args.dataset_name or str(config["dataset"])
    source_split = args.split or str(config.get("split", "train"))
    output_dir = args.output_dir
    categories = select_categories(config["categories"], args.categories)
    target = target_per_category(args, len(categories))
    mappings_by_slug, mappings_by_id = build_class_maps(categories)
    limits = build_balanced_limits(
        categories=categories,
        target=target,
        balance_subcategories=args.balance_subcategories,
        max_per_subcategory=args.max_per_subcategory,
    )

    if args.dry_run:
        print_plan(
            config_path=args.config_path,
            output_dir=output_dir,
            dataset_name=dataset_name,
            split=source_split,
            categories=categories,
            target=target,
            limits=limits,
            args=args,
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
    existing_rows = [
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

    new_rows: list[dict[str, Any]] = []
    total_saved = collect_pass(
        args=args,
        pass_name="balanced",
        dataset_name=dataset_name,
        source_split=source_split,
        output_dir=output_dir,
        categories=categories,
        mappings_by_slug=mappings_by_slug,
        mappings_by_id=mappings_by_id,
        limits=limits,
        seen_source_paths=seen_source_paths,
        category_counts=category_counts,
        class_counts=class_counts,
        next_indices=next_indices,
        new_rows=new_rows,
        checkpoint_metadata_path=checkpoint_metadata_csv,
        target=target,
    )
    effective_limits = dict(limits)

    if args.fill_shortfall and not all_finished(
        categories=categories,
        category_counts=category_counts,
        target=target,
    ):
        fill_limits = build_fill_limits(
            categories=categories,
            category_counts=category_counts,
            class_counts=class_counts,
            target=target,
        )
        if fill_limits:
            print("[INFO] 부족 카테고리 보충 pass를 시작합니다.")
            effective_limits.update(fill_limits)
            total_saved += collect_pass(
                args=args,
                pass_name="fill",
                dataset_name=dataset_name,
                source_split=source_split,
                output_dir=output_dir,
                categories=categories,
                mappings_by_slug=mappings_by_slug,
                mappings_by_id=mappings_by_id,
                limits=effective_limits,
                seen_source_paths=seen_source_paths,
                category_counts=category_counts,
                class_counts=class_counts,
                next_indices=next_indices,
                new_rows=new_rows,
                checkpoint_metadata_path=checkpoint_metadata_csv,
                target=target,
            )

    rows = dedupe_metadata_rows([*existing_rows, *new_rows])
    write_metadata_csv(output_dir / "metadata.csv", rows)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    write_summary(
        output_dir,
        build_summary(
            categories=categories,
            category_counts=category_counts,
            class_counts=class_counts,
            subcategory_limits=effective_limits,
            output_dir=output_dir,
            source_dataset=dataset_name,
            source_split=source_split,
        ),
    )

    print("\n=== 완료 ===")
    print(f"output_dir: {output_dir.resolve()}")
    print(f"new_saved: {total_saved}")
    print(f"metadata: {(output_dir / 'metadata.csv').resolve()}")
    print_progress(
        categories=categories,
        category_counts=category_counts,
        class_counts=class_counts,
        limits=effective_limits,
        saved=total_saved,
        target=target,
    )


if __name__ == "__main__":
    main()
