from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from scripts.title_color_recommendation.common import (
        PROJECT_ROOT,
        clear_output_dir,
        ensure_output_dir,
        project_relative,
        resolve_config_path,
        resolve_output_path,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        PROJECT_ROOT,
        clear_output_dir,
        ensure_output_dir,
        project_relative,
        resolve_config_path,
        resolve_output_path,
    )

from scripts.dataset.collect_places365_v2 import (
    build_class_maps,
    build_shortfall_fill_limits,
    build_subcategory_limits,
    build_summary,
    collect_dataset_pass,
    dedupe_metadata_rows,
    load_config,
    normalize_places_slug,
    prepare_collection_run_state,
    print_progress as print_collection_progress,
    select_categories,
    write_metadata_csv,
    write_metadata_jsonl,
    write_summary,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/title_color_recommendation/places365_background_categories.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/title_color_recommendation/raw/places365"


def target_per_category(args: argparse.Namespace, category_count: int) -> int:
    if category_count <= 0:
        raise ValueError("선택된 카테고리가 없습니다.")
    if args.target_per_category:
        return args.target_per_category
    return math.ceil(args.target_total / category_count)


def all_targets_reached(
    *,
    categories: list[dict[str, Any]],
    category_counts: dict[str, int],
    target: int,
) -> bool:
    return all(
        category_counts[str(category["label"])] >= target
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
    parser.add_argument(
        "--no-balance-subcategories",
        dest="balance_subcategories",
        action="store_false",
    )
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


def configure_args(args: argparse.Namespace) -> None:
    args.config_path = resolve_config_path(args.config_path)
    args.output_dir = resolve_output_path(
        args.output_dir,
        description="output directory",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.target_total <= 0:
        raise ValueError("--target-total은 1 이상이어야 합니다.")
    if args.target_per_category < 0:
        raise ValueError("--target-per-category는 0 이상이어야 합니다.")
    if args.max_per_subcategory < 0:
        raise ValueError("--max-per-subcategory는 0 이상이어야 합니다.")
    if args.max_scan < 0:
        raise ValueError("--max-scan은 0 이상이어야 합니다.")
    if args.shuffle_buffer_size < 0:
        raise ValueError("--shuffle-buffer-size는 0 이상이어야 합니다.")
    if not 1 <= args.quality <= 100:
        raise ValueError("--quality는 1..100 범위여야 합니다.")
    if args.progress_every < 0:
        raise ValueError("--progress-every는 0 이상이어야 합니다.")


def runtime_args(args: argparse.Namespace, target: int) -> argparse.Namespace:
    configured = argparse.Namespace(**vars(args))
    configured.target_per_category = target
    configured.log_each_save = False
    return configured


def main() -> None:
    args = parse_args()
    configure_args(args)
    validate_args(args)
    config = load_config(args.config_path)
    dataset_name = args.dataset_name or str(config["dataset"])
    source_split = args.split or str(config.get("split", "train"))
    output_dir = args.output_dir
    categories = select_categories(config["categories"], args.categories)
    target = target_per_category(args, len(categories))
    run_args = runtime_args(args, target)
    mappings_by_slug, mappings_by_id = build_class_maps(categories)
    limits = build_subcategory_limits(
        categories=categories,
        target_per_category=target,
        max_per_subcategory=args.max_per_subcategory,
        balance_subcategories=args.balance_subcategories,
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

    if args.overwrite:
        clear_output_dir(output_dir)
    else:
        ensure_output_dir(output_dir)

    run_state = prepare_collection_run_state(
        output_dir=output_dir,
        categories=categories,
        mappings_by_slug=mappings_by_slug,
        source_dataset=dataset_name,
        source_split=source_split,
        target_per_category=target,
        subcategory_limits=limits,
    )

    new_rows: list[dict[str, Any]] = []
    total_saved = collect_dataset_pass(
        args=run_args,
        pass_name="balanced",
        dataset_name=dataset_name,
        source_split=source_split,
        output_dir=output_dir,
        categories=categories,
        mappings_by_slug=mappings_by_slug,
        mappings_by_id=mappings_by_id,
        subcategory_limits=limits,
        seen_source_paths=run_state.seen_source_paths,
        category_counts=run_state.category_counts,
        class_counts=run_state.class_counts,
        next_indices=run_state.next_indices,
        new_rows=new_rows,
        checkpoint_metadata_path=run_state.checkpoint_metadata_path,
    )
    effective_limits = dict(limits)

    if args.fill_shortfall and not all_targets_reached(
        categories=categories,
        category_counts=run_state.category_counts,
        target=target,
    ):
        fill_limits = build_shortfall_fill_limits(
            categories=categories,
            category_counts=run_state.category_counts,
            class_counts=run_state.class_counts,
            target_per_category=target,
        )
        if fill_limits:
            print("[INFO] 부족 카테고리 보충 pass를 시작합니다.")
            effective_limits.update(fill_limits)
            total_saved += collect_dataset_pass(
                args=run_args,
                pass_name="fill",
                dataset_name=dataset_name,
                source_split=source_split,
                output_dir=output_dir,
                categories=categories,
                mappings_by_slug=mappings_by_slug,
                mappings_by_id=mappings_by_id,
                subcategory_limits=effective_limits,
                seen_source_paths=run_state.seen_source_paths,
                category_counts=run_state.category_counts,
                class_counts=run_state.class_counts,
                next_indices=run_state.next_indices,
                new_rows=new_rows,
                checkpoint_metadata_path=run_state.checkpoint_metadata_path,
            )

    rows = dedupe_metadata_rows([*run_state.existing_rows, *new_rows])
    write_metadata_csv(output_dir / "metadata.csv", rows)
    write_metadata_jsonl(output_dir / "metadata.jsonl", rows)
    write_summary(
        output_dir,
        build_summary(
            categories=categories,
            category_counts=run_state.category_counts,
            class_counts=run_state.class_counts,
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
    print_collection_progress(
        category_counts=run_state.category_counts,
        class_counts=run_state.class_counts,
        subcategory_limits=effective_limits,
        categories=categories,
        total_saved=total_saved,
    )


if __name__ == "__main__":
    main()
