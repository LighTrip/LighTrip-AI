from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import (
    DEFAULT_PLACES365_DIR,
    append_jsonl,
    build_draft_row,
    configure_gemma_environment,
    filter_rows_by_labels,
    generate_text_for_image,
    guess_content_type,
    iter_image_files,
    labels_by_slug,
    load_generation_runtime,
    parse_category_filter,
    read_jsonl,
    write_jsonl,
)

CATEGORY_HINTS = {
    "카페": "카페에서 시간을 보내며 음료나 디저트를 즐기는 자연스러운 상황을 떠올려라.",
    "식당": "식당에서 식사를 하며 메뉴나 음식에 대한 경험이 드러나는 상황을 떠올려라.",
    "술집": "술집이나 바에서 술과 함께 분위기를 즐기는 상황을 떠올려라.",
    "문화": "전시, 공연, 영화, 책 등 문화 공간에서 시간을 보내는 상황을 떠올려라.",
    "운동": "운동을 하거나 몸을 움직이며 활동적인 시간을 보내는 상황을 떠올려라.",
    "쇼핑": "매장이나 쇼핑 공간에서 물건을 구경하고 고르는 상황을 떠올려라.",
    "공원": "공원이나 자연 속에서 산책하거나 쉬어가는 상황을 떠올려라.",
}


def remove_etc_category(slug_to_label: dict[str, str]) -> dict[str, str]:
    return {
        slug: label
        for slug, label in slug_to_label.items()
        if label != "기타"
    }


def iter_places365_source_dirs(category_dir: Path) -> list[Path]:
    if not category_dir.exists():
        print(f"[WARN] 카테고리 폴더 없음: {category_dir}")
        return []

    return [
        source_dir
        for source_dir in sorted(category_dir.iterdir())
        if source_dir.is_dir()
    ]


def iter_places365_images(
    input_dir: Path,
    slug_to_label: dict[str, str],
) -> list[tuple[Path, str, str, str]]:
    images: list[tuple[Path, str, str, str]] = []

    for slug, label in slug_to_label.items():
        category_dir = input_dir / label

        for source_dir in iter_places365_source_dirs(category_dir):
            source_label = source_dir.name
            images.extend(
                (path, slug, label, source_label)
                for path in iter_image_files(source_dir)
            )

    return images


def build_row_id(slug: str, source_label: str, image_path: Path) -> str:
    return f"{slug}_{source_label}_{image_path.stem}"


def build_dataset_user_prompt(
    label: str,
    extra_prompt: str = "",
) -> str:
    prompt = f"""
이 데이터의 정답 카테고리는 "{label}"이다.
단, 출력에 카테고리명을 억지로 반복하지 마라.
카테고리와 관련된 상황이 자연스럽게 느껴지도록 작성해라.
""".strip()

    category_hint = CATEGORY_HINTS.get(label)
    if category_hint:
        prompt += f"\n카테고리 상황 힌트: {category_hint}"

    if extra_prompt.strip():
        prompt += f"\n추가 요청: {extra_prompt.strip()}"

    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Places365 이미지 폴더를 순회하며 Gemma4 초안을 JSONL로 생성합니다. 기타 카테고리는 제외합니다."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_PLACES365_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_places365/interim/places365_generated_drafts.jsonl"),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/dataset_categories.json"),
    )

    parser.add_argument("--limit-per-category", type=int, default=0)
    parser.add_argument("--limit-total", type=int, default=0)
    parser.add_argument("--categories", default="")
    parser.add_argument("--replace-categories", default="")
    parser.add_argument("--user-prompt", default="")
    parser.add_argument("--no-category-hint", action="store_true")
    parser.add_argument("--include-metadata", action="store_true")
    parser.add_argument("--include-prompt", action="store_true")
    parser.add_argument("--enable-cuda-graphs", action="store_true")
    parser.add_argument("--full-gpu", action="store_true")
    parser.add_argument("--gpu-layers", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def load_places365_categories(args: argparse.Namespace) -> tuple[dict[str, str], set[str]]:
    all_slug_to_label = labels_by_slug(args.config_path)
    slug_to_label = parse_category_filter(
        args.categories,
        all_slug_to_label,
    )
    slug_to_label = remove_etc_category(slug_to_label)

    if not slug_to_label:
        raise ValueError(
            "생성할 카테고리가 없습니다. Places365 스크립트에서는 '기타' 카테고리를 제외합니다."
        )

    replace_slug_to_label = parse_category_filter(
        args.replace_categories,
        all_slug_to_label,
    )
    replace_slug_to_label = remove_etc_category(replace_slug_to_label)
    return slug_to_label, set(replace_slug_to_label.values())


def prepare_existing_rows(args: argparse.Namespace, replace_labels: set[str]) -> list[dict]:
    if args.overwrite and args.output.exists():
        args.output.unlink()

    existing_rows = [] if args.overwrite else read_jsonl(args.output)

    kept_rows = filter_rows_by_labels(existing_rows, replace_labels)

    if replace_labels:
        write_jsonl(args.output, kept_rows)

    return kept_rows


def build_prompt_from_args(args: argparse.Namespace, label: str) -> str:
    if args.no_category_hint:
        return args.user_prompt

    return build_dataset_user_prompt(
        label=label,
        extra_prompt=args.user_prompt,
    )


def print_dry_run_sample(
    args: argparse.Namespace,
    images: list[tuple[Path, str, str, str]],
) -> None:
    if not images:
        print(f"이미지를 찾지 못했습니다: {args.input_dir}")
        return

    image_path, slug, label, source_label = images[0]
    dataset_prompt = build_prompt_from_args(args, label)

    print(f"sample_image: {image_path}")
    print(f"slug: {slug}")
    print(f"label: {label}")
    print(f"source_label: {source_label}")
    print("\nuser_prompt:")
    print(dataset_prompt or "(empty)")


def try_generate_text_for_image(
    *,
    generate_blog_draft_from_bytes: Any,
    llm: Any,
    image_path: Path,
    dataset_prompt: str,
) -> tuple[str, float] | None:
    try:
        return generate_text_for_image(
            generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
            llm=llm,
            image_path=image_path,
            dataset_prompt=dataset_prompt,
        )
    except Exception as e:
        print(f"[ERROR] {image_path} 생성 실패: {e}")
        return None


def generate_places365_rows(
    args: argparse.Namespace,
    *,
    images: list[tuple[Path, str, str, str]],
    existing_ids: set[str],
    counts: dict[str, int],
    total_rows: int,
    allowed_image_types: set[str],
    generate_blog_draft_from_bytes: Any,
    llm: Any,
) -> int:
    for image_path, slug, label, source_label in images:
        if args.limit_total and total_rows >= args.limit_total:
            break

        if args.limit_per_category and counts[label] >= args.limit_per_category:
            continue

        row_id = build_row_id(slug, source_label, image_path)

        if row_id in existing_ids:
            counts[label] += 1
            continue

        content_type = guess_content_type(image_path)
        if content_type not in allowed_image_types:
            continue

        dataset_prompt = build_prompt_from_args(args, label)
        generated = try_generate_text_for_image(
            generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
            llm=llm,
            image_path=image_path,
            dataset_prompt=dataset_prompt,
        )
        if generated is None:
            continue

        generated_text, elapsed = generated
        row = build_draft_row(
            row_id=row_id,
            image_path=image_path,
            generated_text=generated_text,
            label=label,
            include_metadata=args.include_metadata,
            include_prompt=args.include_prompt,
            dataset_prompt=dataset_prompt,
            generation_prompt_type=(
                "custom" if args.no_category_hint else "category_hint"
            ),
            source="places365",
            source_label=source_label,
            elapsed_seconds=elapsed,
        )

        append_jsonl(args.output, row)

        existing_ids.add(row_id)
        counts[label] += 1
        total_rows += 1

        print(
            f"[SAVE] {label}/{source_label} | {row_id} 생성 완료 "
            f"({elapsed}s, total_rows={total_rows})"
        )

    return total_rows


def print_summary(args: argparse.Namespace, total_rows: int, counts: dict[str, int]) -> None:
    print("\n=== 생성 완료 ===")
    print(f"output: {args.output.resolve()}")
    print(f"total rows: {total_rows}")
    print("이번 실행에서 생성/스킵 처리된 카테고리별 개수:")

    for label, count in counts.items():
        print(f"- {label}: {count}")

    print("\n[INFO] Places365 기반 생성에서는 '기타' 카테고리를 제외했습니다.")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output path: {args.output.resolve()}")

    slug_to_label, replace_labels = load_places365_categories(args)
    existing_rows = prepare_existing_rows(args, replace_labels)
    existing_ids = {str(row.get("id")) for row in existing_rows}
    total_rows = len(existing_rows)
    images = iter_places365_images(args.input_dir, slug_to_label)
    counts: dict[str, int] = dict.fromkeys(slug_to_label.values(), 0)

    if args.dry_run:
        print_dry_run_sample(args, images)
        return

    configure_gemma_environment(
        enable_cuda_graphs=args.enable_cuda_graphs,
        full_gpu=args.full_gpu,
        gpu_layers=args.gpu_layers,
        n_ctx=1024,
    )
    allowed_image_types, generate_blog_draft_from_bytes, llm = load_generation_runtime()
    total_rows = generate_places365_rows(
        args,
        images=images,
        existing_ids=existing_ids,
        counts=counts,
        total_rows=total_rows,
        allowed_image_types=allowed_image_types,
        generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
        llm=llm,
    )
    print_summary(args, total_rows, counts)


if __name__ == "__main__":
    main()
