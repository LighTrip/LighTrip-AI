from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import (
    DEFAULT_PLACES365_DIR,
    PROJECT_ROOT,
    DraftImage,
    add_generation_arguments,
    append_jsonl,
    build_generated_draft_row,
    configure_gemma_environment,
    generate_text_for_image,
    iter_image_files,
    load_generation_categories,
    load_generation_rows,
    load_generation_runtime,
    read_jsonl,
    record_generated_draft,
    should_skip_draft_image,
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


@dataclass(frozen=True)
class ManifestDraftImage:
    path: Path
    slug: str
    label: str
    source_label: str = ""
    row_id: str = ""
    split: str = ""


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
) -> list[DraftImage]:
    images: list[DraftImage] = []

    for slug, label in slug_to_label.items():
        category_dir = input_dir / label

        for source_dir in iter_places365_source_dirs(category_dir):
            source_label = source_dir.name
            images.extend(
                DraftImage(path=path, slug=slug, label=label, source_label=source_label)
                for path in iter_image_files(source_dir)
            )

    return images


def project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def iter_manifest_images(
    manifest_path: Path,
    slug_to_label: dict[str, str],
) -> list[ManifestDraftImage]:
    rows = read_jsonl(manifest_path)
    allowed_labels = set(slug_to_label.values())
    slug_by_label = {label: slug for slug, label in slug_to_label.items()}
    images: list[ManifestDraftImage] = []

    for index, row in enumerate(rows, start=1):
        label = str(row.get("label", ""))
        if label not in allowed_labels:
            continue

        image_path = str(row.get("image_path") or row.get("image") or "")
        if not image_path:
            print(f"[WARN] manifest row {index} image_path 없음")
            continue

        path = project_path(image_path)
        if not path.exists():
            print(f"[WARN] manifest 이미지 없음: {path}")
            continue

        slug = str(row.get("category_slug") or slug_by_label[label])
        source_label = str(
            row.get("places365_slug")
            or row.get("source_label")
            or path.parent.name
        )
        row_id = str(row.get("id") or build_row_id(slug, source_label, path))
        split = str(row.get("split") or "")

        images.append(
            ManifestDraftImage(
                path=path,
                slug=slug,
                label=label,
                source_label=source_label,
                row_id=row_id,
                split=split,
            )
        )

    return images


def build_row_id(slug: str, source_label: str, image_path: Path) -> str:
    return f"{slug}_{source_label}_{image_path.stem}"


def row_id_for_image(image: DraftImage | ManifestDraftImage) -> str:
    manifest_row_id = getattr(image, "row_id", "")
    if manifest_row_id:
        return manifest_row_id
    return build_row_id(image.slug, image.source_label, image.path)


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
    add_generation_arguments(
        parser,
        default_input_dir=DEFAULT_PLACES365_DIR,
        default_output=Path("data_places365/interim/places365_generated_drafts.jsonl"),
        default_gpu_layers=20,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="생성 대상 이미지 manifest JSONL입니다. 지정하면 input-dir 전체 순회 대신 manifest의 image_path만 사용합니다.",
    )
    return parser.parse_args()


def load_places365_categories(args: argparse.Namespace) -> tuple[dict[str, str], set[str]]:
    slug_to_label, replace_labels = load_generation_categories(
        args,
        transform=remove_etc_category,
    )

    if not slug_to_label:
        raise ValueError(
            "생성할 카테고리가 없습니다. Places365 스크립트에서는 '기타' 카테고리를 제외합니다."
        )

    return slug_to_label, replace_labels


def prepare_existing_rows(args: argparse.Namespace, replace_labels: set[str]) -> list[dict]:
    return load_generation_rows(
        output=args.output,
        overwrite=args.overwrite,
        replace_labels=replace_labels,
        rewrite_when_replaced=True,
    )


def build_prompt_from_args(args: argparse.Namespace, label: str) -> str:
    if args.no_category_hint:
        return args.user_prompt

    return build_dataset_user_prompt(
        label=label,
        extra_prompt=args.user_prompt,
    )


def print_dry_run_sample(
    args: argparse.Namespace,
    images: list[DraftImage],
) -> None:
    if not images:
        print(f"이미지를 찾지 못했습니다: {args.input_dir}")
        return

    image = images[0]
    dataset_prompt = build_prompt_from_args(args, image.label)

    print(f"sample_image: {image.path}")
    print(f"id: {row_id_for_image(image)}")
    print(f"slug: {image.slug}")
    print(f"label: {image.label}")
    print(f"source_label: {image.source_label}")
    if getattr(image, "split", ""):
        print(f"split: {getattr(image, 'split')}")
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
    images: list[DraftImage | ManifestDraftImage],
    existing_ids: set[str],
    counts: dict[str, int],
    total_rows: int,
    allowed_image_types: set[str],
    generate_blog_draft_from_bytes: Any,
    llm: Any,
) -> int:
    for image in images:
        if args.limit_total and total_rows >= args.limit_total:
            break

        row_id = row_id_for_image(image)
        if should_skip_draft_image(
            args=args,
            image=image,
            row_id=row_id,
            existing_ids=existing_ids,
            counts=counts,
            allowed_image_types=allowed_image_types,
        ):
            continue

        dataset_prompt = build_prompt_from_args(args, image.label)
        generated = try_generate_text_for_image(
            generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
            llm=llm,
            image_path=image.path,
            dataset_prompt=dataset_prompt,
        )
        if generated is None:
            continue

        generated_text, elapsed = generated
        row = build_generated_draft_row(
            args,
            image=image,
            row_id=row_id,
            generated_text=generated_text,
            dataset_prompt=dataset_prompt,
            source="places365_manifest" if args.manifest else "places365",
            elapsed_seconds=elapsed,
        )
        if getattr(image, "split", ""):
            row["split"] = getattr(image, "split")

        append_jsonl(args.output, row)

        record_generated_draft(
            existing_ids=existing_ids,
            counts=counts,
            image=image,
            row_id=row_id,
        )
        total_rows += 1

        print(
            f"[SAVE] {image.label}/{image.source_label} | {row_id} 생성 완료 "
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
    images = (
        iter_manifest_images(args.manifest, slug_to_label)
        if args.manifest
        else iter_places365_images(args.input_dir, slug_to_label)
    )

    if args.dry_run:
        print_dry_run_sample(args, images)
        return

    existing_rows = prepare_existing_rows(args, replace_labels)
    existing_ids = {str(row.get("id")) for row in existing_rows}
    total_rows = len(existing_rows)
    counts: dict[str, int] = dict.fromkeys(slug_to_label.values(), 0)

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
