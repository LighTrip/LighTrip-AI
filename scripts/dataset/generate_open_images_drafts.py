from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from scripts.dataset._bootstrap import bootstrap_project_root

bootstrap_project_root()

from scripts.dataset.common import (
    DraftImage,
    add_generation_arguments,
    build_generated_draft_row,
    configure_gemma_environment,
    generate_text_for_image,
    iter_image_files,
    load_generation_categories,
    load_generation_rows,
    load_generation_runtime,
    record_generated_draft,
    should_skip_draft_image,
    write_jsonl,
)
CATEGORY_HINTS = {
    "카페": "커피, 음료, 디저트, 조용한 자리, 카페 분위기 같은 단서를 자연스럽게 포함해라.",
    "식당": "식사, 메뉴, 음식 맛, 반찬, 한 끼 같은 단서를 자연스럽게 포함해라.",
    "술집": "맥주, 와인, 칵테일, 안주, 한잔, 밤 분위기 같은 단서를 자연스럽게 포함해라.",
    "문화": "전시, 공연, 영화, 책, 예술 공간, 감상 같은 단서를 자연스럽게 포함해라.",
    "운동": "러닝, 헬스, 요가, 자전거, 땀, 몸이 가벼워지는 느낌 같은 단서를 자연스럽게 포함해라.",
    "쇼핑": "백화점 쇼핑, 옷 쇼핑, 마트 장보기 중 이미지와 가장 어울리는 맥락으로 써라. 매장, 진열대, 구경, 고르다, 구매, 쇼핑백, 장바구니, 옷, 신발, 식료품 같은 단서를 자연스럽게 포함해라.",
    "공원": "산책, 나무, 공원, 분수, 산, 숲길, 바람, 자연 속 휴식 같은 맥락으로 써라. 야외를 천천히 걷거나 쉬어가는 블로그 기록처럼 자연스럽게 작성해라.",
    "기타": "특정 카테고리 단어를 억지로 넣지 말고, 이미지 속 장소나 활동을 자연스러운 방문 기록처럼 써라.",
}
PARK_CONTEXT_HINTS = {
    "park": "공원 산책로나 잔디밭을 천천히 걷는 느낌으로 써라.",
    "tree": "나무가 많은 길이나 숲길을 산책하며 쉬어가는 느낌으로 써라.",
    "fountain": "분수가 있는 공원이나 광장을 거닐며 여유를 느끼는 느낌으로 써라.",
    "mountain": "산길이나 전망 좋은 곳을 걷고 자연을 즐기는 느낌으로 써라.",
}
SHOPPING_CONTEXT_HINTS = {
    "shopping_cart": "마트에서 장바구니나 카트를 끌고 식료품을 고르는 장보기 느낌으로 써라.",
    "fruit": "마트나 시장에서 과일을 고르고 장보는 느낌으로 써라.",
    "vegetable": "마트나 시장에서 채소와 식재료를 고르는 장보기 느낌으로 써라.",
    "clothing": "옷가게나 백화점에서 옷을 둘러보고 고르는 쇼핑 느낌으로 써라.",
    "fashion_accessory": "백화점이나 편집숍에서 소품과 액세서리를 구경하는 쇼핑 느낌으로 써라.",
    "handbag": "백화점이나 매장에서 가방을 고르고 쇼핑백을 들고 나오는 느낌으로 써라.",
    "shoe": "신발 매장에서 신발을 신어보고 고르는 쇼핑 느낌으로 써라.",
}


def iter_images(input_dir: Path, slug_to_label: dict[str, str]) -> list[DraftImage]:
    images: list[DraftImage] = []
    for slug, label in slug_to_label.items():
        category_dir = input_dir / slug
        if not category_dir.exists():
            continue
        images.extend(
            DraftImage(path=path, slug=slug, label=label)
            for path in iter_image_files(category_dir)
        )
    return images


def build_row_id(slug: str, image_path: Path) -> str:
    return f"{slug}_{image_path.stem}"


def infer_source_label_from_path(image_path: Path, slug: str) -> str:
    prefix = f"{slug}_"
    stem = image_path.stem
    if not stem.startswith(prefix):
        return ""

    parts = stem[len(prefix) :].split("_", 1)
    if len(parts) != 2:
        return ""

    return parts[1]


def build_dataset_user_prompt(
    label: str,
    extra_prompt: str = "",
    source_label: str = "",
) -> str:
    hint = CATEGORY_HINTS.get(label, "")
    prompt = (
        f'이 데이터의 정답 카테고리는 "{label}"이다.\n'
        "단, 출력에 카테고리명만 기계적으로 반복하지 마라.\n"
        "이미지에서 확인되는 내용과 어울리는 경우에만 장소, 활동, 물건, 감정 단서를 자연스럽게 넣어라.\n"
        "분류 모델이 맥락을 배울 수 있도록 핵심 단어를 최소 1개 이상 포함하되, 실제 블로그 문장처럼 써라."
    )

    if hint:
        prompt += f"\n카테고리 단서 힌트: {hint}"
    if label == "쇼핑" and source_label in SHOPPING_CONTEXT_HINTS:
        prompt += f"\n쇼핑 세부 맥락 힌트: {SHOPPING_CONTEXT_HINTS[source_label]}"
    if label == "공원" and source_label in PARK_CONTEXT_HINTS:
        prompt += f"\n공원 세부 맥락 힌트: {PARK_CONTEXT_HINTS[source_label]}"
    if extra_prompt.strip():
        prompt += f"\n추가 요청: {extra_prompt.strip()}"

    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="이미지 폴더를 순회하며 Gemma4 초안을 JSONL로 생성합니다.")
    add_generation_arguments(
        parser,
        default_input_dir=Path("data/category_classifier/open_images/images"),
        default_output=Path("data/category_classifier/open_images/interim/generated_drafts.jsonl"),
        default_gpu_layers=24,
    )
    return parser.parse_args()


def load_categories_and_replace_labels(
    args: argparse.Namespace,
) -> tuple[dict[str, str], set[str]]:
    return load_generation_categories(args)


def load_existing_rows(args: argparse.Namespace, replace_labels: set[str]) -> list[dict]:
    return load_generation_rows(
        output=args.output,
        overwrite=args.overwrite,
        replace_labels=replace_labels,
        rewrite_when_replaced=True,
    )


def build_prompt_from_args(
    args: argparse.Namespace,
    label: str,
    source_label: str,
) -> str:
    if args.no_category_hint:
        return args.user_prompt

    return build_dataset_user_prompt(label, args.user_prompt, source_label)


def print_dry_run_sample(
    args: argparse.Namespace,
    images: list[DraftImage],
) -> None:
    if not images:
        print(f"이미지를 찾지 못했습니다: {args.input_dir}")
        return

    image = images[0]
    source_label = infer_source_label_from_path(image.path, image.slug)
    dataset_prompt = build_prompt_from_args(args, image.label, source_label)

    print(f"sample_image: {image.path}")
    print(f"label: {image.label}")
    print("user_prompt:")
    print(dataset_prompt or "(empty)")


def generate_open_images_rows(
    args: argparse.Namespace,
    *,
    images: list[DraftImage],
    existing_ids: set[str],
    rows: list[dict],
    counts: dict[str, int],
    allowed_image_types: set[str],
    generate_blog_draft_from_bytes: Any,
    llm: Any,
) -> None:
    for image in images:
        if args.limit_total and len(rows) >= args.limit_total:
            break

        row_id = build_row_id(image.slug, image.path)
        if should_skip_draft_image(
            args=args,
            image=image,
            row_id=row_id,
            existing_ids=existing_ids,
            counts=counts,
            allowed_image_types=allowed_image_types,
        ):
            continue

        source_label = infer_source_label_from_path(image.path, image.slug)
        dataset_prompt = build_prompt_from_args(args, image.label, source_label)
        generated_text, elapsed = generate_text_for_image(
            generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
            llm=llm,
            image_path=image.path,
            dataset_prompt=dataset_prompt,
        )
        row = build_generated_draft_row(
            args,
            image=image,
            row_id=row_id,
            generated_text=generated_text,
            dataset_prompt=dataset_prompt,
            source="local",
            source_label=source_label,
            elapsed_seconds=elapsed,
            include_image=True,
        )

        rows.append(row)
        record_generated_draft(
            existing_ids=existing_ids,
            counts=counts,
            image=image,
            row_id=row_id,
        )
        write_jsonl(args.output, rows)
        print(f"[{image.label}] {row_id} 생성 완료 ({elapsed}s)")

    print(f"완료: {args.output} ({len(rows)} rows)")


def main() -> None:
    args = parse_args()
    slug_to_label, replace_labels = load_categories_and_replace_labels(args)
    images = iter_images(args.input_dir, slug_to_label)

    if args.dry_run:
        print_dry_run_sample(args, images)
        return

    existing_rows = load_existing_rows(args, replace_labels)
    existing_ids = {str(row.get("id")) for row in existing_rows}
    rows = list(existing_rows)
    counts: dict[str, int] = dict.fromkeys(slug_to_label.values(), 0)

    configure_gemma_environment(
        enable_cuda_graphs=args.enable_cuda_graphs,
        full_gpu=args.full_gpu,
        gpu_layers=args.gpu_layers,
        n_ctx=768,
    )
    allowed_image_types, generate_blog_draft_from_bytes, llm = load_generation_runtime()
    generate_open_images_rows(
        args,
        images=images,
        existing_ids=existing_ids,
        rows=rows,
        counts=counts,
        allowed_image_types=allowed_image_types,
        generate_blog_draft_from_bytes=generate_blog_draft_from_bytes,
        llm=llm,
    )


if __name__ == "__main__":
    main()
