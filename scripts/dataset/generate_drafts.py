from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import labels_by_slug, read_jsonl, write_jsonl


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CATEGORY_HINTS = {
    "카페": "커피, 음료, 디저트, 조용한 자리, 카페 분위기 같은 단서를 자연스럽게 포함해라.",
    "식당": "식사, 메뉴, 음식 맛, 반찬, 한 끼 같은 단서를 자연스럽게 포함해라.",
    "술집": "맥주, 와인, 칵테일, 안주, 한잔, 밤 분위기 같은 단서를 자연스럽게 포함해라.",
    "문화": "전시, 공연, 영화, 책, 예술 공간, 감상 같은 단서를 자연스럽게 포함해라.",
    "운동": "러닝, 헬스, 요가, 자전거, 땀, 몸이 가벼워지는 느낌 같은 단서를 자연스럽게 포함해라.",
    "쇼핑": "백화점 쇼핑, 옷 쇼핑, 마트 장보기 중 이미지와 가장 어울리는 맥락으로 써라. 매장, 진열대, 구경, 고르다, 구매, 쇼핑백, 장바구니, 옷, 신발, 식료품 같은 단서를 자연스럽게 포함해라.",
    "공원": "산책, 나무, 공원, 분수, 산, 숲길, 바람, 자연 속 휴식 같은 맥락으로 써라. 야외를 천천히 걷거나 쉬어가는 블로그 기록처럼 자연스럽게 작성해라.",
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


def guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def iter_images(input_dir: Path, slug_to_label: dict[str, str]) -> list[tuple[Path, str, str]]:
    images: list[tuple[Path, str, str]] = []
    for slug, label in slug_to_label.items():
        category_dir = input_dir / slug
        if not category_dir.exists():
            continue
        for path in sorted(category_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                images.append((path, slug, label))
    return images


def parse_category_filter(value: str, slug_to_label: dict[str, str]) -> dict[str, str]:
    if not value.strip():
        return slug_to_label

    requested = {item.strip() for item in value.split(",") if item.strip()}
    labels_to_slug = {label: slug for slug, label in slug_to_label.items()}
    selected: dict[str, str] = {}

    for item in requested:
        if item in slug_to_label:
            selected[item] = slug_to_label[item]
        elif item in labels_to_slug:
            slug = labels_to_slug[item]
            selected[slug] = item
        else:
            allowed = ", ".join([*slug_to_label.keys(), *labels_to_slug.keys()])
            raise ValueError(f"알 수 없는 카테고리입니다: {item}. 사용 가능: {allowed}")

    return selected


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


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 폴더를 순회하며 Gemma4 초안을 JSONL로 생성합니다.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--output", type=Path, default=Path("data/interim/generated_drafts.jsonl"))
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
    parser.add_argument("--gpu-layers", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    slug_to_label = parse_category_filter(args.categories, labels_by_slug())
    existing_rows = [] if args.overwrite else read_jsonl(args.output)
    replace_slug_to_label = parse_category_filter(args.replace_categories, labels_by_slug())
    replace_labels = set(replace_slug_to_label.values())
    if replace_labels:
        existing_rows = [
            row for row in existing_rows if str(row.get("label", "")) not in replace_labels
        ]
    existing_ids = {str(row.get("id")) for row in existing_rows}
    rows = list(existing_rows)

    images = iter_images(args.input_dir, slug_to_label)
    counts: dict[str, int] = {label: 0 for label in slug_to_label.values()}

    if args.dry_run:
        if not images:
            print(f"이미지를 찾지 못했습니다: {args.input_dir}")
            return

        image_path, slug, label = images[0]
        source_label = infer_source_label_from_path(image_path, slug)
        dataset_prompt = (
            args.user_prompt
            if args.no_category_hint
            else build_dataset_user_prompt(label, args.user_prompt, source_label)
        )
        print(f"sample_image: {image_path}")
        print(f"label: {label}")
        print("user_prompt:")
        print(dataset_prompt or "(empty)")
        return

    if not args.enable_cuda_graphs:
        os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")
    if not args.full_gpu:
        os.environ.setdefault("GEMMA_MODEL_FILENAME", "gemma-4-E2B-it-Q4_K_S.gguf")
        os.environ.setdefault("GEMMA_N_CTX", "768")
        os.environ.setdefault("GEMMA_MAX_TOKENS", "64")
        os.environ.setdefault("GEMMA_N_GPU_LAYERS", str(args.gpu_layers))
        os.environ.setdefault("GEMMA_OFFLOAD_KQV", "0")

    from app.services.gemma_service import (
        ALLOWED_IMAGE_TYPES,
        generate_blog_draft_from_bytes,
        get_llm,
        load_model,
    )

    load_model(verbose=True)
    llm = get_llm()
    if llm is None:
        raise RuntimeError("Gemma4 모델 로드에 실패했습니다.")

    for image_path, slug, label in images:
        if args.limit_total and len(rows) >= args.limit_total:
            break
        if args.limit_per_category and counts[label] >= args.limit_per_category:
            continue

        row_id = build_row_id(slug, image_path)
        if row_id in existing_ids:
            counts[label] += 1
            continue

        content_type = guess_content_type(image_path)
        if content_type not in ALLOWED_IMAGE_TYPES:
            continue

        started_at = time.perf_counter()
        source_label = infer_source_label_from_path(image_path, slug)
        dataset_prompt = (
            args.user_prompt
            if args.no_category_hint
            else build_dataset_user_prompt(label, args.user_prompt, source_label)
        )
        generated_text = generate_blog_draft_from_bytes(
            llm=llm,
            image_bytes=image_path.read_bytes(),
            filename=image_path.name,
            user_prompt=dataset_prompt,
        )
        elapsed = round(time.perf_counter() - started_at, 2)

        row = {
            "id": row_id,
            "image": str(image_path),
            "generated_text": generated_text,
            "label": label,
        }
        if args.include_metadata:
            row.update(
                {
                    "source": "local",
                    "source_label": "",
                    "generation_prompt_type": "custom" if args.no_category_hint else "category_hint",
                    "elapsed_seconds": elapsed,
                    "quality_status": "pending",
                }
            )
        if args.include_prompt:
            row["generation_user_prompt"] = dataset_prompt

        rows.append(row)
        counts[label] += 1
        write_jsonl(args.output, rows)
        print(f"[{label}] {row_id} 생성 완료 ({elapsed}s)")

    print(f"완료: {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
