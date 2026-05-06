from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_categories.json"
DEFAULT_PLACES365_DIR = PROJECT_ROOT / "data_places365"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class DraftImage:
    path: Path
    slug: str
    label: str
    source_label: str = ""


def load_categories(config_path: Path = DEFAULT_CONFIG_PATH) -> list[dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"카테고리 설정이 비어 있습니다: {config_path}")

    return categories


def labels_by_slug(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    return {
        str(category["slug"]): str(category["label"])
        for category in load_categories(config_path)
    }


def slugs_by_label(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, str]:
    return {
        str(category["label"]): str(category["slug"])
        for category in load_categories(config_path)
    }


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSONL 파싱 실패: {exc}") from exc

    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()


def filter_rows_by_labels(
    rows: list[dict[str, Any]],
    labels: set[str],
) -> list[dict[str, Any]]:
    if not labels:
        return rows

    return [
        row for row in rows
        if str(row.get("label", "")) not in labels
    ]


def guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"

    return "application/octet-stream"


def iter_image_files(source_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(source_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def configure_gemma_environment(
    *,
    enable_cuda_graphs: bool,
    full_gpu: bool,
    gpu_layers: int,
    n_ctx: int,
) -> None:
    if not enable_cuda_graphs:
        os.environ.setdefault("GGML_CUDA_DISABLE_GRAPHS", "1")

    if not full_gpu:
        os.environ.setdefault("GEMMA_N_CTX", str(n_ctx))
        os.environ.setdefault("GEMMA_N_GPU_LAYERS", str(gpu_layers))


def load_generation_runtime() -> tuple[set[str], Any, Any]:
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

    return ALLOWED_IMAGE_TYPES, generate_blog_draft_from_bytes, llm


def generate_text_for_image(
    *,
    generate_blog_draft_from_bytes: Any,
    llm: Any,
    image_path: Path,
    dataset_prompt: str,
) -> tuple[str, float]:
    started_at = time.perf_counter()
    generated_text = generate_blog_draft_from_bytes(
        llm=llm,
        image_bytes=image_path.read_bytes(),
        filename=image_path.name,
        user_prompt=dataset_prompt,
    )
    elapsed = round(time.perf_counter() - started_at, 2)
    return generated_text, elapsed


def build_draft_row(
    *,
    row_id: str,
    generated_text: str,
    label: str,
    include_metadata: bool,
    include_prompt: bool,
    dataset_prompt: str,
    generation_prompt_type: str,
    image_path: Path | None = None,
    include_image: bool = False,
    source: str = "",
    source_label: str = "",
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "generated_text": generated_text,
        "label": label,
    }

    if include_image and image_path is not None:
        row["image"] = str(image_path)

    if include_metadata:
        if image_path is not None:
            row["image"] = str(image_path)
        row.update(
            {
                "source": source,
                "source_label": source_label,
                "generation_prompt_type": generation_prompt_type,
                "elapsed_seconds": elapsed_seconds,
                "quality_status": "pending",
            }
        )

    if include_prompt:
        row["generation_user_prompt"] = dataset_prompt

    return row


def split_sort_key(row: dict[str, Any], seed: int) -> str:
    identity = row.get("id") or row.get("image") or row.get("generated_text", "")
    payload = f"{seed}:{row.get('label', '')}:{identity}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for label_rows in grouped.values():
        label_rows.sort(key=lambda row: split_sort_key(row, seed))
        total = len(label_rows)
        valid_count = max(1, int(total * valid_ratio)) if total >= 3 else 0
        test_count = max(1, int(total * test_ratio)) if total >= 3 else 0

        valid.extend(label_rows[:valid_count])
        test.extend(label_rows[valid_count : valid_count + test_count])
        train.extend(label_rows[valid_count + test_count :])

    train.sort(key=lambda row: split_sort_key(row, seed))
    valid.sort(key=lambda row: split_sort_key(row, seed))
    test.sort(key=lambda row: split_sort_key(row, seed))
    return train, valid, test


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "image", "generated_text", "label"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_split_outputs(
    *,
    output_dir: Path,
    train: list[dict[str, Any]],
    valid: list[dict[str, Any]],
    test: list[dict[str, Any]],
    include_csv: bool,
) -> None:
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "valid.jsonl", valid)
    write_jsonl(output_dir / "test.jsonl", test)

    if include_csv:
        write_csv(output_dir / "train.csv", train)
        write_csv(output_dir / "valid.csv", valid)
        write_csv(output_dir / "test.csv", test)


def add_split_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_input: Path,
    default_output_dir: Path,
) -> None:
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--csv", action="store_true")


def run_split_cli(
    *,
    description: str,
    default_input: Path,
    default_output_dir: Path,
) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_split_arguments(
        parser,
        default_input=default_input,
        default_output_dir=default_output_dir,
    )
    args = parser.parse_args()

    train, valid, test = stratified_split(
        read_jsonl(args.input),
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )
    write_split_outputs(
        output_dir=args.output_dir,
        train=train,
        valid=valid,
        test=test,
        include_csv=args.csv,
    )

    print(f"train: {len(train)}")
    print(f"valid: {len(valid)}")
    print(f"test: {len(test)}")


def run_open_images_split_cli() -> None:
    run_split_cli(
        description="검증된 JSONL을 train/valid/test로 계층 분할합니다.",
        default_input=Path("data/processed/accepted_drafts.jsonl"),
        default_output_dir=Path("data/processed"),
    )


def run_places365_split_cli() -> None:
    run_split_cli(
        description="검증된 JSONL을 train/valid/test로 계층 분할합니다.",
        default_input=Path("data_places365/interim/places365_generated_drafts.jsonl"),
        default_output_dir=Path("data_places365/processed"),
    )


def add_generation_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_input_dir: Path,
    default_output: Path,
    default_config_path: Path = Path("configs/dataset_categories.json"),
    default_gpu_layers: int,
) -> None:
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--config-path", type=Path, default=default_config_path)
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
    parser.add_argument("--gpu-layers", type=int, default=default_gpu_layers)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def load_generation_categories(
    args: argparse.Namespace,
    *,
    transform: Callable[[dict[str, str]], dict[str, str]] | None = None,
) -> tuple[dict[str, str], set[str]]:
    all_slug_to_label = labels_by_slug(args.config_path)
    slug_to_label = parse_category_filter(args.categories, all_slug_to_label)
    replace_slug_to_label = parse_category_filter(
        args.replace_categories,
        all_slug_to_label,
    )

    if transform is not None:
        slug_to_label = transform(slug_to_label)
        replace_slug_to_label = transform(replace_slug_to_label)

    return slug_to_label, set(replace_slug_to_label.values())


def load_generation_rows(
    *,
    output: Path,
    overwrite: bool,
    replace_labels: set[str],
    rewrite_when_replaced: bool,
) -> list[dict[str, Any]]:
    if overwrite:
        if output.exists():
            output.unlink()
        return []

    existing_rows = read_jsonl(output)
    kept_rows = filter_rows_by_labels(existing_rows, replace_labels)
    if rewrite_when_replaced and replace_labels:
        write_jsonl(output, kept_rows)
    return kept_rows


def generation_prompt_type(args: argparse.Namespace) -> str:
    return "custom" if args.no_category_hint else "category_hint"


def build_generated_draft_row(
    args: argparse.Namespace,
    *,
    image: DraftImage,
    row_id: str,
    generated_text: str,
    dataset_prompt: str,
    source: str,
    elapsed_seconds: float,
    include_image: bool = False,
    source_label: str | None = None,
) -> dict[str, Any]:
    return build_draft_row(
        row_id=row_id,
        image_path=image.path,
        generated_text=generated_text,
        label=image.label,
        include_image=include_image,
        include_metadata=args.include_metadata,
        include_prompt=args.include_prompt,
        dataset_prompt=dataset_prompt,
        generation_prompt_type=generation_prompt_type(args),
        source=source,
        source_label=image.source_label if source_label is None else source_label,
        elapsed_seconds=elapsed_seconds,
    )


def record_generated_draft(
    *,
    existing_ids: set[str],
    counts: dict[str, int],
    image: DraftImage,
    row_id: str,
) -> None:
    existing_ids.add(row_id)
    counts[image.label] += 1


def should_skip_draft_image(
    *,
    args: argparse.Namespace,
    image: DraftImage,
    row_id: str,
    existing_ids: set[str],
    counts: dict[str, int],
    allowed_image_types: set[str],
) -> bool:
    if args.limit_per_category and counts[image.label] >= args.limit_per_category:
        return True
    if row_id in existing_ids:
        counts[image.label] += 1
        return True
    return guess_content_type(image.path) not in allowed_image_types


def remove_tree_inside_root(path: Path, root: Path) -> None:
    root_path = root.resolve()
    target_path = path.resolve()

    if target_path == root_path or root_path not in target_path.parents:
        raise ValueError(f"삭제 대상이 허용된 출력 폴더 밖입니다: {path}")
    if path.is_symlink():
        raise ValueError(f"심볼릭 링크 폴더는 삭제하지 않습니다: {path}")

    children = sorted(
        target_path.rglob("*"),
        key=lambda item: len(item.parts),
        reverse=True,
    )

    for child in children:
        if child.is_dir() and not child.is_symlink():
            child.rmdir()
        else:
            child.unlink()

    target_path.rmdir()
