from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_categories.json"
DEFAULT_PLACES365_DIR = PROJECT_ROOT / "data_places365"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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
        os.environ.setdefault("GEMMA_MODEL_FILENAME", "gemma-4-E2B-it-Q4_K_S.gguf")
        os.environ.setdefault("GEMMA_N_CTX", str(n_ctx))
        os.environ.setdefault("GEMMA_MAX_TOKENS", "64")
        os.environ.setdefault("GEMMA_N_GPU_LAYERS", str(gpu_layers))
        os.environ.setdefault("GEMMA_OFFLOAD_KQV", "0")


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
