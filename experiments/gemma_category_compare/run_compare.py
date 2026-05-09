from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.category_classifier.src.data import read_jsonl
from experiments.category_classifier.src.evaluate import (
    evaluate_predictions,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
)


DEFAULT_DIRECT_LABELS = ("카페", "식당", "술집", "문화", "운동", "쇼핑", "공원", "기타")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
CATEGORY_LINE_PATTERN = re.compile(r"(?:카테고리|category)\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
DRAFT_LINE_PATTERN = re.compile(r"(?:초안|draft)\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
GEMMA_ENV_DEFAULTS = {
    "GEMMA_MODEL_PATH": "models/gemma-4-E2B-it-Q4_K_S.gguf",
    "GEMMA_MMPROJ_PATH": "models/mmproj-F16.gguf",
    "GEMMA_PROMPT_PATH": "configs/draft_prompt.txt",
    "GEMMA_N_CTX": "4096",
    "GEMMA_MAX_TOKENS": "256",
    "GEMMA_TEMPERATURE": "0.2",
    "GEMMA_TOP_P": "0.9",
    "GEMMA_TOP_K": "40",
    "GEMMA_REPEAT_PENALTY": "1.1",
    "GEMMA_STOP_TOKENS": "<end_of_turn>",
    "GEMMA_N_GPU_LAYERS": "20",
    "GEMMA_MAIN_GPU": "0",
    "GEMMA_OFFLOAD_KQV": "true",
    "GEMMA_MMPROJ_USE_GPU": "true",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gemma 단일 모델 기반 카테고리 분류와 TF-IDF + Linear SVM 분리형 구조를 "
            "동일 데이터셋에서 비교합니다."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("data_places365/processed/test.jsonl"),
        help="정답 label과 generated_text가 들어 있는 평가 JSONL입니다.",
    )
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--image-field", default="image")
    parser.add_argument(
        "--classifier-artifact",
        type=Path,
        default=Path("experiments/category_classifier/artifacts/linear_svm_tfidf.joblib"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="결과 저장 폴더입니다. 생략하면 timestamp 폴더를 생성합니다.",
    )
    parser.add_argument("--limit", type=int, default=0, help="앞에서부터 N개만 평가합니다.")
    parser.add_argument(
        "--categories",
        default="",
        help="정답 label 기준 필터입니다. 예: 카페,식당,운동",
    )
    parser.add_argument(
        "--direct-labels",
        default=",".join(DEFAULT_DIRECT_LABELS),
        help=(
            "Gemma 직접 분류 프롬프트에서 허용할 label 목록입니다. "
            "현재 Places365의 공원까지 비교하려면 카페,식당,술집,문화,운동,쇼핑,공원,기타 처럼 지정하세요."
        ),
    )
    parser.add_argument(
        "--direct-prompt",
        type=Path,
        default=Path("experiments/gemma_category_compare/prompt_json_strict.txt"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("data_places365"),
        help="JSONL에 image 필드가 없을 때 id로 Places365 이미지 경로를 복원할 루트입니다.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/dataset_categories.json"),
        help="slug/label 매핑 설정입니다.",
    )
    parser.add_argument(
        "--run-gemma-direct",
        action="store_true",
        help="image -> Gemma draft+category 직접 생성 실험을 수행합니다.",
    )
    parser.add_argument(
        "--run-split-pipeline",
        action="store_true",
        help="image -> Gemma draft -> SVM 분리형 end-to-end 실험을 수행합니다.",
    )
    parser.add_argument(
        "--skip-classifier-baseline",
        action="store_true",
        help="저장된 generated_text -> SVM 기준선 평가를 생략합니다.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Gemma 직접 분류를 같은 입력에 반복 실행할 횟수입니다. 일관성 평가에 사용합니다.",
    )
    parser.add_argument(
        "--direct-temperature",
        type=float,
        default=0.0,
        help="Gemma 직접 분류 generation temperature입니다. 안정성 비교 기본값은 0.0입니다.",
    )
    parser.add_argument(
        "--direct-max-tokens",
        type=int,
        help="Gemma 직접 분류 max_tokens override입니다.",
    )
    parser.add_argument(
        "--gemma-verbose",
        action="store_true",
        help="llama.cpp/Gemma 런타임 로그를 자세히 출력합니다.",
    )
    parser.add_argument(
        "--no-local-gemma-defaults",
        action="store_true",
        help="Gemma 환경변수가 없을 때 models/의 로컬 기본값을 자동 주입하지 않습니다.",
    )
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        help="predict_proba 지원 classifier에서 threshold 미만이면 기타로 처리합니다.",
    )
    parser.add_argument(
        "--fail-on-missing-image",
        action="store_true",
        help="Gemma 실험 중 이미지 경로를 찾지 못하면 즉시 실패합니다.",
    )
    return parser.parse_args()


def comma_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def ordered_unique(values: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def configure_local_gemma_defaults(*, enabled: bool) -> None:
    if not enabled:
        return

    for name, value in GEMMA_ENV_DEFAULTS.items():
        if name in {"GEMMA_MODEL_PATH", "GEMMA_MMPROJ_PATH", "GEMMA_PROMPT_PATH"}:
            candidate = resolve_project_path(Path(value))
            if candidate.exists():
                os.environ.setdefault(name, value)
            continue
        os.environ.setdefault(name, value)


def path_for_output(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "experiments" / "gemma_category_compare" / "results" / timestamp


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_category_config(config_path: Path) -> dict[str, str]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    categories = payload.get("categories", [])
    if not isinstance(categories, list):
        raise ValueError(f"categories 형식이 올바르지 않습니다: {config_path}")

    slug_to_label: dict[str, str] = {}
    for category in categories:
        if not isinstance(category, dict):
            continue
        slug = category.get("slug")
        label = category.get("label")
        if isinstance(slug, str) and isinstance(label, str):
            slug_to_label[slug] = label
    return slug_to_label


def build_places365_image_index(image_root: Path, config_path: Path) -> dict[str, Path]:
    slug_to_label = load_category_config(config_path)
    image_index: dict[str, Path] = {}

    for slug, label in slug_to_label.items():
        category_dir = image_root / label
        if not category_dir.exists():
            continue

        for source_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            for image_path in sorted(source_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                row_id = f"{slug}_{source_dir.name}_{image_path.stem}"
                image_index[row_id] = image_path

    return image_index


def resolve_image_path(
    row: dict[str, Any],
    *,
    image_field: str,
    id_field: str,
    image_index: dict[str, Path],
) -> Path | None:
    raw_image = row.get(image_field)
    if isinstance(raw_image, str) and raw_image.strip():
        image_path = resolve_project_path(Path(raw_image))
        if image_path.exists():
            return image_path

    row_id = row.get(id_field)
    if isinstance(row_id, str):
        return image_index.get(row_id)

    return None


def load_eval_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = read_jsonl(resolve_project_path(args.input_jsonl))
    category_filter = set(comma_values(args.categories))
    if category_filter:
        rows = [
            row for row in rows
            if str(row.get(args.label_field, "")) in category_filter
        ]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("평가할 row가 없습니다.")
    return rows


def load_classifier(artifact_path: Path) -> dict[str, Any]:
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise ValueError(f"classifier artifact 형식이 올바르지 않습니다: {artifact_path}")
    return artifact


def max_probability(pipeline: Any, text: str) -> float | None:
    if not hasattr(pipeline, "predict_proba"):
        return None
    probabilities = pipeline.predict_proba([text])
    return float(probabilities[0].max())


def apply_unknown_threshold(
    label: str,
    probability: float | None,
    threshold: float | None,
) -> str:
    if threshold is not None and probability is not None and probability < threshold:
        return "기타"
    return label


def predict_classifier_text(
    pipeline: Any,
    text: str,
    *,
    unknown_threshold: float | None,
) -> tuple[str, str, float | None, float]:
    started_at = time.perf_counter()
    raw_label = str(pipeline.predict([text])[0])
    probability = max_probability(pipeline, text)
    label = apply_unknown_threshold(raw_label, probability, unknown_threshold)
    elapsed = time.perf_counter() - started_at
    return label, raw_label, probability, elapsed


def run_classifier_baseline(
    rows: list[dict[str, Any]],
    artifact: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pipeline = artifact["pipeline"]
    results: list[dict[str, Any]] = []

    for row in rows:
        text = str(row.get(args.text_field, "")).strip()
        if not text:
            raise ValueError(f"{row.get(args.id_field, '<no-id>')} row의 {args.text_field}가 비어 있습니다.")

        predicted_label, raw_label, probability, elapsed = predict_classifier_text(
            pipeline,
            text,
            unknown_threshold=args.unknown_threshold,
        )
        results.append(
            {
                "id": row.get(args.id_field),
                "true_label": row.get(args.label_field),
                "generated_text": text,
                "predicted_label": predicted_label,
                "raw_predicted_label": raw_label,
                "confidence": probability,
                "elapsed_seconds": round(elapsed, 6),
            }
        )

    return results


def render_direct_prompt(prompt_path: Path, labels: list[str]) -> str:
    template = prompt_path.read_text(encoding="utf-8").strip()
    labels_block = "\n".join(f"- {label}" for label in labels)
    labels_inline = ", ".join(labels)
    return template.replace("{labels_block}", labels_block).replace("{labels}", labels_inline)


def prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]


def normalize_label(candidate: Any, labels: list[str]) -> str | None:
    if not isinstance(candidate, str):
        return None

    stripped = candidate.strip().strip('"').strip("'").strip()
    if stripped in labels:
        return stripped

    for label in labels:
        if label in stripped:
            return label

    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    match = JSON_OBJECT_PATTERN.search(text)
    if not match:
        return None

    payload = match.group(0)
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, dict) else None


def parse_direct_output(raw_output: str, labels: list[str]) -> dict[str, Any]:
    decoded = parse_json_object(raw_output)
    if decoded is not None:
        category = normalize_label(
            decoded.get("category") or decoded.get("카테고리") or decoded.get("label"),
            labels,
        )
        draft = decoded.get("draft") or decoded.get("초안") or decoded.get("generated_text")
        if category is not None:
            return {
                "draft": str(draft).strip() if isinstance(draft, str) else "",
                "predicted_label": category,
                "parse_status": "json",
            }

    category = None
    line_match = CATEGORY_LINE_PATTERN.search(raw_output)
    if line_match:
        category = normalize_label(line_match.group(1), labels)

    if category is None:
        category = normalize_label(raw_output, labels)

    draft = ""
    draft_match = DRAFT_LINE_PATTERN.search(raw_output)
    if draft_match:
        draft = draft_match.group(1).strip()
        category_split = CATEGORY_LINE_PATTERN.search(draft)
        if category_split:
            draft = draft[: category_split.start()].strip()

    return {
        "draft": draft,
        "predicted_label": category or "기타",
        "parse_status": "text" if category is not None else "failed",
    }


def gemma_generation_kwargs(
    *,
    direct_temperature: float,
    direct_max_tokens: int | None,
) -> dict[str, Any]:
    from app.config.gemma_config import GEMMA_CONFIG

    kwargs = GEMMA_CONFIG.generation.as_chat_completion_kwargs()
    kwargs["temperature"] = direct_temperature
    if direct_max_tokens is not None:
        kwargs["max_tokens"] = direct_max_tokens
    return kwargs


def load_gemma_llm(*, verbose: bool) -> Any:
    from app.services.gemma_service import get_llm, load_model

    load_model(verbose=verbose)
    llm = get_llm()
    if llm is None:
        raise RuntimeError("Gemma 모델 로드에 실패했습니다.")
    return llm


def generate_direct_with_gemma(
    llm: Any,
    image_path: Path,
    prompt_text: str,
    completion_kwargs: dict[str, Any],
    *,
    verbose: bool,
) -> tuple[str, float]:
    from app.prompts.gemma_formatter import (
        build_vision_messages,
        bytes_to_data_uri,
        extract_chat_message_text,
    )
    from llama_cpp.llama_chat_format import suppress_stdout_stderr

    image_data_uri = bytes_to_data_uri(image_path.read_bytes(), image_path.name)
    started_at = time.perf_counter()
    with suppress_stdout_stderr(disable=verbose):
        response = llm.create_chat_completion(
            messages=build_vision_messages(image_data_uri, prompt_text),
            **completion_kwargs,
        )
    elapsed = time.perf_counter() - started_at
    return extract_chat_message_text(response), elapsed


def run_gemma_direct(
    rows: list[dict[str, Any]],
    *,
    llm: Any,
    image_index: dict[str, Path],
    direct_labels: list[str],
    prompt_text: str,
    completion_kwargs: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for repeat_index in range(max(1, args.repeat)):
        for row in rows:
            image_path = resolve_image_path(
                row,
                image_field=args.image_field,
                id_field=args.id_field,
                image_index=image_index,
            )
            if image_path is None:
                message = f"이미지 경로를 찾지 못했습니다: {row.get(args.id_field)}"
                if args.fail_on_missing_image:
                    raise FileNotFoundError(message)
                results.append(
                    {
                        "id": row.get(args.id_field),
                        "true_label": row.get(args.label_field),
                        "image": None,
                        "predicted_label": "기타",
                        "raw_output": "",
                        "generated_text": "",
                        "parse_status": "missing_image",
                        "repeat_index": repeat_index,
                        "elapsed_seconds": 0.0,
                        "error": message,
                    }
                )
                continue

            raw_output, elapsed = generate_direct_with_gemma(
                llm=llm,
                image_path=image_path,
                prompt_text=prompt_text,
                completion_kwargs=completion_kwargs,
                verbose=args.gemma_verbose,
            )
            parsed = parse_direct_output(raw_output, direct_labels)
            results.append(
                    {
                        "id": row.get(args.id_field),
                        "true_label": row.get(args.label_field),
                        "image": path_for_output(image_path),
                        "predicted_label": parsed["predicted_label"],
                        "generated_text": parsed["draft"],
                        "raw_output": raw_output,
                    "parse_status": parsed["parse_status"],
                    "repeat_index": repeat_index,
                    "elapsed_seconds": round(elapsed, 6),
                }
            )

    return results


def generate_split_draft(
    llm: Any,
    image_path: Path,
    *,
    verbose: bool,
) -> tuple[str, float]:
    from app.services.gemma_service import generate_blog_draft_from_bytes
    from llama_cpp.llama_chat_format import suppress_stdout_stderr

    started_at = time.perf_counter()
    with suppress_stdout_stderr(disable=verbose):
        draft = generate_blog_draft_from_bytes(
            llm=llm,
            image_bytes=image_path.read_bytes(),
            filename=image_path.name,
        )
    elapsed = time.perf_counter() - started_at
    return draft, elapsed


def run_split_pipeline(
    rows: list[dict[str, Any]],
    *,
    llm: Any,
    artifact: dict[str, Any],
    image_index: dict[str, Path],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pipeline = artifact["pipeline"]
    results: list[dict[str, Any]] = []

    for row in rows:
        image_path = resolve_image_path(
            row,
            image_field=args.image_field,
            id_field=args.id_field,
            image_index=image_index,
        )
        if image_path is None:
            message = f"이미지 경로를 찾지 못했습니다: {row.get(args.id_field)}"
            if args.fail_on_missing_image:
                raise FileNotFoundError(message)
            results.append(
                {
                    "id": row.get(args.id_field),
                    "true_label": row.get(args.label_field),
                    "image": None,
                    "generated_text": "",
                    "predicted_label": "기타",
                    "raw_predicted_label": "기타",
                    "draft_elapsed_seconds": 0.0,
                    "classifier_elapsed_seconds": 0.0,
                    "elapsed_seconds": 0.0,
                    "error": message,
                }
            )
            continue

        total_started_at = time.perf_counter()
        draft, draft_elapsed = generate_split_draft(
            llm,
            image_path,
            verbose=args.gemma_verbose,
        )
        predicted_label, raw_label, probability, classifier_elapsed = predict_classifier_text(
            pipeline,
            draft,
            unknown_threshold=args.unknown_threshold,
        )
        total_elapsed = time.perf_counter() - total_started_at
        results.append(
            {
                "id": row.get(args.id_field),
                "true_label": row.get(args.label_field),
                "image": path_for_output(image_path),
                "generated_text": draft,
                "predicted_label": predicted_label,
                "raw_predicted_label": raw_label,
                "confidence": probability,
                "draft_elapsed_seconds": round(draft_elapsed, 6),
                "classifier_elapsed_seconds": round(classifier_elapsed, 6),
                "elapsed_seconds": round(total_elapsed, 6),
            }
        )

    return results


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def time_summary(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    elapsed_values = [
        float(record["elapsed_seconds"])
        for record in records
        if isinstance(record.get("elapsed_seconds"), (int, float))
    ]
    if not elapsed_values:
        return {
            "count": 0,
            "total_seconds": None,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }

    return {
        "count": len(elapsed_values),
        "total_seconds": round(sum(elapsed_values), 6),
        "mean_seconds": round(statistics.mean(elapsed_values), 6),
        "median_seconds": round(statistics.median(elapsed_values), 6),
        "p95_seconds": round(percentile(elapsed_values, 0.95) or 0.0, 6),
        "min_seconds": round(min(elapsed_values), 6),
        "max_seconds": round(max(elapsed_values), 6),
    }


def first_repeat_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any("repeat_index" in record for record in records):
        return records
    return [
        record for record in records
        if int(record.get("repeat_index", 0)) == 0
    ]


def parse_success_rate(records: list[dict[str, Any]]) -> float | None:
    parse_records = [
        record for record in records
        if "parse_status" in record
    ]
    if not parse_records:
        return None
    success_count = sum(
        1 for record in parse_records
        if record.get("parse_status") in {"json", "text"}
    )
    return round(success_count / len(parse_records), 6)


def consistency_summary(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        row_id = str(record.get("id"))
        grouped[row_id].append(str(record.get("predicted_label")))

    repeated_groups = {
        row_id: labels for row_id, labels in grouped.items()
        if len(labels) > 1
    }
    if not repeated_groups:
        return {
            "groups": len(grouped),
            "repeated_groups": 0,
            "consistent_groups": None,
            "consistency_rate": None,
        }

    consistent_groups = sum(
        1 for labels in repeated_groups.values()
        if len(set(labels)) == 1
    )
    return {
        "groups": len(grouped),
        "repeated_groups": len(repeated_groups),
        "consistent_groups": consistent_groups,
        "consistency_rate": round(consistent_groups / len(repeated_groups), 6),
    }


def method_metrics(
    method_name: str,
    records: list[dict[str, Any]],
    labels: list[str],
    report_dir: Path,
) -> dict[str, Any]:
    metric_records = first_repeat_records(records)
    y_true = [str(record["true_label"]) for record in metric_records]
    y_pred = [str(record["predicted_label"]) for record in metric_records]
    metrics = evaluate_predictions(y_true, y_pred, labels)
    metrics.update(
        {
            "method": method_name,
            "rows": len(metric_records),
            "time": time_summary(metric_records),
            "parse_success_rate": parse_success_rate(records),
            "consistency": consistency_summary(records),
        }
    )

    save_classification_report(
        report_dir / f"{method_name}_classification_report.txt",
        y_true,
        y_pred,
        labels,
    )
    save_confusion_matrix(
        report_dir / f"{method_name}_confusion_matrix.csv",
        y_true,
        y_pred,
        labels,
    )
    return metrics


def evaluation_label_order(
    rows: list[dict[str, Any]],
    results_by_method: dict[str, list[dict[str, Any]]],
    direct_labels: list[str],
    label_field: str,
) -> list[str]:
    values: list[str] = []
    values.extend(direct_labels)
    values.extend(str(row[label_field]) for row in rows)
    for records in results_by_method.values():
        values.extend(str(record.get("predicted_label")) for record in records)
    return ordered_unique(values)


def metric_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics.get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def seconds_value(metrics: dict[str, Any], key: str) -> str:
    value = metrics["time"].get(key)
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def build_summary_markdown(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    direct_labels: list[str],
    prompt_text: str,
    metrics_by_method: dict[str, dict[str, Any]],
    label_field: str,
) -> str:
    true_labels = ordered_unique([str(row.get(label_field)) for row in rows])
    direct_missing = [label for label in true_labels if label not in direct_labels]

    lines = [
        "# Gemma Direct Category vs Split Classifier Experiment",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- input_jsonl: `{args.input_jsonl}`",
        f"- rows: {len(rows)}",
        f"- direct_labels: {', '.join(direct_labels)}",
        f"- direct_prompt: `{args.direct_prompt}`",
        f"- direct_prompt_hash: `{prompt_hash(prompt_text)}`",
        f"- output_dir: `{output_dir}`",
        "",
    ]

    if direct_missing:
        lines.extend(
            [
                "## Label Mismatch Note",
                "",
                "평가 데이터에는 있지만 Gemma 직접 분류 프롬프트에는 없는 label이 있습니다.",
                f"- missing_from_direct_labels: {', '.join(direct_missing)}",
                "필요하면 `--direct-labels`에 해당 label을 추가해 공정 비교를 다시 실행하세요.",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Summary",
            "",
            "| method | rows | accuracy | macro_f1 | mean_sec | p95_sec | parse_success | consistency |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_name, metrics in metrics_by_method.items():
        parse_rate = metrics.get("parse_success_rate")
        consistency = metrics.get("consistency", {}).get("consistency_rate")
        parse_text = f"{parse_rate:.4f}" if isinstance(parse_rate, float) else "-"
        consistency_text = f"{consistency:.4f}" if isinstance(consistency, float) else "-"
        lines.append(
            f"| {method_name} | "
            f"{metrics.get('rows', 0)} | "
            f"{metric_value(metrics, 'accuracy')} | "
            f"{metric_value(metrics, 'f1_macro')} | "
            f"{seconds_value(metrics, 'mean_seconds')} | "
            f"{seconds_value(metrics, 'p95_seconds')} | "
            f"{parse_text} | "
            f"{consistency_text} |"
        )

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `metrics.json`: 전체 수치와 classification report dict",
            "- `*_predictions.jsonl`: method별 raw 예측 결과",
            "- `reports/*_classification_report.txt`: 카테고리별 precision/recall/F1",
            "- `reports/*_confusion_matrix.csv`: confusion matrix",
            "",
            "## Recommended Read",
            "",
            "- `classifier_existing_draft`는 저장된 Gemma 초안을 SVM에 넣은 빠른 기준선입니다.",
            "- `gemma_direct`는 같은 이미지에서 Gemma가 초안과 카테고리를 동시에 출력한 결과입니다.",
            "- `split_pipeline`은 `--run-split-pipeline`을 켰을 때만 생성되며 실제 운영형 end-to-end와 가장 가깝습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_experiment_outputs(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    direct_labels: list[str],
    prompt_text: str,
    results_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "reports"

    for method_name, records in results_by_method.items():
        write_jsonl(output_dir / f"{method_name}_predictions.jsonl", records)

    labels = evaluation_label_order(
        rows,
        results_by_method,
        direct_labels,
        args.label_field,
    )
    metrics_by_method = {
        method_name: method_metrics(method_name, records, labels, report_dir)
        for method_name, records in results_by_method.items()
    }

    metrics_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_jsonl": str(args.input_jsonl),
        "rows": len(rows),
        "labels": labels,
        "direct_labels": direct_labels,
        "direct_prompt_path": str(args.direct_prompt),
        "direct_prompt_hash": prompt_hash(prompt_text),
        "methods": metrics_by_method,
    }
    save_metrics(output_dir / "metrics.json", metrics_payload)

    summary = build_summary_markdown(
        args=args,
        output_dir=output_dir,
        rows=rows,
        direct_labels=direct_labels,
        prompt_text=prompt_text,
        metrics_by_method=metrics_by_method,
        label_field=args.label_field,
    )
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    return metrics_payload


def print_run_summary(metrics_payload: dict[str, Any], output_dir: Path) -> None:
    summary = {
        "output_dir": str(output_dir),
        "rows": metrics_payload["rows"],
        "methods": {
            method_name: {
                "accuracy": method_metrics["accuracy"],
                "f1_macro": method_metrics["f1_macro"],
                "mean_seconds": method_metrics["time"]["mean_seconds"],
            }
            for method_name, method_metrics in metrics_payload["methods"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir) if args.output_dir else default_output_dir()
    rows = load_eval_rows(args)
    direct_labels = ordered_unique(comma_values(args.direct_labels))
    if "기타" not in direct_labels:
        direct_labels.append("기타")

    prompt_path = resolve_project_path(args.direct_prompt)
    prompt_text = render_direct_prompt(prompt_path, direct_labels)
    image_root = resolve_project_path(args.image_root)
    config_path = resolve_project_path(args.config_path)

    image_index: dict[str, Path] = {}
    if args.run_gemma_direct or args.run_split_pipeline:
        image_index = build_places365_image_index(image_root, config_path)

    artifact: dict[str, Any] | None = None
    if not args.skip_classifier_baseline or args.run_split_pipeline:
        artifact = load_classifier(resolve_project_path(args.classifier_artifact))
    results_by_method: dict[str, list[dict[str, Any]]] = {}

    if not args.skip_classifier_baseline:
        assert artifact is not None
        results_by_method["classifier_existing_draft"] = run_classifier_baseline(
            rows,
            artifact,
            args,
        )

    llm: Any | None = None
    if args.run_gemma_direct or args.run_split_pipeline:
        configure_local_gemma_defaults(enabled=not args.no_local_gemma_defaults)
        llm = load_gemma_llm(verbose=args.gemma_verbose)

    if args.run_gemma_direct:
        assert llm is not None
        completion_kwargs = gemma_generation_kwargs(
            direct_temperature=args.direct_temperature,
            direct_max_tokens=args.direct_max_tokens,
        )
        results_by_method["gemma_direct"] = run_gemma_direct(
            rows,
            llm=llm,
            image_index=image_index,
            direct_labels=direct_labels,
            prompt_text=prompt_text,
            completion_kwargs=completion_kwargs,
            args=args,
        )

    if args.run_split_pipeline:
        assert llm is not None
        assert artifact is not None
        results_by_method["split_pipeline"] = run_split_pipeline(
            rows,
            llm=llm,
            artifact=artifact,
            image_index=image_index,
            args=args,
        )

    if not results_by_method:
        raise SystemExit("실행된 method가 없습니다. 옵션을 확인하세요.")

    metrics_payload = save_experiment_outputs(
        args=args,
        output_dir=output_dir,
        rows=rows,
        direct_labels=direct_labels,
        prompt_text=prompt_text,
        results_by_method=results_by_method,
    )
    print_run_summary(metrics_payload, output_dir)


if __name__ == "__main__":
    main()
