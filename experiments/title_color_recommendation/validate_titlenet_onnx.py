from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from scripts.title_color_recommendation.export_titlenet_onnx import (
    DEFAULT_CHECKPOINT,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    LOGITS_OUTPUT_NAME,
    TOP1_OUTPUT_NAME,
    build_model,
    load_checkpoint,
    model_config_from_checkpoint,
)
from src.title_color_recommendation.data.dataset import TitleColorDataset


DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_LOGITS_ONNX = Path("outputs/title_color_recommendation/onnx/titlenet_logits.onnx")
DEFAULT_TOP1_ONNX = Path("outputs/title_color_recommendation/onnx/titlenet_top1.onnx")
DEFAULT_PALETTE = Path("data/title_color_recommendation/processed/palette.json")
DEFAULT_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/onnx/titlenet_onnx_parity_report.md"
)
DEFAULT_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/titlenet_onnx_parity_metrics.json"
)
DEFAULT_SPLIT = "test"
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SEED = 42
DEFAULT_MAX_ABS_DIFF_THRESHOLD = 1e-4
DEFAULT_MEAN_ABS_DIFF_THRESHOLD = 1e-5


@dataclass(frozen=True)
class SampleParityResult:
    image_id: str
    pytorch_top1: int
    onnx_logits_top1: int
    onnx_top1: int
    top3_match: bool
    top5_match: bool
    max_abs_diff: float
    mean_abs_diff: float

    @property
    def top1_match(self) -> bool:
        return self.pytorch_top1 == self.onnx_logits_top1 == self.onnx_top1


@dataclass(frozen=True)
class ParityMetrics:
    split: str
    sample_count: int
    seed: int
    max_abs_diff: float
    mean_abs_diff: float
    top1_agreement: float
    top3_agreement: float
    top5_agreement: float
    max_abs_diff_threshold: float
    mean_abs_diff_threshold: float
    passed: bool
    failure_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate TitLeNet PyTorch and ONNX inference parity."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--logits-onnx", type=Path, default=DEFAULT_LOGITS_ONNX)
    parser.add_argument("--top1-onnx", type=Path, default=DEFAULT_TOP1_ONNX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--max-abs-diff-threshold",
        type=float,
        default=DEFAULT_MAX_ABS_DIFF_THRESHOLD,
    )
    parser.add_argument(
        "--mean-abs-diff-threshold",
        type=float,
        default=DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
    )
    return parser.parse_args(argv)


def display_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def optional_project_path(
    value: Path | None,
    *,
    must_exist: bool,
    description: str,
) -> Path | None:
    if value is None:
        return None
    return resolve_inside_project(
        PROJECT_ROOT,
        value,
        must_exist=must_exist,
        description=description,
    )


def load_palette_ids(path: Path, *, num_classes: int) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"palette must be a list: {path}")
    ids = {int(item["id"]) for item in payload}
    expected_ids = set(range(num_classes))
    if ids != expected_ids:
        raise ValueError(f"palette ids must be 0..{num_classes - 1}: actual={sorted(ids)}")
    return ids


def selected_indices(*, dataset_size: int, sample_count: int, seed: int) -> list[int]:
    if sample_count <= 0:
        raise ValueError(f"sample-count must be positive: {sample_count}")
    count = min(sample_count, dataset_size)
    ranked_indices = sorted(
        range(dataset_size),
        key=lambda index: hashlib.sha256(f"{seed}:{index}".encode("utf-8")).digest(),
    )
    return sorted(ranked_indices[:count])


def topk_indices(values: np.ndarray, *, k: int) -> list[int]:
    return np.argsort(values, axis=1)[:, ::-1][0, :k].astype(int).tolist()


def validate_session_outputs(
    *,
    logits_output: np.ndarray,
    top1_output: np.ndarray,
    num_classes: int,
) -> None:
    if list(logits_output.shape) != [1, num_classes]:
        raise ValueError(
            f"ONNX logits output shape must be [1, {num_classes}]: "
            f"actual={list(logits_output.shape)}"
        )
    if list(top1_output.shape) != [1]:
        raise ValueError(f"ONNX top1 output shape must be [1]: actual={list(top1_output.shape)}")
    if top1_output.dtype != np.int64:
        raise TypeError(f"ONNX top1 output dtype must be int64: actual={top1_output.dtype}")


def parity_failure(
    result: SampleParityResult,
    *,
    max_abs_diff_threshold: float,
    mean_abs_diff_threshold: float,
) -> bool:
    return (
        not result.top1_match
        or result.max_abs_diff > max_abs_diff_threshold
        or result.mean_abs_diff > mean_abs_diff_threshold
    )


def run_parity_validation(
    *,
    model: Any,
    dataset: TitleColorDataset,
    indices: list[int],
    logits_session: Any,
    top1_session: Any,
    num_classes: int,
    palette_ids: set[int],
    torch_module: Any,
) -> list[SampleParityResult]:
    results: list[SampleParityResult] = []
    model.eval()

    with torch_module.no_grad():
        for index in indices:
            sample = dataset[index]
            input_tensor = sample["x"].unsqueeze(0).float()
            if tuple(input_tensor.shape) != DEFAULT_INPUT_SHAPE:
                raise ValueError(
                    f"input tensor shape must be {DEFAULT_INPUT_SHAPE}: "
                    f"actual={tuple(input_tensor.shape)} image_id={sample['image_id']}"
                )

            pytorch_logits = model(input_tensor).detach().cpu().numpy().astype(np.float32)
            input_array = input_tensor.detach().cpu().numpy().astype(np.float32)
            onnx_logits = logits_session.run(
                [LOGITS_OUTPUT_NAME],
                {"input": input_array},
            )[0]
            onnx_top1 = top1_session.run(
                [TOP1_OUTPUT_NAME],
                {"input": input_array},
            )[0]
            validate_session_outputs(
                logits_output=onnx_logits,
                top1_output=onnx_top1,
                num_classes=num_classes,
            )

            pytorch_top1 = int(np.argmax(pytorch_logits, axis=1)[0])
            onnx_logits_top1 = int(np.argmax(onnx_logits, axis=1)[0])
            onnx_top1_index = int(onnx_top1[0])
            for top1_index in (pytorch_top1, onnx_logits_top1, onnx_top1_index):
                if top1_index not in palette_ids:
                    raise ValueError(
                        f"top1 index out of palette ids: {top1_index} "
                        f"image_id={sample['image_id']}"
                    )

            diff = np.abs(pytorch_logits - onnx_logits)
            results.append(
                SampleParityResult(
                    image_id=str(sample["image_id"]),
                    pytorch_top1=pytorch_top1,
                    onnx_logits_top1=onnx_logits_top1,
                    onnx_top1=onnx_top1_index,
                    top3_match=topk_indices(pytorch_logits, k=3)
                    == topk_indices(onnx_logits, k=3),
                    top5_match=topk_indices(pytorch_logits, k=5)
                    == topk_indices(onnx_logits, k=5),
                    max_abs_diff=float(diff.max()),
                    mean_abs_diff=float(diff.mean()),
                )
            )
    return results


def summarize_results(
    *,
    split: str,
    seed: int,
    results: list[SampleParityResult],
    max_abs_diff_threshold: float,
    mean_abs_diff_threshold: float,
) -> ParityMetrics:
    if not results:
        raise ValueError("at least one parity result is required")
    sample_count = len(results)
    max_abs_diff = max(result.max_abs_diff for result in results)
    mean_abs_diff = float(np.mean([result.mean_abs_diff for result in results]))
    top1_match_count = sum(result.top1_match for result in results)
    top1_agreement = top1_match_count / sample_count
    top3_agreement = sum(result.top3_match for result in results) / sample_count
    top5_agreement = sum(result.top5_match for result in results) / sample_count
    failed_results = [
        result
        for result in results
        if parity_failure(
            result,
            max_abs_diff_threshold=max_abs_diff_threshold,
            mean_abs_diff_threshold=mean_abs_diff_threshold,
        )
    ]
    return ParityMetrics(
        split=split,
        sample_count=sample_count,
        seed=seed,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        top1_agreement=top1_agreement,
        top3_agreement=top3_agreement,
        top5_agreement=top5_agreement,
        max_abs_diff_threshold=max_abs_diff_threshold,
        mean_abs_diff_threshold=mean_abs_diff_threshold,
        passed=len(failed_results) == 0 and top1_match_count == sample_count,
        failure_count=len(failed_results),
    )


def write_metrics_json(
    *,
    path: Path,
    metrics: ParityMetrics,
    results: list[SampleParityResult],
    inputs: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": asdict(metrics),
        "inputs": dict(inputs),
        "samples": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def format_ratio(value: float) -> str:
    return f"{value * 100:.2f}%"


def write_report(
    *,
    path: Path,
    metrics: ParityMetrics,
    results: list[SampleParityResult],
    inputs: Mapping[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [
        result
        for result in results
        if parity_failure(
            result,
            max_abs_diff_threshold=metrics.max_abs_diff_threshold,
            mean_abs_diff_threshold=metrics.mean_abs_diff_threshold,
        )
    ]
    lines = [
        "# TitLeNet PyTorch-ONNX Parity Report",
        "",
        "## Summary",
        "",
        f"- split: `{metrics.split}`",
        f"- samples: `{metrics.sample_count}`",
        f"- seed: `{metrics.seed}`",
        f"- passed: `{metrics.passed}`",
        "",
        "| metric | value | threshold |",
        "| --- | ---: | ---: |",
        f"| top1_agreement | {format_ratio(metrics.top1_agreement)} | 100.00% |",
        f"| top3_agreement | {format_ratio(metrics.top3_agreement)} | - |",
        f"| top5_agreement | {format_ratio(metrics.top5_agreement)} | - |",
        f"| max_abs_diff | {metrics.max_abs_diff:.8g} | {metrics.max_abs_diff_threshold:.8g} |",
        f"| mean_abs_diff | {metrics.mean_abs_diff:.8g} | {metrics.mean_abs_diff_threshold:.8g} |",
        "",
        "## Inputs",
        "",
        "| item | path |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | `{value}` |" for key, value in inputs.items())
    lines.extend(
        [
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        lines.extend(
            [
                "| image_id | pytorch_top1 | onnx_logits_top1 | onnx_top1 | max_abs_diff | mean_abs_diff |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        lines.extend(
            (
                f"| `{result.image_id}` | {result.pytorch_top1} | "
                f"{result.onnx_logits_top1} | {result.onnx_top1} | "
                f"{result.max_abs_diff:.8g} | {result.mean_abs_diff:.8g} |"
            )
            for result in failures
        )
    else:
        lines.append("No failures.")

    lines.extend(
        [
            "",
            "## Sample Preview",
            "",
            "| image_id | top1 | max_abs_diff | mean_abs_diff | top3_match | top5_match |",
            "| --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    lines.extend(
        (
            f"| `{result.image_id}` | {result.pytorch_top1} | "
            f"{result.max_abs_diff:.8g} | {result.mean_abs_diff:.8g} | "
            f"{result.top3_match} | {result.top5_match} |"
        )
        for result in results[:20]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_inputs_summary(
    *,
    checkpoint_path: Path,
    logits_onnx_path: Path,
    top1_onnx_path: Path,
    data_root: Path,
    palette_path: Path,
    report_path: Path,
    metrics_path: Path,
) -> dict[str, str]:
    return {
        "checkpoint": display_path(checkpoint_path),
        "logits_onnx": display_path(logits_onnx_path),
        "top1_onnx": display_path(top1_onnx_path),
        "data_root": display_path(data_root),
        "palette": display_path(palette_path),
        "report": display_path(report_path),
        "metrics": display_path(metrics_path),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import onnxruntime as ort
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch and ONNX Runtime are required for TitLeNet parity validation. "
            "Run this script in the title color training/export environment."
        ) from exc

    checkpoint_path = resolve_inside_project(
        PROJECT_ROOT,
        args.checkpoint,
        must_exist=True,
        description="checkpoint",
    )
    logits_onnx_path = resolve_inside_project(
        PROJECT_ROOT,
        args.logits_onnx,
        must_exist=True,
        description="logits onnx",
    )
    top1_onnx_path = resolve_inside_project(
        PROJECT_ROOT,
        args.top1_onnx,
        must_exist=True,
        description="top1 onnx",
    )
    data_root = resolve_inside_project(
        PROJECT_ROOT,
        args.data_root,
        must_exist=True,
        description="data root",
    )
    labels_matrix = optional_project_path(
        args.labels_matrix,
        must_exist=True,
        description="labels matrix",
    )
    labels_soft = optional_project_path(
        args.labels_soft,
        must_exist=True,
        description="labels soft",
    )
    palette_path = resolve_inside_project(
        PROJECT_ROOT,
        args.palette,
        must_exist=True,
        description="palette",
    )
    report_path = resolve_inside_project(
        PROJECT_ROOT,
        args.report_path,
        must_exist=False,
        description="report",
    )
    metrics_path = resolve_inside_project(
        PROJECT_ROOT,
        args.metrics_path,
        must_exist=False,
        description="metrics",
    )

    checkpoint = load_checkpoint(checkpoint_path, torch)
    model_config = model_config_from_checkpoint(
        checkpoint,
        model_name=None,
        dropout=None,
        weight_init=None,
        activation=None,
    )
    model = build_model(checkpoint, model_config)
    dataset_kwargs: dict[str, Any] = {"augment": False}
    if labels_matrix is not None:
        dataset_kwargs["labels_matrix_path"] = labels_matrix
    if labels_soft is not None:
        dataset_kwargs["labels_soft_path"] = labels_soft
    dataset = TitleColorDataset(
        args.split,
        data_root=data_root,
        project_root=PROJECT_ROOT,
        **dataset_kwargs,
    )
    indices = selected_indices(
        dataset_size=len(dataset),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    palette_ids = load_palette_ids(palette_path, num_classes=DEFAULT_NUM_CLASSES)
    logits_session = ort.InferenceSession(
        str(logits_onnx_path),
        providers=["CPUExecutionProvider"],
    )
    top1_session = ort.InferenceSession(
        str(top1_onnx_path),
        providers=["CPUExecutionProvider"],
    )
    results = run_parity_validation(
        model=model,
        dataset=dataset,
        indices=indices,
        logits_session=logits_session,
        top1_session=top1_session,
        num_classes=DEFAULT_NUM_CLASSES,
        palette_ids=palette_ids,
        torch_module=torch,
    )
    metrics = summarize_results(
        split=args.split,
        seed=args.seed,
        results=results,
        max_abs_diff_threshold=args.max_abs_diff_threshold,
        mean_abs_diff_threshold=args.mean_abs_diff_threshold,
    )
    inputs = build_inputs_summary(
        checkpoint_path=checkpoint_path,
        logits_onnx_path=logits_onnx_path,
        top1_onnx_path=top1_onnx_path,
        data_root=data_root,
        palette_path=palette_path,
        report_path=report_path,
        metrics_path=metrics_path,
    )
    write_metrics_json(
        path=metrics_path,
        metrics=metrics,
        results=results,
        inputs=inputs,
    )
    write_report(
        path=report_path,
        metrics=metrics,
        results=results,
        inputs=inputs,
    )

    print(f"Parity validation passed: {metrics.passed}")
    print(f"Samples: {metrics.sample_count}")
    print(f"Top-1 agreement: {format_ratio(metrics.top1_agreement)}")
    print(f"Max abs diff: {metrics.max_abs_diff:.8g}")
    print(f"Mean abs diff: {metrics.mean_abs_diff:.8g}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote metrics: {metrics_path}")
    if not metrics.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
