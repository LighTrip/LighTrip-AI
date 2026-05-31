from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from experiments.title_color_recommendation.run_model_comparison import (
    measure_latency,
    model_size_mb,
)
from experiments.title_color_recommendation.validate_titlenet_onnx import (
    DEFAULT_MAX_ABS_DIFF_THRESHOLD,
    DEFAULT_MEAN_ABS_DIFF_THRESHOLD,
    build_inputs_summary,
    load_palette_ids,
    run_parity_validation,
    selected_indices,
    summarize_results,
    write_metrics_json as write_parity_metrics_json,
    write_report as write_parity_report,
)
from scripts.title_color_recommendation.export_titlenet_onnx import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_OPSET,
    LOGITS_OUTPUT_NAME,
    TOP1_OUTPUT_NAME,
    build_model,
    export_paths,
    export_checkpoint_from_args,
    load_checkpoint,
    make_onnxruntime_sessions,
    model_config_from_checkpoint,
)
from src.models.fixed_palette_classifier import (
    count_total_parameters,
    count_trainable_parameters,
)
from src.title_color_recommendation.data.dataset import TitleColorDataset


MODEL_ID = "titlenet_student_warm_kd90"
MODEL_LABEL = "TitLeNet Student warm_start kd_90_10"
DEFAULT_CHECKPOINT = Path(
    "outputs/checkpoints/titlenet_student_kd_weight_sweep/"
    "warm_start/kd_90_10/checkpoint_best.pt"
)
DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_PALETTE = Path("data/title_color_recommendation/processed/palette.json")
DEFAULT_OUTPUT_DIR = Path("outputs/title_color_recommendation/onnx")
DEFAULT_LOGITS_ONNX = DEFAULT_OUTPUT_DIR / f"{MODEL_ID}_logits.onnx"
DEFAULT_TOP1_ONNX = DEFAULT_OUTPUT_DIR / f"{MODEL_ID}_top1.onnx"
DEFAULT_EXPORT_SUMMARY = DEFAULT_OUTPUT_DIR / f"{MODEL_ID}_onnx_export_summary.json"
DEFAULT_PARITY_REPORT = Path(
    f"outputs/reports/model_evaluation/onnx/{MODEL_ID}_parity_report.md"
)
DEFAULT_PARITY_METRICS = Path(
    f"outputs/reports/model_evaluation/onnx/{MODEL_ID}_parity_metrics.json"
)
BASELINE_REPORT_DISPLAY = (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_warm_kd90_baseline_report.md"
)
BASELINE_METRICS_DISPLAY = (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_warm_kd90_baseline_metrics.json"
)
DEFAULT_REFERENCE_METRICS = Path(
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep/"
    "warm_start/kd_90_10/metrics.json"
)
DEFAULT_CALIBRATION_DIR = Path(
    f"outputs/title_color_recommendation/quantization/calibration_samples/{MODEL_ID}"
)
DEFAULT_CALIBRATION_MANIFEST = Path(
    f"outputs/title_color_recommendation/quantization/{MODEL_ID}_calibration_manifest.json"
)
DEFAULT_PARITY_SAMPLE_COUNT = 100
DEFAULT_CALIBRATION_SAMPLE_COUNT = 200
DEFAULT_SEED = 42


@dataclass(frozen=True)
class CalibrationSummary:
    manifest_path: str
    sample_dir: str
    split: str
    sample_count: int
    seed: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the best TitLeNet Student checkpoint to FP32 ONNX and "
            "prepare the fixed pre-quantization baseline artifacts."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-init", default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-onnx-check", action="store_true")
    parser.add_argument("--skip-onnxruntime-check", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--parity-split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--parity-sample-count",
        type=int,
        default=DEFAULT_PARITY_SAMPLE_COUNT,
    )
    parser.add_argument("--parity-seed", type=int, default=DEFAULT_SEED)
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
    parser.add_argument("--calibration-split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--calibration-sample-count",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLE_COUNT,
    )
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-warmup-steps", type=int, default=10)
    parser.add_argument("--latency-benchmark-steps", type=int, default=50)
    return parser.parse_args(argv)


def display_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_optional_path(
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


def dataset_kwargs(
    *,
    labels_matrix: Path | None,
    labels_soft: Path | None,
) -> dict[str, Path]:
    kwargs: dict[str, Path] = {}
    if labels_matrix is not None:
        kwargs["labels_matrix_path"] = labels_matrix
    if labels_soft is not None:
        kwargs["labels_soft_path"] = labels_soft
    return kwargs


def make_dataset(
    *,
    split: str,
    data_root: Path,
    labels_matrix: Path | None,
    labels_soft: Path | None,
) -> TitleColorDataset:
    return TitleColorDataset(
        split,
        data_root=data_root,
        project_root=PROJECT_ROOT,
        augment=False,
        **dataset_kwargs(labels_matrix=labels_matrix, labels_soft=labels_soft),
    )


def validate_input_array(array: np.ndarray, *, image_id: str) -> None:
    if list(array.shape) != list(DEFAULT_INPUT_SHAPE):
        raise ValueError(
            f"calibration input shape must be {list(DEFAULT_INPUT_SHAPE)}: "
            f"actual={list(array.shape)} image_id={image_id}"
        )
    if array.dtype != np.float32:
        raise TypeError(f"calibration input dtype must be float32: actual={array.dtype}")
    rgb = array[:, :3]
    mask = array[:, 3:]
    if float(rgb.min()) < 0.0 or float(rgb.max()) > 1.0:
        raise ValueError(f"RGB range must stay within 0..1: image_id={image_id}")
    if not np.isin(mask, [0.0, 1.0]).all():
        raise ValueError(f"mask channel must contain only 0/1: image_id={image_id}")


def write_calibration_samples(
    *,
    dataset: TitleColorDataset,
    indices: list[int],
    sample_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    seed: int,
) -> CalibrationSummary:
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for order, dataset_index in enumerate(indices):
        sample = dataset[dataset_index]
        image_id = str(sample["image_id"])
        input_array = sample["x"].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
        validate_input_array(input_array, image_id=image_id)
        sample_path = sample_dir / f"sample_{order:04d}.npy"
        np.save(sample_path, input_array)
        entries.append(
            {
                "order": order,
                "dataset_index": int(dataset_index),
                "image_id": image_id,
                "path": display_path(sample_path),
                "shape": list(input_array.shape),
                "dtype": str(input_array.dtype),
            }
        )

    payload = {
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "checkpoint": display_path(checkpoint_path),
        "split": split,
        "seed": seed,
        "sample_count": len(entries),
        "input": {
            "shape": list(DEFAULT_INPUT_SHAPE),
            "dtype": "float32",
            "channel_order": ["R", "G", "B", "mask"],
            "rgb_range": [0.0, 1.0],
            "mask_range": [0.0, 1.0],
        },
        "samples": entries,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return CalibrationSummary(
        manifest_path=display_path(manifest_path),
        sample_dir=display_path(sample_dir),
        split=split,
        sample_count=len(entries),
        seed=seed,
    )


def path_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def load_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return payload


def benchmark_onnx_session(
    *,
    session: Any,
    output_name: str,
    input_array: np.ndarray,
    warmup_steps: int,
    benchmark_steps: int,
) -> dict[str, float]:
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative: {warmup_steps}")
    if benchmark_steps <= 0:
        raise ValueError(f"benchmark_steps must be positive: {benchmark_steps}")
    for _index in range(warmup_steps):
        session.run([output_name], {"input": input_array})
    start = time.perf_counter()
    for _index in range(benchmark_steps):
        session.run([output_name], {"input": input_array})
    elapsed = time.perf_counter() - start
    batch_size = int(input_array.shape[0])
    return {
        "inference_time_ms": (elapsed * 1000.0) / benchmark_steps,
        "images_per_second": (batch_size * benchmark_steps) / max(elapsed, 1e-12),
    }


def benchmark_fp32_baseline(
    *,
    model: Any,
    logits_onnx_path: Path,
    torch_module: Any,
    warmup_steps: int,
    benchmark_steps: int,
    device_name: str,
    skip_latency: bool,
) -> Mapping[str, Any]:
    if skip_latency:
        return {"checked": False, "reason": "latency benchmark skipped"}
    import onnxruntime as ort

    device = torch_module.device(device_name)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA latency requested, but CUDA is not available")

    pytorch_latency = {
        "batch1": measure_latency(
            model,
            device=device,
            batch_size=1,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        ),
        "batch64": measure_latency(
            model,
            device=device,
            batch_size=64,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        ),
    }
    input_array = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)
    logits_session = ort.InferenceSession(
        str(logits_onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnxruntime_latency = {
        "batch1_logits": benchmark_onnx_session(
            session=logits_session,
            output_name=LOGITS_OUTPUT_NAME,
            input_array=input_array,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        )
    }
    return {
        "checked": True,
        "pytorch_device": device.type,
        "onnxruntime_provider": "CPUExecutionProvider",
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "pytorch": pytorch_latency,
        "onnxruntime": onnxruntime_latency,
    }


def write_baseline_metrics(*, payload: Mapping[str, Any]) -> None:
    os.makedirs("outputs/reports/model_evaluation/onnx", exist_ok=True)
    with open(
        "outputs/reports/model_evaluation/onnx/"
        "titlenet_student_warm_kd90_baseline_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_baseline_report(*, payload: Mapping[str, Any]) -> None:
    os.makedirs("outputs/reports/model_evaluation/onnx", exist_ok=True)
    test_metrics = payload.get("reference_metrics", {}).get("test_metrics", {})
    teacher_agreement = payload.get("reference_metrics", {}).get(
        "test_teacher_agreement",
        {},
    )
    parity = payload.get("parity_metrics") or {}
    latency = payload.get("latency") or {}
    onnx = payload["onnx"]
    model = payload["model"]
    lines = [
        "# TitLeNet Student FP32 ONNX Baseline",
        "",
        "## Summary",
        "",
        f"- model_id: `{payload['model_id']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- logits_onnx: `{onnx['logits']['path']}`",
        f"- top1_onnx: `{onnx['top1']['path']}`",
        f"- calibration_manifest: `{(payload.get('calibration') or {}).get('manifest_path', '-')}`",
        "",
        "## Model",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| model_name | `{model['name']}` |",
        f"| activation | `{model['activation']}` |",
        f"| total_parameters | {model['total_parameters']} |",
        f"| trainable_parameters | {model['trainable_parameters']} |",
        f"| pytorch_size_mb | {model['pytorch_size_mb']:.6f} |",
        f"| logits_onnx_size_mb | {onnx['logits']['size_mb']:.6f} |",
        f"| top1_onnx_size_mb | {onnx['top1']['size_mb']:.6f} |",
        "",
        "## Reference Test Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| NDCG@3 | {format_metric(test_metrics.get('val_ndcg@3'))} |",
        f"| NDCG@5 | {format_metric(test_metrics.get('val_ndcg@5'))} |",
        f"| loss | {format_metric(test_metrics.get('val_loss'))} |",
        f"| teacher_top1_agreement | {format_metric(teacher_agreement.get('teacher_top1_agreement'))} |",
        "",
        "## ONNX Parity",
        "",
        "| metric | value | threshold |",
        "| --- | ---: | ---: |",
        f"| passed | {parity.get('passed', '-')} | - |",
        f"| sample_count | {parity.get('sample_count', '-')} | - |",
        f"| top1_agreement | {format_metric(parity.get('top1_agreement'))} | 1.000000 |",
        f"| max_abs_diff | {format_metric(parity.get('max_abs_diff'))} | {format_metric(parity.get('max_abs_diff_threshold'))} |",
        f"| mean_abs_diff | {format_metric(parity.get('mean_abs_diff'))} | {format_metric(parity.get('mean_abs_diff_threshold'))} |",
        "",
        "## Latency",
        "",
    ]
    if latency.get("checked"):
        pytorch_batch1 = latency["pytorch"]["batch1"]["inference_time_ms"]
        pytorch_batch64 = latency["pytorch"]["batch64"]["inference_time_ms"]
        onnx_batch1 = latency["onnxruntime"]["batch1_logits"]["inference_time_ms"]
        lines.extend(
            [
                "| runtime | target | latency_ms |",
                "| --- | --- | ---: |",
                f"| PyTorch | batch1/{latency['pytorch_device']} | {pytorch_batch1:.6f} |",
                f"| PyTorch | batch64/{latency['pytorch_device']} | {pytorch_batch64:.6f} |",
                f"| ONNX Runtime | batch1/{latency['onnxruntime_provider']} | {onnx_batch1:.6f} |",
            ]
        )
    else:
        lines.append(f"- skipped: `{latency.get('reason', 'unknown')}`")
    with open(
        "outputs/reports/model_evaluation/onnx/"
        "titlenet_student_warm_kd90_baseline_report.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import onnxruntime as ort
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch and ONNX Runtime are required for Student ONNX baseline preparation."
        ) from exc

    checkpoint_path = resolve_inside_project(
        PROJECT_ROOT,
        args.checkpoint,
        must_exist=True,
        description="checkpoint",
    )
    data_root = resolve_inside_project(
        PROJECT_ROOT,
        args.data_root,
        must_exist=True,
        description="data root",
    )
    labels_matrix = resolve_optional_path(
        args.labels_matrix,
        must_exist=True,
        description="labels matrix",
    )
    labels_soft = resolve_optional_path(
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
    parity_report = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_PARITY_REPORT,
        must_exist=False,
        description="parity report",
    )
    parity_metrics_path = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_PARITY_METRICS,
        must_exist=False,
        description="parity metrics",
    )
    baseline_report = BASELINE_REPORT_DISPLAY
    calibration_dir = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_CALIBRATION_DIR,
        must_exist=False,
        description="calibration dir",
    )
    calibration_manifest = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_CALIBRATION_MANIFEST,
        must_exist=False,
        description="calibration manifest",
    )
    reference_metrics_path = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_REFERENCE_METRICS,
        must_exist=False,
        description="reference metrics",
    )

    output_dir = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_OUTPUT_DIR,
        must_exist=False,
        description="output dir",
    )
    logits_onnx = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_LOGITS_ONNX,
        must_exist=False,
        description="logits onnx",
    )
    top1_onnx = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_TOP1_ONNX,
        must_exist=False,
        description="top1 onnx",
    )
    export_summary = resolve_inside_project(
        PROJECT_ROOT,
        DEFAULT_EXPORT_SUMMARY,
        must_exist=False,
        description="export summary",
    )
    paths = export_paths(
        output_dir=output_dir,
        logits_output=logits_onnx,
        top1_output=top1_onnx,
        summary_output=export_summary,
    )
    if not args.skip_export:
        export_result = export_checkpoint_from_args(
            args=args,
            paths=paths,
            checkpoint_path=checkpoint_path,
            torch_module=torch,
        )
        config = export_result.config
        model = export_result.model
    else:
        checkpoint = load_checkpoint(checkpoint_path, torch)
        config = model_config_from_checkpoint(
            checkpoint,
            model_name=args.model_name,
            dropout=args.dropout,
            weight_init=args.weight_init,
            activation=args.activation,
        )
        model = build_model(checkpoint, config)

    if not paths.logits_output.exists() or not paths.top1_output.exists():
        raise FileNotFoundError(
            "Student ONNX files are required; run without --skip-export first."
        )

    parity_metrics = None
    if not args.skip_parity:
        parity_dataset = make_dataset(
            split=args.parity_split,
            data_root=data_root,
            labels_matrix=labels_matrix,
            labels_soft=labels_soft,
        )
        parity_indices = selected_indices(
            dataset_size=len(parity_dataset),
            sample_count=args.parity_sample_count,
            seed=args.parity_seed,
        )
        palette_ids = load_palette_ids(palette_path, num_classes=DEFAULT_NUM_CLASSES)
        logits_session, top1_session = make_onnxruntime_sessions(paths, ort)
        parity_results = run_parity_validation(
            model=model.cpu(),
            dataset=parity_dataset,
            indices=parity_indices,
            logits_session=logits_session,
            top1_session=top1_session,
            num_classes=DEFAULT_NUM_CLASSES,
            palette_ids=palette_ids,
            torch_module=torch,
        )
        parity_metrics = summarize_results(
            split=args.parity_split,
            seed=args.parity_seed,
            results=parity_results,
            max_abs_diff_threshold=args.max_abs_diff_threshold,
            mean_abs_diff_threshold=args.mean_abs_diff_threshold,
        )
        parity_inputs = build_inputs_summary(
            checkpoint_path=checkpoint_path,
            logits_onnx_path=paths.logits_output,
            top1_onnx_path=paths.top1_output,
            data_root=data_root,
            palette_path=palette_path,
            report_path=parity_report,
            metrics_path=parity_metrics_path,
        )
        write_parity_metrics_json(
            path=parity_metrics_path,
            metrics=parity_metrics,
            results=parity_results,
            inputs=parity_inputs,
        )
        write_parity_report(
            path=parity_report,
            metrics=parity_metrics,
            results=parity_results,
            inputs=parity_inputs,
        )

    calibration_summary = None
    if not args.skip_calibration:
        calibration_dataset = make_dataset(
            split=args.calibration_split,
            data_root=data_root,
            labels_matrix=labels_matrix,
            labels_soft=labels_soft,
        )
        calibration_indices = selected_indices(
            dataset_size=len(calibration_dataset),
            sample_count=args.calibration_sample_count,
            seed=args.calibration_seed,
        )
        calibration_summary = write_calibration_samples(
            dataset=calibration_dataset,
            indices=calibration_indices,
            sample_dir=calibration_dir,
            manifest_path=calibration_manifest,
            checkpoint_path=checkpoint_path,
            split=args.calibration_split,
            seed=args.calibration_seed,
        )

    latency = benchmark_fp32_baseline(
        model=model.cpu(),
        logits_onnx_path=paths.logits_output,
        torch_module=torch,
        warmup_steps=args.latency_warmup_steps,
        benchmark_steps=args.latency_benchmark_steps,
        device_name=args.device,
        skip_latency=args.skip_latency,
    )
    reference_metrics = load_json_if_exists(reference_metrics_path)
    baseline_payload = {
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "checkpoint": display_path(checkpoint_path),
        "model": {
            "name": config.model_name,
            "num_classes": config.num_classes,
            "dropout": config.dropout,
            "weight_init": config.weight_init,
            "activation": config.activation,
            "total_parameters": count_total_parameters(model),
            "trainable_parameters": count_trainable_parameters(model),
            "pytorch_size_mb": model_size_mb(model),
        },
        "input": {
            "shape": list(DEFAULT_INPUT_SHAPE),
            "dtype": "float32",
            "channel_order": ["R", "G", "B", "mask"],
        },
        "onnx": {
            "opset": args.opset,
            "logits": {
                "path": display_path(paths.logits_output),
                "output_name": LOGITS_OUTPUT_NAME,
                "shape": [1, config.num_classes],
                "dtype": "float32",
                "size_mb": path_size_mb(paths.logits_output),
            },
            "top1": {
                "path": display_path(paths.top1_output),
                "output_name": TOP1_OUTPUT_NAME,
                "shape": [1],
                "dtype": "int64",
                "size_mb": path_size_mb(paths.top1_output),
            },
        },
        "parity_metrics": asdict(parity_metrics) if parity_metrics is not None else None,
        "calibration": asdict(calibration_summary)
        if calibration_summary is not None
        else None,
        "latency": dict(latency),
        "reference_metrics": dict(reference_metrics),
        "artifacts": {
            "export_summary": display_path(paths.summary_output),
            "parity_report": display_path(parity_report),
            "parity_metrics": display_path(parity_metrics_path),
            "baseline_report": baseline_report,
            "baseline_metrics": BASELINE_METRICS_DISPLAY,
        },
    }
    write_baseline_metrics(payload=baseline_payload)
    write_baseline_report(payload=baseline_payload)

    print(f"Prepared {MODEL_LABEL}")
    print(f"Logits ONNX: {paths.logits_output}")
    print(f"Top1 ONNX: {paths.top1_output}")
    print(f"Baseline report: {baseline_report}")
    if parity_metrics is not None:
        print(f"Parity passed: {parity_metrics.passed}")
        if not parity_metrics.passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
