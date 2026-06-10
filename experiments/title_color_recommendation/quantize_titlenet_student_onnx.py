from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.prepare_titlenet_student_quantization_baseline import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LOGITS_ONNX,
    DEFAULT_PALETTE,
    DEFAULT_TOP1_ONNX,
    MODEL_ID,
    MODEL_LABEL,
    benchmark_onnx_session,
    path_size_mb,
)
from experiments.title_color_recommendation.validate_titlenet_onnx import (
    selected_indices,
)
from scripts.title_color_recommendation.export_titlenet_onnx import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    LOGITS_OUTPUT_NAME,
    TOP1_OUTPUT_NAME,
)
from src.title_color_recommendation.data.dataset import TitleColorDataset


TRIAL_FP16 = "fp16"
TRIAL_INT8_DYNAMIC = "int8_dynamic"
TRIAL_INT8_STATIC = "int8_static"
TRIAL_INT8_STATIC_PER_CHANNEL = "int8_static_per_channel"
TRIAL_INT4_WEIGHT_ONLY = "int4_weight_only"
DEFAULT_TRIALS = (
    TRIAL_FP16,
    TRIAL_INT8_DYNAMIC,
    TRIAL_INT8_STATIC,
    TRIAL_INT8_STATIC_PER_CHANNEL,
    TRIAL_INT4_WEIGHT_ONLY,
)
DEFAULT_OUTPUT_DIR = Path("outputs/title_color_recommendation/quantization")
DEFAULT_REPORT_PATH = Path(
    f"outputs/reports/model_evaluation/onnx/{MODEL_ID}_quantization_report.md"
)
DEFAULT_METRICS_PATH = Path(
    f"outputs/reports/model_evaluation/onnx/{MODEL_ID}_quantization_metrics.json"
)
DEFAULT_SPLIT = "test"
DEFAULT_CALIBRATION_SPLIT = "val"
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_CALIBRATION_SAMPLE_COUNT = 200
DEFAULT_SEED = 42
DEFAULT_MIN_TOP1_AGREEMENT = 0.98
DEFAULT_MAX_NDCG5_DROP = 0.005


@dataclass(frozen=True)
class QuantizationTrial:
    name: str
    precision: str
    method: str
    per_channel: bool = False


@dataclass(frozen=True)
class QuantizationValidation:
    checked: bool
    sample_count: int
    logits_top1_agreement: float | None
    top1_model_agreement: float | None
    top3_agreement: float | None
    top5_agreement: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    fp32_ndcg_at_3: float | None
    fp32_ndcg_at_5: float | None
    quantized_ndcg_at_3: float | None
    quantized_ndcg_at_5: float | None
    ndcg_at_3_drop: float | None
    ndcg_at_5_drop: float | None
    valid_top1_range: bool
    passed: bool
    failure_count: int
    reason: str | None = None


@dataclass(frozen=True)
class QuantizationResult:
    name: str
    precision: str
    method: str
    status: str
    logits_path: str | None
    top1_path: str | None
    logits_size_mb: float | None
    top1_size_mb: float | None
    latency: Mapping[str, Any]
    validation: QuantizationValidation | None
    reason: str | None = None


@dataclass(frozen=True)
class QuantizationValidationContext:
    fp32_logits_session: Any
    fp32_top1_session: Any
    dataset: TitleColorDataset
    indices: list[int]
    min_top1_agreement: float
    max_ndcg5_drop: float


class NumpyCalibrationDataReader:
    def __init__(self, arrays: Iterable[np.ndarray]) -> None:
        self._arrays = [array.astype(np.float32, copy=False) for array in arrays]
        self.rewind()

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            return {"input": next(self._iterator)}
        except StopIteration:
            return None

    def rewind(self) -> None:
        self._iterator = iter(self._arrays)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FP16 and INT8 post-training quantization experiments for the "
            "best TitLeNet Student ONNX model."
        )
    )
    parser.add_argument("--logits-onnx", type=Path, default=DEFAULT_LOGITS_ONNX)
    parser.add_argument("--top1-onnx", type=Path, default=DEFAULT_TOP1_ONNX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE)
    parser.add_argument("--split", choices=("val", "test"), default=DEFAULT_SPLIT)
    parser.add_argument(
        "--calibration-split",
        choices=("train", "val"),
        default=DEFAULT_CALIBRATION_SPLIT,
    )
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument(
        "--calibration-sample-count",
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLE_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--trials",
        nargs="+",
        choices=DEFAULT_TRIALS,
        default=list(DEFAULT_TRIALS),
    )
    parser.add_argument(
        "--min-top1-agreement",
        type=float,
        default=DEFAULT_MIN_TOP1_AGREEMENT,
    )
    parser.add_argument("--max-ndcg5-drop", type=float, default=DEFAULT_MAX_NDCG5_DROP)
    parser.add_argument("--latency-warmup-steps", type=int, default=10)
    parser.add_argument("--latency-benchmark-steps", type=int, default=50)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--fail-on-regression", action="store_true")
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


def trial_from_name(name: str) -> QuantizationTrial:
    trials = {
        TRIAL_FP16: QuantizationTrial(
            name=TRIAL_FP16,
            precision="fp16",
            method="float16_conversion",
        ),
        TRIAL_INT8_DYNAMIC: QuantizationTrial(
            name=TRIAL_INT8_DYNAMIC,
            precision="int8",
            method="dynamic",
        ),
        TRIAL_INT8_STATIC: QuantizationTrial(
            name=TRIAL_INT8_STATIC,
            precision="int8",
            method="static_qdq",
        ),
        TRIAL_INT8_STATIC_PER_CHANNEL: QuantizationTrial(
            name=TRIAL_INT8_STATIC_PER_CHANNEL,
            precision="int8",
            method="static_qdq",
            per_channel=True,
        ),
        TRIAL_INT4_WEIGHT_ONLY: QuantizationTrial(
            name=TRIAL_INT4_WEIGHT_ONLY,
            precision="int4",
            method="weight_only",
        ),
    }
    return trials[name]


def output_path(output_dir: Path, *, trial_name: str, output_kind: str) -> Path:
    return output_dir / f"{MODEL_ID}_{trial_name}_{output_kind}.onnx"


def validate_input_array(array: np.ndarray, *, description: str) -> None:
    if list(array.shape) != list(DEFAULT_INPUT_SHAPE):
        raise ValueError(
            f"{description} shape must be {list(DEFAULT_INPUT_SHAPE)}: "
            f"actual={list(array.shape)}"
        )
    if array.dtype != np.float32:
        raise TypeError(f"{description} dtype must be float32: actual={array.dtype}")


def sample_arrays(dataset: TitleColorDataset, indices: list[int]) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for index in indices:
        sample = dataset[index]
        array = sample["x"].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
        validate_input_array(array, description=f"sample index {index}")
        arrays.append(array)
    return arrays


def convert_fp16_model(*, input_path: Path, output_path: Path) -> None:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = onnx.load(str(input_path))
    fp16_model = convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )
    onnx.save(fp16_model, str(output_path))


def quantize_dynamic_model(
    *,
    input_path: Path,
    output_path: Path,
    per_channel: bool,
) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=per_channel,
        weight_type=QuantType.QInt8,
    )


def quantize_static_model(
    *,
    input_path: Path,
    output_path: Path,
    calibration_reader_factory: Callable[[], NumpyCalibrationDataReader],
    per_channel: bool,
) -> None:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader_factory(),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )


def create_quantized_pair(
    *,
    trial: QuantizationTrial,
    logits_onnx: Path,
    top1_onnx: Path,
    output_dir: Path,
    calibration_reader_factory: Callable[[], NumpyCalibrationDataReader],
) -> tuple[Path | None, Path | None, str | None]:
    if trial.name == TRIAL_INT4_WEIGHT_ONLY:
        return (
            None,
            None,
            (
                "ONNX Runtime 1.23 quantization does not provide a Conv-friendly "
                "INT4 PTQ path for this TitLeNet Student model. Keep this as a "
                "research-only slot unless the mobile runtime supports INT4 Conv."
            ),
        )

    logits_output = output_path(output_dir, trial_name=trial.name, output_kind="logits")
    top1_output = output_path(output_dir, trial_name=trial.name, output_kind="top1")

    if trial.name == TRIAL_FP16:
        convert_fp16_model(input_path=logits_onnx, output_path=logits_output)
        convert_fp16_model(input_path=top1_onnx, output_path=top1_output)
    elif trial.name == TRIAL_INT8_DYNAMIC:
        quantize_dynamic_model(
            input_path=logits_onnx,
            output_path=logits_output,
            per_channel=trial.per_channel,
        )
        quantize_dynamic_model(
            input_path=top1_onnx,
            output_path=top1_output,
            per_channel=trial.per_channel,
        )
    else:
        quantize_static_model(
            input_path=logits_onnx,
            output_path=logits_output,
            calibration_reader_factory=calibration_reader_factory,
            per_channel=trial.per_channel,
        )
        quantize_static_model(
            input_path=top1_onnx,
            output_path=top1_output,
            calibration_reader_factory=calibration_reader_factory,
            per_channel=trial.per_channel,
        )

    return logits_output, top1_output, None


def topk_indices(values: np.ndarray, *, k: int) -> list[int]:
    return np.argsort(values, axis=1)[:, ::-1][0, :k].astype(int).tolist()


def ndcg_at_k_numpy(logits: np.ndarray, target_distribution: np.ndarray, *, k: int) -> float:
    scores = logits.reshape(-1)
    targets = target_distribution.reshape(-1).astype(np.float64)
    order = np.argsort(scores)[::-1][:k]
    ideal_order = np.argsort(targets)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(targets[order] * discounts))
    ideal_dcg = float(np.sum(targets[ideal_order] * discounts))
    if ideal_dcg <= 0.0:
        return 0.0
    return dcg / ideal_dcg


def validate_session_outputs(
    *,
    logits_output: np.ndarray,
    top1_output: np.ndarray,
) -> None:
    if list(logits_output.shape) != [1, DEFAULT_NUM_CLASSES]:
        raise ValueError(
            f"logits output shape must be [1, {DEFAULT_NUM_CLASSES}]: "
            f"actual={list(logits_output.shape)}"
        )
    if list(top1_output.shape) != [1]:
        raise ValueError(f"top1 output shape must be [1]: actual={list(top1_output.shape)}")
    if top1_output.dtype != np.int64:
        raise TypeError(f"top1 output dtype must be int64: actual={top1_output.dtype}")


def run_onnx(
    session: Any,
    output_name: str,
    input_array: np.ndarray,
) -> np.ndarray:
    return session.run([output_name], {"input": input_array})[0]


def validate_quantized_pair(
    *,
    fp32_logits_session: Any,
    fp32_top1_session: Any,
    quantized_logits_session: Any,
    quantized_top1_session: Any,
    dataset: TitleColorDataset,
    indices: list[int],
    min_top1_agreement: float,
    max_ndcg5_drop: float,
) -> QuantizationValidation:
    if not indices:
        raise ValueError("at least one validation sample is required")

    max_abs_diff = 0.0
    mean_abs_diffs: list[float] = []
    fp32_ndcg3_values: list[float] = []
    fp32_ndcg5_values: list[float] = []
    quantized_ndcg3_values: list[float] = []
    quantized_ndcg5_values: list[float] = []
    logits_top1_matches = 0
    top1_model_matches = 0
    top3_matches = 0
    top5_matches = 0
    valid_top1_range = True

    for index in indices:
        sample = dataset[index]
        input_array = sample["x"].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
        target_distribution = (
            sample["target_distribution"].detach().cpu().numpy().astype(np.float32)
        )
        fp32_logits = run_onnx(fp32_logits_session, LOGITS_OUTPUT_NAME, input_array)
        fp32_top1 = run_onnx(fp32_top1_session, TOP1_OUTPUT_NAME, input_array)
        quantized_logits = run_onnx(
            quantized_logits_session,
            LOGITS_OUTPUT_NAME,
            input_array,
        )
        quantized_top1 = run_onnx(
            quantized_top1_session,
            TOP1_OUTPUT_NAME,
            input_array,
        )
        validate_session_outputs(
            logits_output=quantized_logits,
            top1_output=quantized_top1,
        )

        fp32_logits_top1 = int(np.argmax(fp32_logits, axis=1)[0])
        fp32_top1_index = int(fp32_top1[0])
        quantized_logits_top1 = int(np.argmax(quantized_logits, axis=1)[0])
        quantized_top1_index = int(quantized_top1[0])
        if not 0 <= quantized_top1_index < DEFAULT_NUM_CLASSES:
            valid_top1_range = False

        logits_top1_matches += int(fp32_logits_top1 == quantized_logits_top1)
        top1_model_matches += int(fp32_top1_index == quantized_top1_index)
        top3_matches += int(
            topk_indices(fp32_logits, k=3) == topk_indices(quantized_logits, k=3)
        )
        top5_matches += int(
            topk_indices(fp32_logits, k=5) == topk_indices(quantized_logits, k=5)
        )
        diff = np.abs(fp32_logits.astype(np.float32) - quantized_logits.astype(np.float32))
        max_abs_diff = max(max_abs_diff, float(diff.max()))
        mean_abs_diffs.append(float(diff.mean()))
        fp32_ndcg3_values.append(
            ndcg_at_k_numpy(fp32_logits, target_distribution, k=3)
        )
        fp32_ndcg5_values.append(
            ndcg_at_k_numpy(fp32_logits, target_distribution, k=5)
        )
        quantized_ndcg3_values.append(
            ndcg_at_k_numpy(quantized_logits, target_distribution, k=3)
        )
        quantized_ndcg5_values.append(
            ndcg_at_k_numpy(quantized_logits, target_distribution, k=5)
        )

    sample_count = len(indices)
    logits_top1_agreement = logits_top1_matches / sample_count
    top1_model_agreement = top1_model_matches / sample_count
    top3_agreement = top3_matches / sample_count
    top5_agreement = top5_matches / sample_count
    fp32_ndcg_at_3 = float(np.mean(fp32_ndcg3_values))
    fp32_ndcg_at_5 = float(np.mean(fp32_ndcg5_values))
    quantized_ndcg_at_3 = float(np.mean(quantized_ndcg3_values))
    quantized_ndcg_at_5 = float(np.mean(quantized_ndcg5_values))
    ndcg_at_3_drop = fp32_ndcg_at_3 - quantized_ndcg_at_3
    ndcg_at_5_drop = fp32_ndcg_at_5 - quantized_ndcg_at_5
    failure_count = sample_count - top1_model_matches
    passed = (
        valid_top1_range
        and top1_model_agreement >= min_top1_agreement
        and ndcg_at_5_drop <= max_ndcg5_drop
    )
    return QuantizationValidation(
        checked=True,
        sample_count=sample_count,
        logits_top1_agreement=logits_top1_agreement,
        top1_model_agreement=top1_model_agreement,
        top3_agreement=top3_agreement,
        top5_agreement=top5_agreement,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=float(np.mean(mean_abs_diffs)),
        fp32_ndcg_at_3=fp32_ndcg_at_3,
        fp32_ndcg_at_5=fp32_ndcg_at_5,
        quantized_ndcg_at_3=quantized_ndcg_at_3,
        quantized_ndcg_at_5=quantized_ndcg_at_5,
        ndcg_at_3_drop=ndcg_at_3_drop,
        ndcg_at_5_drop=ndcg_at_5_drop,
        valid_top1_range=valid_top1_range,
        passed=passed,
        failure_count=failure_count,
    )


def make_session(path: Path, ort_module: Any) -> Any:
    return ort_module.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )


def benchmark_quantized_logits(
    *,
    logits_path: Path,
    warmup_steps: int,
    benchmark_steps: int,
    skip_latency: bool,
) -> Mapping[str, Any]:
    if skip_latency:
        return {"checked": False, "reason": "latency benchmark skipped"}
    import onnxruntime as ort

    session = make_session(logits_path, ort)
    input_array = np.zeros(DEFAULT_INPUT_SHAPE, dtype=np.float32)
    return {
        "checked": True,
        "onnxruntime_provider": "CPUExecutionProvider",
        "batch1_logits": benchmark_onnx_session(
            session=session,
            output_name=LOGITS_OUTPUT_NAME,
            input_array=input_array,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        ),
    }


def skipped_validation(reason: str) -> QuantizationValidation:
    return QuantizationValidation(
        checked=False,
        sample_count=0,
        logits_top1_agreement=None,
        top1_model_agreement=None,
        top3_agreement=None,
        top5_agreement=None,
        max_abs_diff=None,
        mean_abs_diff=None,
        fp32_ndcg_at_3=None,
        fp32_ndcg_at_5=None,
        quantized_ndcg_at_3=None,
        quantized_ndcg_at_5=None,
        ndcg_at_3_drop=None,
        ndcg_at_5_drop=None,
        valid_top1_range=False,
        passed=False,
        failure_count=0,
        reason=reason,
    )


def run_trial(
    *,
    trial: QuantizationTrial,
    logits_onnx: Path,
    top1_onnx: Path,
    output_dir: Path,
    calibration_reader_factory: Callable[[], NumpyCalibrationDataReader],
    validation_context: QuantizationValidationContext,
    warmup_steps: int,
    benchmark_steps: int,
    skip_latency: bool,
) -> QuantizationResult:
    try:
        quantized_logits, quantized_top1, skip_reason = create_quantized_pair(
            trial=trial,
            logits_onnx=logits_onnx,
            top1_onnx=top1_onnx,
            output_dir=output_dir,
            calibration_reader_factory=calibration_reader_factory,
        )
        if skip_reason is not None:
            return QuantizationResult(
                name=trial.name,
                precision=trial.precision,
                method=trial.method,
                status="unsupported",
                logits_path=None,
                top1_path=None,
                logits_size_mb=None,
                top1_size_mb=None,
                latency={"checked": False, "reason": skip_reason},
                validation=skipped_validation(skip_reason),
                reason=skip_reason,
            )
        if quantized_logits is None or quantized_top1 is None:
            raise RuntimeError(f"{trial.name} did not create quantized ONNX files")

        import onnxruntime as ort

        quantized_logits_session = make_session(quantized_logits, ort)
        quantized_top1_session = make_session(quantized_top1, ort)
        validation = validate_quantized_pair(
            fp32_logits_session=validation_context.fp32_logits_session,
            fp32_top1_session=validation_context.fp32_top1_session,
            quantized_logits_session=quantized_logits_session,
            quantized_top1_session=quantized_top1_session,
            dataset=validation_context.dataset,
            indices=validation_context.indices,
            min_top1_agreement=validation_context.min_top1_agreement,
            max_ndcg5_drop=validation_context.max_ndcg5_drop,
        )
        latency = benchmark_quantized_logits(
            logits_path=quantized_logits,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
            skip_latency=skip_latency,
        )
        return QuantizationResult(
            name=trial.name,
            precision=trial.precision,
            method=trial.method,
            status="passed" if validation.passed else "regressed",
            logits_path=display_path(quantized_logits),
            top1_path=display_path(quantized_top1),
            logits_size_mb=path_size_mb(quantized_logits),
            top1_size_mb=path_size_mb(quantized_top1),
            latency=latency,
            validation=validation,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return QuantizationResult(
            name=trial.name,
            precision=trial.precision,
            method=trial.method,
            status="failed",
            logits_path=None,
            top1_path=None,
            logits_size_mb=None,
            top1_size_mb=None,
            latency={"checked": False, "reason": reason},
            validation=skipped_validation(reason),
            reason=reason,
        )


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def format_latency(latency: Mapping[str, Any]) -> str:
    if not latency.get("checked"):
        return "-"
    return format_metric(latency["batch1_logits"]["inference_time_ms"])


def result_to_row(result: QuantizationResult) -> list[str]:
    validation = result.validation
    return [
        result.name,
        result.precision,
        result.method,
        result.status,
        format_metric(result.logits_size_mb),
        format_metric(result.top1_size_mb),
        format_latency(result.latency),
        format_metric(validation.top1_model_agreement if validation else None),
        format_metric(validation.ndcg_at_3_drop if validation else None),
        format_metric(validation.ndcg_at_5_drop if validation else None),
        format_metric(validation.max_abs_diff if validation else None),
        result.reason or ((validation.reason if validation else None) or ""),
    ]


def write_metrics_json(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report(
    *,
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    results = [
        QuantizationResult(
            name=str(item["name"]),
            precision=str(item["precision"]),
            method=str(item["method"]),
            status=str(item["status"]),
            logits_path=item.get("logits_path"),
            top1_path=item.get("top1_path"),
            logits_size_mb=item.get("logits_size_mb"),
            top1_size_mb=item.get("top1_size_mb"),
            latency=item.get("latency", {}),
            validation=QuantizationValidation(**item["validation"])
            if item.get("validation")
            else None,
            reason=item.get("reason"),
        )
        for item in payload["results"]
    ]
    lines = [
        "# TitLeNet Student Quantization Report",
        "",
        "## Summary",
        "",
        f"- model_id: `{payload['model_id']}`",
        f"- model_label: `{payload['model_label']}`",
        f"- fp32_logits: `{payload['inputs']['fp32_logits']}`",
        f"- fp32_top1: `{payload['inputs']['fp32_top1']}`",
        f"- split: `{payload['validation']['split']}`",
        f"- samples: `{payload['validation']['sample_count']}`",
        f"- calibration_split: `{payload['calibration']['split']}`",
        f"- calibration_samples: `{payload['calibration']['sample_count']}`",
        "",
        "## Trial Results",
        "",
        (
            "| trial | precision | method | status | logits_mb | top1_mb | "
            "latency_ms | top1_agreement | ndcg@3_drop | ndcg@5_drop | "
            "max_abs_diff | note |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        lines.append("| " + " | ".join(result_to_row(result)) + " |")

    lines.extend(
        [
            "",
            "## Acceptance Criteria",
            "",
            f"- top1_model_agreement >= `{payload['thresholds']['min_top1_agreement']}`",
            f"- ndcg@5_drop <= `{payload['thresholds']['max_ndcg5_drop']}`",
            "- top1 output remains an int64 palette index in `0..31`.",
            "",
            "## Notes",
            "",
            (
                "- INT4 is tracked as an experimental slot. The current ONNX Runtime "
                "quantization path does not provide practical Conv INT4 PTQ for this "
                "model/runtime combination."
            ),
            "- QAT should be evaluated after PTQ if INT8 accuracy drops beyond the criteria.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataclass_payload(result: QuantizationResult) -> dict[str, Any]:
    return asdict(result)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_count <= 0:
        raise ValueError(f"sample-count must be positive: {args.sample_count}")
    if args.calibration_sample_count <= 0:
        raise ValueError(
            f"calibration-sample-count must be positive: {args.calibration_sample_count}"
        )

    logits_onnx = resolve_inside_project(
        PROJECT_ROOT,
        args.logits_onnx,
        must_exist=True,
        description="fp32 logits onnx",
    )
    top1_onnx = resolve_inside_project(
        PROJECT_ROOT,
        args.top1_onnx,
        must_exist=True,
        description="fp32 top1 onnx",
    )
    output_dir = resolve_inside_project(
        PROJECT_ROOT,
        args.output_dir,
        must_exist=False,
        description="quantization output dir",
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
    report_path = resolve_inside_project(
        PROJECT_ROOT,
        args.report_path,
        must_exist=False,
        description="quantization report",
    )
    metrics_path = resolve_inside_project(
        PROJECT_ROOT,
        args.metrics_path,
        must_exist=False,
        description="quantization metrics",
    )

    validation_dataset = make_dataset(
        split=args.split,
        data_root=data_root,
        labels_matrix=labels_matrix,
        labels_soft=labels_soft,
    )
    calibration_dataset = make_dataset(
        split=args.calibration_split,
        data_root=data_root,
        labels_matrix=labels_matrix,
        labels_soft=labels_soft,
    )
    validation_indices = selected_indices(
        dataset_size=len(validation_dataset),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    calibration_indices = selected_indices(
        dataset_size=len(calibration_dataset),
        sample_count=args.calibration_sample_count,
        seed=args.calibration_seed,
    )
    calibration_arrays = sample_arrays(calibration_dataset, calibration_indices)

    def calibration_reader_factory() -> NumpyCalibrationDataReader:
        return NumpyCalibrationDataReader(calibration_arrays)

    import onnxruntime as ort

    fp32_logits_session = make_session(logits_onnx, ort)
    fp32_top1_session = make_session(top1_onnx, ort)
    validation_context = QuantizationValidationContext(
        fp32_logits_session=fp32_logits_session,
        fp32_top1_session=fp32_top1_session,
        dataset=validation_dataset,
        indices=validation_indices,
        min_top1_agreement=args.min_top1_agreement,
        max_ndcg5_drop=args.max_ndcg5_drop,
    )
    started_at = time.time()
    results = [
        run_trial(
            trial=trial_from_name(trial_name),
            logits_onnx=logits_onnx,
            top1_onnx=top1_onnx,
            output_dir=output_dir,
            calibration_reader_factory=calibration_reader_factory,
            validation_context=validation_context,
            warmup_steps=args.latency_warmup_steps,
            benchmark_steps=args.latency_benchmark_steps,
            skip_latency=args.skip_latency,
        )
        for trial_name in args.trials
    ]
    payload = {
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "created_at_unix": started_at,
        "inputs": {
            "fp32_logits": display_path(logits_onnx),
            "fp32_top1": display_path(top1_onnx),
            "data_root": display_path(data_root),
        },
        "validation": {
            "split": args.split,
            "sample_count": len(validation_indices),
            "seed": args.seed,
        },
        "calibration": {
            "split": args.calibration_split,
            "sample_count": len(calibration_indices),
            "seed": args.calibration_seed,
        },
        "thresholds": {
            "min_top1_agreement": args.min_top1_agreement,
            "max_ndcg5_drop": args.max_ndcg5_drop,
        },
        "results": [dataclass_payload(result) for result in results],
    }
    write_metrics_json(path=metrics_path, payload=payload)
    write_report(path=report_path, payload=payload)

    print(f"Wrote quantization report: {report_path}")
    print(f"Wrote quantization metrics: {metrics_path}")
    for result in results:
        print(f"{result.name}: {result.status}")

    if args.fail_on_regression:
        failed = [
            result
            for result in results
            if result.status not in {"passed", "unsupported"}
        ]
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
