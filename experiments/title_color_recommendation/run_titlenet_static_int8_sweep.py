from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import (
    quantize_titlenet_student_onnx as quantize,
)
from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.prepare_titlenet_student_quantization_baseline import (
    DEFAULT_DATA_ROOT,
    MODEL_ID,
    MODEL_LABEL,
    path_size_mb,
)
from experiments.title_color_recommendation.validate_titlenet_onnx import (
    selected_indices,
)
from scripts.title_color_recommendation.export_titlenet_onnx import (
    LOGITS_OUTPUT_NAME,
    TOP1_OUTPUT_NAME,
)


TARGET_STUDENT = "student"
TARGET_QAT = "qat"
DEFAULT_TARGET = TARGET_QAT
DEFAULT_SAMPLE_COUNT = 100
DEFAULT_SEED = 42
DEFAULT_MIN_TOP1_AGREEMENT = 0.98
DEFAULT_MAX_NDCG5_DROP = 0.005
DEFAULT_WARMUP_STEPS = 10
DEFAULT_BENCHMARK_STEPS = 50
DEFAULT_TRIAL_SET = "base"
STATIC_SWEEP_OUTPUT_DIR = Path(
    "outputs/title_color_recommendation/quantization/static_int8_sweep"
)
REPORT_OUTPUT = (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_static_int8_sweep_report.md"
)
METRICS_OUTPUT = (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_static_int8_sweep_metrics.json"
)


@dataclass(frozen=True)
class StaticInt8Target:
    name: str
    label: str
    logits_onnx: Path
    top1_onnx: Path


@dataclass(frozen=True)
class StaticInt8Trial:
    name: str
    calibration_split: str
    calibration_sample_count: int
    calibration_method: str
    exclude_preset: str = "none"
    per_channel: bool = False


@dataclass(frozen=True)
class StaticInt8Result:
    target: str
    trial: str
    status: str
    calibration_split: str
    calibration_sample_count: int
    calibration_method: str
    exclude_preset: str
    excluded_nodes: list[str]
    per_channel: bool
    logits_path: str | None
    top1_path: str | None
    logits_size_mb: float | None
    top1_size_mb: float | None
    latency_ms: float | None
    top1_agreement: float | None
    ndcg_at_3_drop: float | None
    ndcg_at_5_drop: float | None
    max_abs_diff: float | None
    reason: str | None = None


def default_targets() -> dict[str, StaticInt8Target]:
    return {
        TARGET_STUDENT: StaticInt8Target(
            name=TARGET_STUDENT,
            label=MODEL_LABEL,
            logits_onnx=Path(
                "outputs/title_color_recommendation/onnx/"
                f"{MODEL_ID}_logits.onnx"
            ),
            top1_onnx=Path(
                "outputs/title_color_recommendation/onnx/"
                f"{MODEL_ID}_top1.onnx"
            ),
        ),
        TARGET_QAT: StaticInt8Target(
            name=TARGET_QAT,
            label="TitLeNet Student QAT kd_90_10",
            logits_onnx=Path(
                "outputs/title_color_recommendation/onnx/"
                "titlenet_student_qat_kd90_logits.onnx"
            ),
            top1_onnx=Path(
                "outputs/title_color_recommendation/onnx/"
                "titlenet_student_qat_kd90_top1.onnx"
            ),
        ),
    }


def default_trials() -> list[StaticInt8Trial]:
    return [
        StaticInt8Trial("minmax_val200", "val", 200, "minmax"),
        StaticInt8Trial("minmax_val500", "val", 500, "minmax"),
        StaticInt8Trial("minmax_train500", "train", 500, "minmax"),
        StaticInt8Trial("minmax_train1000", "train", 1000, "minmax"),
        StaticInt8Trial("entropy_val200", "val", 200, "entropy"),
        StaticInt8Trial("entropy_train500", "train", 500, "entropy"),
        StaticInt8Trial("percentile_val200", "val", 200, "percentile"),
        StaticInt8Trial("percentile_train500", "train", 500, "percentile"),
        StaticInt8Trial("minmax_val200_exclude_first_conv", "val", 200, "minmax", "first_conv"),
        StaticInt8Trial("minmax_val200_exclude_head", "val", 200, "minmax", "head"),
        StaticInt8Trial(
            "minmax_val200_exclude_first_conv_head",
            "val",
            200,
            "minmax",
            "first_conv_head",
        ),
        StaticInt8Trial(
            "minmax_train500_exclude_first_conv_head",
            "train",
            500,
            "minmax",
            "first_conv_head",
        ),
    ]


def focused_trials() -> list[StaticInt8Trial]:
    return [
        StaticInt8Trial(
            "minmax_val200_exclude_final_gemm",
            "val",
            200,
            "minmax",
            "final_gemm",
        ),
        StaticInt8Trial(
            "minmax_val500_exclude_final_gemm",
            "val",
            500,
            "minmax",
            "final_gemm",
        ),
        StaticInt8Trial(
            "minmax_train500_exclude_final_gemm",
            "train",
            500,
            "minmax",
            "final_gemm",
        ),
        StaticInt8Trial(
            "entropy_val200_exclude_final_gemm",
            "val",
            200,
            "entropy",
            "final_gemm",
        ),
        StaticInt8Trial(
            "percentile_val200_exclude_final_gemm",
            "val",
            200,
            "percentile",
            "final_gemm",
        ),
        StaticInt8Trial(
            "minmax_val200_exclude_first_gemm",
            "val",
            200,
            "minmax",
            "first_gemm",
        ),
        StaticInt8Trial(
            "minmax_val200_exclude_attention",
            "val",
            200,
            "minmax",
            "attention",
        ),
        StaticInt8Trial(
            "minmax_val200_exclude_attention_final_gemm",
            "val",
            200,
            "minmax",
            "attention_final_gemm",
        ),
        StaticInt8Trial(
            "minmax_val200_exclude_first_conv_final_gemm",
            "val",
            200,
            "minmax",
            "first_conv_final_gemm",
        ),
    ]


def trials_for_set(name: str) -> list[StaticInt8Trial]:
    if name == "base":
        return default_trials()
    if name == "focused":
        return focused_trials()
    if name == "all":
        return [*default_trials(), *focused_trials()]
    raise ValueError(f"unsupported trial set: {name}")


def target_names(selected: str) -> list[str]:
    if selected == "both":
        return [TARGET_STUDENT, TARGET_QAT]
    if selected not in {TARGET_STUDENT, TARGET_QAT}:
        raise ValueError(f"unsupported target: {selected}")
    return [selected]


def parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run static INT8 improvement sweep for TitLeNet Student ONNX."
    )
    parser.add_argument(
        "--target",
        choices=(TARGET_STUDENT, TARGET_QAT, "both"),
        default=DEFAULT_TARGET,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--calibration-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--min-top1-agreement",
        type=float,
        default=DEFAULT_MIN_TOP1_AGREEMENT,
    )
    parser.add_argument("--max-ndcg5-drop", type=float, default=DEFAULT_MAX_NDCG5_DROP)
    parser.add_argument("--latency-warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument(
        "--latency-benchmark-steps",
        type=int,
        default=DEFAULT_BENCHMARK_STEPS,
    )
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument(
        "--trial-set",
        choices=("base", "focused", "all"),
        default=DEFAULT_TRIAL_SET,
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


def calibration_method(name: str) -> Any:
    from onnxruntime.quantization import CalibrationMethod

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return methods[name]


def quantizable_nodes(path: Path) -> list[tuple[str, str]]:
    import onnx

    model = onnx.load(str(path))
    return [
        (node.name, node.op_type)
        for node in model.graph.node
        if node.op_type in {"Conv", "Gemm", "MatMul"}
    ]


def select_excluded_node_names(
    nodes: Sequence[tuple[str, str]],
    *,
    preset: str,
) -> list[str]:
    if preset == "none":
        return []

    first_conv = [name for name, op_type in nodes if op_type == "Conv"][:1]
    head = [name for name, _op_type in nodes if "/head/" in name]
    first_gemm = head[:1]
    final_gemm = head[-1:]
    attention = [name for name, _op_type in nodes if "/attention/" in name]
    presets = {
        "first_conv": first_conv,
        "head": head,
        "first_gemm": first_gemm,
        "final_gemm": final_gemm,
        "first_conv_head": [*first_conv, *head],
        "first_conv_final_gemm": [*first_conv, *final_gemm],
        "attention": attention,
        "attention_final_gemm": [*attention, *final_gemm],
        "first_conv_attention_head": [*first_conv, *attention, *head],
    }
    if preset not in presets:
        raise ValueError(f"unsupported exclude preset: {preset}")
    return presets[preset]


def quantize_static_model(
    *,
    input_path: Path,
    output_path: Path,
    calibration_reader_factory: Callable[[], quantize.NumpyCalibrationDataReader],
    trial: StaticInt8Trial,
) -> list[str]:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    excluded_nodes = select_excluded_node_names(
        quantizable_nodes(input_path),
        preset=trial.exclude_preset,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        model_input=input_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader_factory(),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
        per_channel=trial.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded_nodes,
        calibrate_method=calibration_method(trial.calibration_method),
    )
    return excluded_nodes


def output_path(
    *,
    target: StaticInt8Target,
    trial: StaticInt8Trial,
    kind: str,
) -> Path:
    return STATIC_SWEEP_OUTPUT_DIR / target.name / f"{target.name}_{trial.name}_{kind}.onnx"


def create_quantized_pair(
    *,
    target: StaticInt8Target,
    trial: StaticInt8Trial,
    calibration_reader_factory: Callable[[], quantize.NumpyCalibrationDataReader],
) -> tuple[Path, Path, list[str]]:
    logits_output = output_path(target=target, trial=trial, kind="logits")
    top1_output = output_path(target=target, trial=trial, kind="top1")
    excluded_nodes = quantize_static_model(
        input_path=target.logits_onnx,
        output_path=logits_output,
        calibration_reader_factory=calibration_reader_factory,
        trial=trial,
    )
    quantize_static_model(
        input_path=target.top1_onnx,
        output_path=top1_output,
        calibration_reader_factory=calibration_reader_factory,
        trial=trial,
    )
    return logits_output, top1_output, excluded_nodes


def latency_ms(
    *,
    logits_path: Path,
    warmup_steps: int,
    benchmark_steps: int,
    skip_latency: bool,
) -> float | None:
    if skip_latency:
        return None
    import onnxruntime as ort

    session = quantize.make_session(logits_path, ort)
    input_array = np.zeros(quantize.DEFAULT_INPUT_SHAPE, dtype=np.float32)
    metrics = quantize.benchmark_onnx_session(
        session=session,
        output_name=LOGITS_OUTPUT_NAME,
        input_array=input_array,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )
    return float(metrics["inference_time_ms"])


def make_calibration_factory(
    *,
    dataset: Any,
    sample_count: int,
    seed: int,
) -> Callable[[], quantize.NumpyCalibrationDataReader]:
    indices = selected_indices(
        dataset_size=len(dataset),
        sample_count=sample_count,
        seed=seed,
    )
    arrays = quantize.sample_arrays(dataset, indices)

    def factory() -> quantize.NumpyCalibrationDataReader:
        return quantize.NumpyCalibrationDataReader(arrays)

    return factory


def result_status(validation: quantize.QuantizationValidation) -> str:
    return "passed" if validation.passed else "regressed"


def run_trial(
    *,
    target: StaticInt8Target,
    trial: StaticInt8Trial,
    calibration_reader_factory: Callable[[], quantize.NumpyCalibrationDataReader],
    fp32_logits_session: Any,
    fp32_top1_session: Any,
    validation_dataset: Any,
    validation_indices: list[int],
    args: Any,
) -> StaticInt8Result:
    try:
        logits_output, top1_output, excluded_nodes = create_quantized_pair(
            target=target,
            trial=trial,
            calibration_reader_factory=calibration_reader_factory,
        )
        import onnxruntime as ort

        quantized_logits_session = quantize.make_session(logits_output, ort)
        quantized_top1_session = quantize.make_session(top1_output, ort)
        validation = quantize.validate_quantized_pair(
            fp32_logits_session=fp32_logits_session,
            fp32_top1_session=fp32_top1_session,
            quantized_logits_session=quantized_logits_session,
            quantized_top1_session=quantized_top1_session,
            dataset=validation_dataset,
            indices=validation_indices,
            min_top1_agreement=args.min_top1_agreement,
            max_ndcg5_drop=args.max_ndcg5_drop,
        )
        return StaticInt8Result(
            target=target.name,
            trial=trial.name,
            status=result_status(validation),
            calibration_split=trial.calibration_split,
            calibration_sample_count=trial.calibration_sample_count,
            calibration_method=trial.calibration_method,
            exclude_preset=trial.exclude_preset,
            excluded_nodes=excluded_nodes,
            per_channel=trial.per_channel,
            logits_path=display_path(logits_output),
            top1_path=display_path(top1_output),
            logits_size_mb=path_size_mb(logits_output),
            top1_size_mb=path_size_mb(top1_output),
            latency_ms=latency_ms(
                logits_path=logits_output,
                warmup_steps=args.latency_warmup_steps,
                benchmark_steps=args.latency_benchmark_steps,
                skip_latency=args.skip_latency,
            ),
            top1_agreement=validation.top1_model_agreement,
            ndcg_at_3_drop=validation.ndcg_at_3_drop,
            ndcg_at_5_drop=validation.ndcg_at_5_drop,
            max_abs_diff=validation.max_abs_diff,
        )
    except Exception as exc:
        return StaticInt8Result(
            target=target.name,
            trial=trial.name,
            status="failed",
            calibration_split=trial.calibration_split,
            calibration_sample_count=trial.calibration_sample_count,
            calibration_method=trial.calibration_method,
            exclude_preset=trial.exclude_preset,
            excluded_nodes=[],
            per_channel=trial.per_channel,
            logits_path=None,
            top1_path=None,
            logits_size_mb=None,
            top1_size_mb=None,
            latency_ms=None,
            top1_agreement=None,
            ndcg_at_3_drop=None,
            ndcg_at_5_drop=None,
            max_abs_diff=None,
            reason=f"{type(exc).__name__}: {exc}",
        )


def run_sweep(args: Any) -> list[StaticInt8Result]:
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
    validation_dataset = quantize.make_dataset(
        split="test",
        data_root=data_root,
        labels_matrix=labels_matrix,
        labels_soft=labels_soft,
    )
    validation_indices = selected_indices(
        dataset_size=len(validation_dataset),
        sample_count=args.sample_count,
        seed=args.seed,
    )
    calibration_datasets = {
        split: quantize.make_dataset(
            split=split,
            data_root=data_root,
            labels_matrix=labels_matrix,
            labels_soft=labels_soft,
        )
        for split in ("train", "val")
    }
    calibration_factories: dict[tuple[str, int], Callable[[], quantize.NumpyCalibrationDataReader]] = {}
    trials = trials_for_set(args.trial_set)
    for trial in trials:
        key = (trial.calibration_split, trial.calibration_sample_count)
        if key not in calibration_factories:
            calibration_factories[key] = make_calibration_factory(
                dataset=calibration_datasets[trial.calibration_split],
                sample_count=trial.calibration_sample_count,
                seed=args.calibration_seed,
            )

    import onnxruntime as ort

    targets = default_targets()
    results: list[StaticInt8Result] = []
    for target_name in target_names(args.target):
        target = targets[target_name]
        target = StaticInt8Target(
            name=target.name,
            label=target.label,
            logits_onnx=resolve_inside_project(
                PROJECT_ROOT,
                target.logits_onnx,
                must_exist=True,
                description=f"{target.name} logits onnx",
            ),
            top1_onnx=resolve_inside_project(
                PROJECT_ROOT,
                target.top1_onnx,
                must_exist=True,
                description=f"{target.name} top1 onnx",
            ),
        )
        fp32_logits_session = quantize.make_session(target.logits_onnx, ort)
        fp32_top1_session = quantize.make_session(target.top1_onnx, ort)
        for trial in trials:
            results.append(
                run_trial(
                    target=target,
                    trial=trial,
                    calibration_reader_factory=calibration_factories[
                        (trial.calibration_split, trial.calibration_sample_count)
                    ],
                    fp32_logits_session=fp32_logits_session,
                    fp32_top1_session=fp32_top1_session,
                    validation_dataset=validation_dataset,
                    validation_indices=validation_indices,
                    args=args,
                )
            )
    return results


def result_sort_key(result: StaticInt8Result) -> tuple[float, float, float, float]:
    return (
        -float(result.top1_agreement or 0.0),
        float(result.ndcg_at_5_drop or 999999.0),
        float(result.latency_ms or 999999.0),
        float(result.logits_size_mb or 999999.0),
    )


def best_results(results: Sequence[StaticInt8Result]) -> list[StaticInt8Result]:
    return sorted(
        [result for result in results if result.status != "failed"],
        key=result_sort_key,
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_outputs(
    *,
    args: Any,
    results: list[StaticInt8Result],
    started_at: float,
) -> None:
    os.makedirs("outputs/reports/model_evaluation/onnx", exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "created_at_unix": started_at,
        "target": args.target,
        "trial_set": args.trial_set,
        "thresholds": {
            "min_top1_agreement": args.min_top1_agreement,
            "max_ndcg5_drop": args.max_ndcg5_drop,
        },
        "results": [asdict(result) for result in results],
        "best_results": [asdict(result) for result in best_results(results)[:5]],
    }
    with open(
        "outputs/reports/model_evaluation/onnx/"
        "titlenet_student_static_int8_sweep_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    ranked = best_results(results)
    best = ranked[0] if ranked else None
    lines = [
        "# TitLeNet Student Static INT8 Sweep",
        "",
        "## Summary",
        "",
        (
            f"- best_trial: `{best.target} / {best.trial}`"
            if best is not None
            else "- best_trial: `none`"
        ),
        f"- target: `{args.target}`",
        f"- trial_set: `{args.trial_set}`",
        f"- min_top1_agreement: `{args.min_top1_agreement}`",
        f"- max_ndcg5_drop: `{args.max_ndcg5_drop}`",
        "",
        "## Results",
        "",
        (
            "| target | trial | status | split | calib n | method | exclude | "
            "top1 | NDCG@3 drop | NDCG@5 drop | logits MB | latency ms | note |"
        ),
        "| --- | --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in sorted(results, key=result_sort_key):
        lines.append(
            f"| {result.target} | {result.trial} | {result.status} | "
            f"{result.calibration_split} | {result.calibration_sample_count} | "
            f"{result.calibration_method} | {result.exclude_preset} | "
            f"{fmt(result.top1_agreement)} | {fmt(result.ndcg_at_3_drop)} | "
            f"{fmt(result.ndcg_at_5_drop)} | {fmt(result.logits_size_mb)} | "
            f"{fmt(result.latency_ms)} | {result.reason or ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Select a static INT8 model only if top-1 agreement reaches the threshold.",
            "- If no trial passes, keep FP16 or INT8 dynamic as the deployment candidate.",
        ]
    )
    with open(
        "outputs/reports/model_evaluation/onnx/"
        "titlenet_student_static_int8_sweep_report.md",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started_at = time.time()
    results = run_sweep(args)
    write_outputs(args=args, results=results, started_at=started_at)
    print(f"Wrote static INT8 sweep report: {REPORT_OUTPUT}")
    print(f"Wrote static INT8 sweep metrics: {METRICS_OUTPUT}")
    for result in best_results(results)[:5]:
        print(
            f"{result.target}/{result.trial}: {result.status} "
            f"top1={fmt(result.top1_agreement)} "
            f"ndcg5_drop={fmt(result.ndcg_at_5_drop)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
