from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import resolve_project_path


DEFAULT_INPUT_SHAPE = (1, 4, 36, 136)
DEFAULT_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_mobile_proxy_latency_report.md"
)
DEFAULT_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_mobile_proxy_latency_metrics.json"
)
DEFAULT_WARMUP_STEPS = 50
DEFAULT_BENCHMARK_STEPS = 300
DEFAULT_REPEATS = 5
DEFAULT_SEED = 42
OUTPUT_MIN = 0
OUTPUT_MAX = 31


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_variant: str
    method: str
    path: Path
    decision: str


@dataclass(frozen=True)
class ThreadMode:
    key: str
    label: str
    intra_op_threads: int | None
    inter_op_threads: int | None


@dataclass(frozen=True)
class LatencyStats:
    min_ms: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    model_key: str
    model_variant: str
    method: str
    decision: str
    model_path: str
    top1_size_mb: float
    thread_mode: str
    intra_op_threads: int | None
    inter_op_threads: int | None
    load_ms: float
    latency: LatencyStats
    output_name: str
    output_dtype: str
    output_shape: list[int]
    output_value: int
    output_range_valid: bool


DEFAULT_MODELS = (
    ModelSpec(
        key="student_kd_fp32",
        model_variant="Student KD",
        method="FP32 ONNX",
        path=Path("outputs/title_color_recommendation/onnx/titlenet_student_warm_kd90_top1.onnx"),
        decision="Reference",
    ),
    ModelSpec(
        key="student_kd_ptq_fp16",
        model_variant="Student KD",
        method="PTQ FP16",
        path=Path(
            "outputs/title_color_recommendation/quantization/"
            "titlenet_student_warm_kd90_fp16_top1.onnx"
        ),
        decision="Passed",
    ),
    ModelSpec(
        key="student_kd_ptq_int8_dynamic",
        model_variant="Student KD",
        method="PTQ INT8 Dynamic",
        path=Path(
            "outputs/title_color_recommendation/quantization/"
            "titlenet_student_warm_kd90_int8_dynamic_top1.onnx"
        ),
        decision="Passed",
    ),
    ModelSpec(
        key="student_kd_ptq_int8_static",
        model_variant="Student KD",
        method="PTQ INT8 Static",
        path=Path(
            "outputs/title_color_recommendation/quantization/"
            "titlenet_student_warm_kd90_int8_static_top1.onnx"
        ),
        decision="Rejected",
    ),
    ModelSpec(
        key="student_qat_fp32",
        model_variant="Student QAT",
        method="FP32 ONNX",
        path=Path("outputs/title_color_recommendation/onnx/titlenet_student_qat_kd90_top1.onnx"),
        decision="Reference",
    ),
    ModelSpec(
        key="student_qat_ptq_fp16",
        model_variant="Student QAT",
        method="QAT + PTQ FP16",
        path=Path("outputs/title_color_recommendation/deployment/titlenet_student_qat_fp16_top1.onnx"),
        decision="Selected",
    ),
    ModelSpec(
        key="student_qat_ptq_int8_dynamic",
        model_variant="Student QAT",
        method="QAT + PTQ INT8 Dynamic",
        path=Path(
            "outputs/title_color_recommendation/quantization/qat_kd90/"
            "titlenet_student_warm_kd90_int8_dynamic_top1.onnx"
        ),
        decision="Passed, not final",
    ),
    ModelSpec(
        key="student_qat_ptq_int8_static",
        model_variant="Student QAT",
        method="QAT + PTQ INT8 Static",
        path=Path(
            "outputs/title_color_recommendation/quantization/qat_kd90/"
            "titlenet_student_warm_kd90_int8_static_top1.onnx"
        ),
        decision="Rejected",
    ),
    ModelSpec(
        key="student_qat_static_int8_sweep_best",
        model_variant="Student QAT",
        method="Static INT8 Sweep Best",
        path=Path(
            "outputs/title_color_recommendation/quantization/static_int8_sweep/qat/"
            "qat_minmax_val200_exclude_head_top1.onnx"
        ),
        decision="Rejected",
    ),
)


THREAD_MODES = (
    ThreadMode(
        key="default",
        label="ORT CPU default",
        intra_op_threads=None,
        inter_op_threads=None,
    ),
    ThreadMode(
        key="single_thread",
        label="ORT CPU 1-thread",
        intra_op_threads=1,
        inter_op_threads=1,
    ),
    ThreadMode(
        key="two_thread",
        label="ORT CPU 2-thread",
        intra_op_threads=2,
        inter_op_threads=1,
    ),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TitLeNet top-1 ONNX models with ONNX Runtime CPU as a "
            "mobile proxy latency measurement."
        )
    )
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--benchmark-steps", type=int, default=DEFAULT_BENCHMARK_STEPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    return parser.parse_args(argv)


def display_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def require_positive_int(value: int, *, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive: {value}")


def make_proxy_input(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    array = np.empty(DEFAULT_INPUT_SHAPE, dtype=np.float32)
    array[:, :3] = rng.random((1, 3, 36, 136), dtype=np.float32)
    array[:, 3:] = (rng.random((1, 1, 36, 136)) >= 0.5).astype(np.float32)
    return np.ascontiguousarray(array)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"ratio must be in 0..1: {ratio}")
    sorted_values = sorted(values)
    index = min(math.ceil(len(sorted_values) * ratio) - 1, len(sorted_values) - 1)
    return float(sorted_values[max(index, 0)])


def latency_stats(values: list[float]) -> LatencyStats:
    if not values:
        raise ValueError("latency values must not be empty")
    return LatencyStats(
        min_ms=float(min(values)),
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        mean_ms=float(sum(values) / len(values)),
        max_ms=float(max(values)),
    )


def make_session_options(thread_mode: ThreadMode) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    if thread_mode.intra_op_threads is not None:
        options.intra_op_num_threads = thread_mode.intra_op_threads
    if thread_mode.inter_op_threads is not None:
        options.inter_op_num_threads = thread_mode.inter_op_threads
    return options


def make_session(path: Path, thread_mode: ThreadMode) -> tuple[Any, float]:
    import onnxruntime as ort

    options = make_session_options(thread_mode)
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    load_ms = (time.perf_counter() - started) * 1000.0
    return session, load_ms


def run_once(session: Any, *, input_name: str, output_name: str, input_array: np.ndarray) -> np.ndarray:
    outputs = session.run([output_name], {input_name: input_array})
    return np.asarray(outputs[0])


def benchmark_session(
    *,
    session: Any,
    input_array: np.ndarray,
    warmup_steps: int,
    benchmark_steps: int,
    repeats: int,
) -> tuple[LatencyStats, np.ndarray]:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    last_output = run_once(
        session,
        input_name=input_name,
        output_name=output_name,
        input_array=input_array,
    )
    for _index in range(warmup_steps):
        last_output = run_once(
            session,
            input_name=input_name,
            output_name=output_name,
            input_array=input_array,
        )

    timings: list[float] = []
    for _repeat in range(repeats):
        for _index in range(benchmark_steps):
            started = time.perf_counter()
            last_output = run_once(
                session,
                input_name=input_name,
                output_name=output_name,
                input_array=input_array,
            )
            timings.append((time.perf_counter() - started) * 1000.0)
    return latency_stats(timings), last_output


def benchmark_model(
    *,
    model: ModelSpec,
    thread_mode: ThreadMode,
    input_array: np.ndarray,
    warmup_steps: int,
    benchmark_steps: int,
    repeats: int,
) -> BenchmarkResult:
    resolved_path = resolve_project_path(
        PROJECT_ROOT,
        model.path,
        must_exist=True,
        description=f"{model.key} ONNX model",
    )
    session, load_ms = make_session(resolved_path, thread_mode)
    stats, output = benchmark_session(
        session=session,
        input_array=input_array,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
        repeats=repeats,
    )
    output_value = int(output.reshape(-1)[0])
    return BenchmarkResult(
        model_key=model.key,
        model_variant=model.model_variant,
        method=model.method,
        decision=model.decision,
        model_path=display_path(resolved_path),
        top1_size_mb=resolved_path.stat().st_size / (1024 * 1024),
        thread_mode=thread_mode.key,
        intra_op_threads=thread_mode.intra_op_threads,
        inter_op_threads=thread_mode.inter_op_threads,
        load_ms=load_ms,
        latency=stats,
        output_name=session.get_outputs()[0].name,
        output_dtype=str(output.dtype),
        output_shape=[int(dim) for dim in output.shape],
        output_value=output_value,
        output_range_valid=OUTPUT_MIN <= output_value <= OUTPUT_MAX,
    )


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def result_by_mode(
    results: list[BenchmarkResult],
    *,
    model_key: str,
    thread_mode: str,
) -> BenchmarkResult | None:
    for result in results:
        if result.model_key == model_key and result.thread_mode == thread_mode:
            return result
    return None


def write_report(
    *,
    path: Path,
    results: list[BenchmarkResult],
    warmup_steps: int,
    benchmark_steps: int,
    repeats: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# TitLeNet Student Mobile Proxy Latency Benchmark",
        "",
        "## Summary",
        "",
        "This benchmark measures TitLeNet top-1 ONNX inference latency with "
        "ONNX Runtime CPU as a proxy for React Native on-device inference. "
        "It is not a replacement for Android/iOS release-build profiling.",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| input_shape | `{list(DEFAULT_INPUT_SHAPE)}` |",
        "| input_type | random RGB `0..1` + binary mask `0/1` |",
        "| target_output | top-1 palette index |",
        "| provider | `CPUExecutionProvider` |",
        f"| warmup_steps | `{warmup_steps}` |",
        f"| benchmark_steps_per_repeat | `{benchmark_steps}` |",
        f"| repeats | `{repeats}` |",
        f"| seed | `{seed}` |",
        "",
        "## Results",
        "",
        "| Model Variant | Method | Size (MB) | Default P50/P95 (ms) | 1-thread P50/P95 (ms) | 2-thread P50/P95 (ms) | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for model in DEFAULT_MODELS:
        default = result_by_mode(results, model_key=model.key, thread_mode="default")
        single = result_by_mode(results, model_key=model.key, thread_mode="single_thread")
        two = result_by_mode(results, model_key=model.key, thread_mode="two_thread")
        size = default.top1_size_mb if default is not None else None
        default_latency = (
            f"{fmt(default.latency.p50_ms)}/{fmt(default.latency.p95_ms)}"
            if default is not None
            else "-"
        )
        single_latency = (
            f"{fmt(single.latency.p50_ms)}/{fmt(single.latency.p95_ms)}"
            if single is not None
            else "-"
        )
        two_latency = (
            f"{fmt(two.latency.p50_ms)}/{fmt(two.latency.p95_ms)}"
            if two is not None
            else "-"
        )
        lines.append(
            f"| {model.model_variant} | {model.method} | {fmt(size)} | "
            f"{default_latency} | {single_latency} | {two_latency} | "
            f"{model.decision} |"
        )

    lines.extend(
        [
            "",
            "## Load Time",
            "",
            "| Model Variant | Method | Default load (ms) | 1-thread load (ms) | 2-thread load (ms) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for model in DEFAULT_MODELS:
        default = result_by_mode(results, model_key=model.key, thread_mode="default")
        single = result_by_mode(results, model_key=model.key, thread_mode="single_thread")
        two = result_by_mode(results, model_key=model.key, thread_mode="two_thread")
        lines.append(
            f"| {model.model_variant} | {model.method} | "
            f"{fmt(default.load_ms if default else None)} | "
            f"{fmt(single.load_ms if single else None)} | "
            f"{fmt(two.load_ms if two else None)} |"
        )

    invalid_outputs = [result for result in results if not result.output_range_valid]
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The reported latency is pure ONNX inference latency for top-1 models only.",
            "- React Native preprocessing, asset loading, UI work, and bridge overhead are not included.",
            "- The single-thread and two-thread settings are intended as conservative mobile CPU proxy measurements.",
            "- Final deployment latency must still be measured on target Android/iOS devices in release builds.",
        ]
    )
    if invalid_outputs:
        lines.append("- At least one model produced an out-of-range top-1 index.")
    else:
        lines.append("- All measured outputs were valid palette indices in `0..31`.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics(
    *,
    path: Path,
    results: list[BenchmarkResult],
    warmup_steps: int,
    benchmark_steps: int,
    repeats: int,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_unix": time.time(),
        "benchmark": {
            "input_shape": list(DEFAULT_INPUT_SHAPE),
            "input_type": "random_rgb_0_1_binary_mask_0_1",
            "provider": "CPUExecutionProvider",
            "warmup_steps": warmup_steps,
            "benchmark_steps_per_repeat": benchmark_steps,
            "repeats": repeats,
            "seed": seed,
        },
        "thread_modes": [asdict(mode) for mode in THREAD_MODES],
        "results": [asdict(result) for result in results],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_positive_int(args.warmup_steps, name="warmup_steps")
    require_positive_int(args.benchmark_steps, name="benchmark_steps")
    require_positive_int(args.repeats, name="repeats")

    input_array = make_proxy_input(args.seed)
    results: list[BenchmarkResult] = []
    for model in DEFAULT_MODELS:
        for thread_mode in THREAD_MODES:
            results.append(
                benchmark_model(
                    model=model,
                    thread_mode=thread_mode,
                    input_array=input_array,
                    warmup_steps=args.warmup_steps,
                    benchmark_steps=args.benchmark_steps,
                    repeats=args.repeats,
                )
            )

    report_path = resolve_project_path(
        PROJECT_ROOT,
        args.report_path,
        must_exist=False,
        description="report path",
    )
    metrics_path = resolve_project_path(
        PROJECT_ROOT,
        args.metrics_path,
        must_exist=False,
        description="metrics path",
    )
    write_report(
        path=report_path,
        results=results,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        repeats=args.repeats,
        seed=args.seed,
    )
    write_metrics(
        path=metrics_path,
        results=results,
        warmup_steps=args.warmup_steps,
        benchmark_steps=args.benchmark_steps,
        repeats=args.repeats,
        seed=args.seed,
    )

    print(f"Wrote report: {display_path(report_path)}")
    print(f"Wrote metrics: {display_path(metrics_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
