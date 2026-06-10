from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import resolve_project_path


BASELINE_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_warm_kd90_baseline_metrics.json"
)
PTQ_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_warm_kd90_quantization_metrics.json"
)
QAT_TRAINING_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/"
    "titlenet_student_qat_kd_90_10_metrics.json"
)
QAT_QUANTIZATION_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_qat_kd90_quantization_metrics.json"
)
MOBILE_PROXY_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_mobile_proxy_latency_metrics.json"
)
REPORT_OUTPUT = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_quantization_comparison_report.md"
)
METRICS_OUTPUT = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_quantization_comparison_metrics.json"
)


@dataclass(frozen=True)
class ComparisonRow:
    model: str
    precision: str
    method: str
    status: str
    ndcg_at_3: float | None
    ndcg_at_5: float | None
    ndcg_at_3_drop: float | None
    ndcg_at_5_drop: float | None
    top1_agreement: float | None
    logits_size_mb: float | None
    top1_size_mb: float | None
    latency_ms: float | None
    latency_p95_ms: float | None
    eval_split: str
    eval_sample_count: int | None
    eval_seed: int | None
    note: str


def load_json(path: Path) -> Mapping[str, Any]:
    metrics_path = resolve_project_path(
        PROJECT_ROOT,
        path,
        must_exist=True,
        description="metrics path",
    )
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"metrics payload must be a mapping: {path}")
    return payload


def nested_float(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> float | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return float(value)


def path_size_mb(path_text: str | None) -> float | None:
    if not path_text:
        return None
    try:
        path = resolve_project_path(
            PROJECT_ROOT,
            path_text,
            must_exist=True,
            description="model artifact path",
        )
    except (FileNotFoundError, ValueError):
        return None
    return path.stat().st_size / (1024 * 1024)


def nested_int(
    payload: Mapping[str, Any],
    keys: tuple[str, ...],
) -> int | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return int(value)


def first_reference_validation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for item in payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        validation = item.get("validation")
        if not isinstance(validation, Mapping):
            continue
        if validation.get("fp32_ndcg_at_3") is not None and validation.get(
            "fp32_ndcg_at_5"
        ) is not None:
            return validation
    raise ValueError("quantization payload is missing FP32 reference validation metrics")


def validation_split(payload: Mapping[str, Any]) -> str:
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        return "-"
    return str(validation.get("split", "-"))


def validation_seed(payload: Mapping[str, Any]) -> int | None:
    return nested_int(payload, ("validation", "seed"))


def validation_sample_count(
    payload: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> int | None:
    sample_count = validation.get("sample_count")
    if sample_count is not None:
        return int(sample_count)
    return nested_int(payload, ("validation", "sample_count"))


def mobile_latency_lookup(
    payload: Mapping[str, Any] | None,
    *,
    thread_mode: str = "default",
) -> dict[str, Mapping[str, Any]]:
    if payload is None:
        return {}
    lookup: dict[str, Mapping[str, Any]] = {}
    for item in payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("thread_mode") != thread_mode:
            continue
        lookup[str(item.get("model_key", ""))] = item
    return lookup


def latency_metric_ms(
    lookup: Mapping[str, Mapping[str, Any]],
    model_key: str | None,
    metric: str,
) -> float | None:
    if model_key is None:
        return None
    row = lookup.get(model_key)
    if row is None:
        return None
    return nested_float(row, ("latency", metric))


def latency_p50_ms(
    lookup: Mapping[str, Mapping[str, Any]],
    model_key: str | None,
) -> float | None:
    return latency_metric_ms(lookup, model_key, "p50_ms")


def latency_p95_ms(
    lookup: Mapping[str, Mapping[str, Any]],
    model_key: str | None,
) -> float | None:
    return latency_metric_ms(lookup, model_key, "p95_ms")


MOBILE_MODEL_KEYS = {
    ("Student", "fp32"): "student_kd_fp32",
    ("Student", "fp16"): "student_kd_ptq_fp16",
    ("Student", "int8_dynamic"): "student_kd_ptq_int8_dynamic",
    ("Student", "int8_static"): "student_kd_ptq_int8_static",
    ("Student-QAT", "fp32"): "student_qat_fp32",
    ("Student-QAT", "fp16"): "student_qat_ptq_fp16",
    ("Student-QAT", "int8_dynamic"): "student_qat_ptq_int8_dynamic",
    ("Student-QAT", "int8_static"): "student_qat_ptq_int8_static",
}


def fp32_reference_row(
    payload: Mapping[str, Any],
    *,
    model_name: str,
    method: str,
    mobile_model_key: str,
    mobile_lookup: Mapping[str, Mapping[str, Any]],
    note: str,
) -> ComparisonRow:
    validation = first_reference_validation(payload)
    inputs = payload.get("inputs", {})
    if not isinstance(inputs, Mapping):
        inputs = {}
    return ComparisonRow(
        model=model_name,
        precision="FP32",
        method=method,
        status="reference",
        ndcg_at_3=nested_float(validation, ("fp32_ndcg_at_3",)),
        ndcg_at_5=nested_float(validation, ("fp32_ndcg_at_5",)),
        ndcg_at_3_drop=0.0,
        ndcg_at_5_drop=0.0,
        top1_agreement=1.0,
        logits_size_mb=path_size_mb(str(inputs.get("fp32_logits", ""))),
        top1_size_mb=path_size_mb(str(inputs.get("fp32_top1", ""))),
        latency_ms=latency_p50_ms(mobile_lookup, mobile_model_key),
        latency_p95_ms=latency_p95_ms(mobile_lookup, mobile_model_key),
        eval_split=validation_split(payload),
        eval_sample_count=validation_sample_count(payload, validation),
        eval_seed=validation_seed(payload),
        note=note,
    )


def quantization_rows(
    payload: Mapping[str, Any],
    *,
    model_name: str,
    method_prefix: str,
    mobile_lookup: Mapping[str, Mapping[str, Any]],
) -> list[ComparisonRow]:
    rows: list[ComparisonRow] = []
    for item in payload.get("results", []):
        if not isinstance(item, Mapping):
            continue
        validation = item.get("validation") or {}
        if not isinstance(validation, Mapping):
            validation = {}
        trial_name = str(item.get("name", ""))
        method = str(item.get("method", "-"))
        if "per_channel" in trial_name:
            method = f"{method} per-channel"
        mobile_model_key = MOBILE_MODEL_KEYS.get((model_name, trial_name))
        rows.append(
            ComparisonRow(
                model=model_name,
                precision=str(item.get("precision", "-")).upper(),
                method=f"{method_prefix} {method}",
                status=str(item.get("status", "-")),
                ndcg_at_3=nested_float(validation, ("quantized_ndcg_at_3",)),
                ndcg_at_5=nested_float(validation, ("quantized_ndcg_at_5",)),
                ndcg_at_3_drop=nested_float(validation, ("ndcg_at_3_drop",)),
                ndcg_at_5_drop=nested_float(validation, ("ndcg_at_5_drop",)),
                top1_agreement=nested_float(validation, ("top1_model_agreement",)),
                logits_size_mb=nested_float(item, ("logits_size_mb",)),
                top1_size_mb=nested_float(item, ("top1_size_mb",)),
                latency_ms=latency_p50_ms(mobile_lookup, mobile_model_key),
                latency_p95_ms=latency_p95_ms(mobile_lookup, mobile_model_key),
                eval_split=validation_split(payload),
                eval_sample_count=validation_sample_count(payload, validation),
                eval_seed=validation_seed(payload),
                note=str(item.get("reason") or validation.get("reason") or ""),
            )
        )
    return rows


def build_rows(
    *,
    baseline_payload: Mapping[str, Any],
    ptq_payload: Mapping[str, Any],
    qat_training_payload: Mapping[str, Any],
    qat_quantization_payload: Mapping[str, Any],
    mobile_payload: Mapping[str, Any] | None = None,
) -> list[ComparisonRow]:
    _ = baseline_payload
    mobile_lookup = mobile_latency_lookup(mobile_payload)
    return [
        fp32_reference_row(
            ptq_payload,
            model_name="Student",
            method="baseline",
            mobile_model_key="student_kd_fp32",
            mobile_lookup=mobile_lookup,
            note="FP32 ONNX reference from the same quantization validation set",
        ),
        *quantization_rows(
            ptq_payload,
            model_name="Student",
            method_prefix="PTQ",
            mobile_lookup=mobile_lookup,
        ),
        fp32_reference_row(
            qat_quantization_payload,
            model_name="Student-QAT",
            method="QAT baseline",
            mobile_model_key="student_qat_fp32",
            mobile_lookup=mobile_lookup,
            note=(
                "QAT-trained FP32 ONNX reference from the same quantization "
                f"validation set; best_epoch={qat_training_payload.get('best_epoch', '-')}"
            ),
        ),
        *quantization_rows(
            qat_quantization_payload,
            model_name="Student-QAT",
            method_prefix="QAT+PTQ",
            mobile_lookup=mobile_lookup,
        ),
    ]


def deployment_candidates(rows: list[ComparisonRow]) -> list[ComparisonRow]:
    candidates = [
        row
        for row in rows
        if row.status == "passed"
        and row.top1_agreement is not None
        and row.ndcg_at_5_drop is not None
    ]
    return sorted(
        candidates,
        key=lambda row: (
            -float(row.top1_agreement or 0.0),
            float(row.ndcg_at_5_drop or 0.0),
            float(row.latency_ms or 999999.0),
            float(row.logits_size_mb or 999999.0),
        ),
    )


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_row(row: ComparisonRow) -> str:
    return (
        f"| {row.model} | {row.precision} | {row.method} | {row.status} | "
        f"{fmt(row.ndcg_at_3)} | {fmt(row.ndcg_at_5)} | "
        f"{fmt(row.ndcg_at_3_drop)} | {fmt(row.ndcg_at_5_drop)} | "
        f"{fmt(row.top1_agreement)} | {fmt(row.logits_size_mb)} | "
        f"{fmt(row.latency_ms)} | {fmt(row.latency_p95_ms)} | {row.eval_split} | "
        f"{fmt(row.eval_sample_count)} | {fmt(row.eval_seed)} | {row.note} |"
    )


def build_report(rows: list[ComparisonRow]) -> str:
    candidates = deployment_candidates(rows)
    best = candidates[0] if candidates else None
    lines = [
        "# TitLeNet Student Quantization Comparison",
        "",
        "## Summary",
        "",
        (
            f"- best_current_candidate: `{best.model} {best.precision} {best.method}`"
            if best is not None
            else "- best_current_candidate: `none`"
        ),
        "- static INT8 is kept as an improvement target when its status is `regressed`.",
        "- INT4 remains unsupported in the current ONNX Runtime Conv quantization path.",
        "- NDCG/top-1 values use the same ONNX validation split/seed/sample count per model family.",
        "- latency ms uses the mobile-proxy top-1 ONNX benchmark default-thread p50/p95 values.",
        "",
        "## Comparison Table",
        "",
        (
            "| model | precision | method | status | NDCG@3 | NDCG@5 | "
            "NDCG@3 drop | NDCG@5 drop | top1 agreement | logits MB | "
            "latency p50 ms | latency p95 ms | eval split | eval n | eval seed | note |"
        ),
        (
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | --- | ---: | ---: | --- |"
        ),
    ]
    lines.extend(markdown_row(row) for row in rows)
    lines.extend(
        [
            "",
            "## Deployment Candidates",
            "",
        ]
    )
    if candidates:
        lines.extend(
            f"{rank}. `{row.model} {row.precision} {row.method}` "
            f"(top1={fmt(row.top1_agreement)}, "
            f"NDCG@5 drop={fmt(row.ndcg_at_5_drop)}, "
            f"latency p50/p95={fmt(row.latency_ms)}/{fmt(row.latency_p95_ms)} ms, "
            f"logits={fmt(row.logits_size_mb)} MB)"
            for rank, row in enumerate(candidates[:4], start=1)
        )
    else:
        lines.append("No candidate passed the current criteria.")
    lines.extend(
        [
            "",
            "## Immediate Conclusion",
            "",
            "- Prefer QAT FP16 first when mobile FP16 execution is supported.",
            "- Use QAT INT8 dynamic as the second candidate only after mobile operator support is confirmed.",
            "- Do not select static INT8 yet; run calibration/layer-exclusion experiments first.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[ComparisonRow], report: str) -> None:
    metrics_path = resolve_project_path(
        PROJECT_ROOT,
        METRICS_OUTPUT,
        must_exist=False,
        description="comparison metrics output",
    )
    report_path = resolve_project_path(
        PROJECT_ROOT,
        REPORT_OUTPUT,
        must_exist=False,
        description="comparison report output",
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": [asdict(row) for row in rows],
        "deployment_candidates": [
            asdict(row)
            for row in deployment_candidates(rows)
        ],
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(report, encoding="utf-8")


def main() -> int:
    rows = build_rows(
        baseline_payload=load_json(BASELINE_METRICS_PATH),
        ptq_payload=load_json(PTQ_METRICS_PATH),
        qat_training_payload=load_json(QAT_TRAINING_METRICS_PATH),
        qat_quantization_payload=load_json(QAT_QUANTIZATION_METRICS_PATH),
        mobile_payload=load_json(MOBILE_PROXY_METRICS_PATH),
    )
    report = build_report(rows)
    write_outputs(rows, report)
    print(f"Wrote comparison report: {REPORT_OUTPUT}")
    print(f"Wrote comparison metrics: {METRICS_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
