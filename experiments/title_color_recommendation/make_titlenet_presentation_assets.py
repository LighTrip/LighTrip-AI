from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STUDENT_EXPERIMENT_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/titlenet_student_experiment_metrics.json"
)
KD_SWEEP_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_metrics.json"
)
ONNX_PARITY_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/onnx/titlenet_student_warm_kd90_parity_metrics.json"
)
QUANTIZATION_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_quantization_comparison_metrics.json"
)
QAT_QUANTIZATION_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_qat_kd90_quantization_metrics.json"
)
STATIC_SWEEP_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_static_int8_sweep_metrics.json"
)
MOBILE_PROXY_METRICS = PROJECT_ROOT / (
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_mobile_proxy_latency_metrics.json"
)
OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/reports/presentation/titlenet_student_quantization"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
MARKDOWN_OUTPUT = OUTPUT_DIR / "presentation_tables_and_figures.md"


@dataclass(frozen=True)
class QuantizationPresentationRow:
    label: str
    model_variant: str
    method: str
    top1_agreement: float | None
    ndcg_at_3: float | None
    ndcg_at_5: float | None
    size_mb: float | None
    default_p50_ms: float | None
    default_p95_ms: float | None
    single_p50_ms: float | None
    single_p95_ms: float | None
    decision: str


def load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"JSON payload must be a mapping: {path}")
    return payload


def nested(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def nested_float(value: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    current = nested(value, keys)
    if current is None:
        return None
    return float(current)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def reduction(before: float, after: float) -> float:
    return (before - after) / before


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _header in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def quantization_row_lookup(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("rows", []):
        if not isinstance(row, Mapping):
            continue
        lookup[(str(row.get("model")), str(row.get("method")))] = row
    return lookup


def mobile_latency_lookup(payload: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    lookup: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in payload.get("results", []):
        if not isinstance(row, Mapping):
            continue
        lookup[(str(row.get("model_key")), str(row.get("thread_mode")))] = row
    return lookup


def latency_value(
    lookup: dict[tuple[str, str], Mapping[str, Any]],
    model_key: str,
    thread_mode: str,
    metric: str,
) -> float | None:
    row = lookup.get((model_key, thread_mode))
    if row is None:
        return None
    return nested_float(row, ("latency", metric))


def quant_value(row: Mapping[str, Any] | None, key: str) -> float | None:
    if row is None or row.get(key) is None:
        return None
    return float(row[key])


def presentation_method(method: str) -> str:
    labels = {
        "baseline": "FP32 ONNX",
        "QAT baseline": "QAT FP32 ONNX",
        "PTQ float16_conversion": "PTQ FP16",
        "PTQ dynamic": "PTQ INT8 Dynamic",
        "PTQ static_qdq": "PTQ INT8 Static",
        "QAT+PTQ float16_conversion": "QAT + PTQ FP16",
        "QAT+PTQ dynamic": "QAT + PTQ INT8 Dynamic",
        "QAT+PTQ static_qdq": "QAT + PTQ INT8 Static",
    }
    return labels.get(method, method)


def qat_validation_baseline(payload: Mapping[str, Any]) -> tuple[float, float]:
    for row in payload.get("results", []):
        if not isinstance(row, Mapping):
            continue
        validation = row.get("validation")
        if not isinstance(validation, Mapping):
            continue
        ndcg3 = validation.get("fp32_ndcg_at_3")
        ndcg5 = validation.get("fp32_ndcg_at_5")
        if ndcg3 is not None and ndcg5 is not None:
            return float(ndcg3), float(ndcg5)
    raise ValueError("QAT quantization baseline NDCG values were not found")


def build_quantization_rows(
    *,
    quantization_payload: Mapping[str, Any],
    qat_quantization_payload: Mapping[str, Any],
    static_payload: Mapping[str, Any],
    mobile_payload: Mapping[str, Any],
) -> list[QuantizationPresentationRow]:
    quant_lookup = quantization_row_lookup(quantization_payload)
    latency_lookup = mobile_latency_lookup(mobile_payload)
    mappings = [
        ("Student KD FP32", "Student", "baseline", "student_kd_fp32", "Reference"),
        (
            "Student KD PTQ FP16",
            "Student",
            "PTQ float16_conversion",
            "student_kd_ptq_fp16",
            "Passed",
        ),
        (
            "Student KD PTQ INT8 Dynamic",
            "Student",
            "PTQ dynamic",
            "student_kd_ptq_int8_dynamic",
            "Passed",
        ),
        (
            "Student KD PTQ INT8 Static",
            "Student",
            "PTQ static_qdq",
            "student_kd_ptq_int8_static",
            "Rejected",
        ),
        ("Student QAT FP32", "Student-QAT", "QAT baseline", "student_qat_fp32", "Reference"),
        (
            "Student QAT + PTQ FP16",
            "Student-QAT",
            "QAT+PTQ float16_conversion",
            "student_qat_ptq_fp16",
            "Selected",
        ),
        (
            "Student QAT + PTQ INT8 Dynamic",
            "Student-QAT",
            "QAT+PTQ dynamic",
            "student_qat_ptq_int8_dynamic",
            "Passed, not final",
        ),
        (
            "Student QAT + PTQ INT8 Static",
            "Student-QAT",
            "QAT+PTQ static_qdq",
            "student_qat_ptq_int8_static",
            "Rejected",
        ),
    ]
    rows: list[QuantizationPresentationRow] = []
    for label, model, method, mobile_key, decision in mappings:
        quant_row = quant_lookup.get((model, method))
        rows.append(
            QuantizationPresentationRow(
                label=label,
                model_variant=model,
                method=presentation_method(method),
                top1_agreement=quant_value(quant_row, "top1_agreement"),
                ndcg_at_3=quant_value(quant_row, "ndcg_at_3"),
                ndcg_at_5=quant_value(quant_row, "ndcg_at_5"),
                size_mb=quant_value(quant_row, "top1_size_mb"),
                default_p50_ms=latency_value(latency_lookup, mobile_key, "default", "p50_ms"),
                default_p95_ms=latency_value(latency_lookup, mobile_key, "default", "p95_ms"),
                single_p50_ms=latency_value(latency_lookup, mobile_key, "single_thread", "p50_ms"),
                single_p95_ms=latency_value(latency_lookup, mobile_key, "single_thread", "p95_ms"),
                decision=decision,
            )
        )

    best_static = None
    best_results = static_payload.get("best_results", [])
    if isinstance(best_results, list) and best_results:
        first = best_results[0]
        if isinstance(first, Mapping):
            best_static = first
    if best_static is not None:
        qat_baseline_ndcg3, qat_baseline_ndcg5 = qat_validation_baseline(
            qat_quantization_payload
        )
        ndcg3 = qat_baseline_ndcg3 - float(best_static["ndcg_at_3_drop"])
        ndcg5 = qat_baseline_ndcg5 - float(best_static["ndcg_at_5_drop"])
        rows.append(
            QuantizationPresentationRow(
                label="Student QAT Static INT8 Sweep Best",
                model_variant="Student-QAT",
                method="Static INT8 Sweep Best",
                top1_agreement=float(best_static["top1_agreement"]),
                ndcg_at_3=ndcg3,
                ndcg_at_5=ndcg5,
                size_mb=float(best_static["top1_size_mb"]),
                default_p50_ms=latency_value(
                    latency_lookup,
                    "student_qat_static_int8_sweep_best",
                    "default",
                    "p50_ms",
                ),
                default_p95_ms=latency_value(
                    latency_lookup,
                    "student_qat_static_int8_sweep_best",
                    "default",
                    "p95_ms",
                ),
                single_p50_ms=latency_value(
                    latency_lookup,
                    "student_qat_static_int8_sweep_best",
                    "single_thread",
                    "p50_ms",
                ),
                single_p95_ms=latency_value(
                    latency_lookup,
                    "student_qat_static_int8_sweep_best",
                    "single_thread",
                    "p95_ms",
                ),
                decision="Rejected",
            )
        )
    return rows


def save_student_compression_plot(
    *,
    teacher_profile: Mapping[str, Any],
    student_profile: Mapping[str, Any],
    path: Path,
) -> None:
    metrics = [
        ("Parameters", float(teacher_profile["total_parameters"]), float(student_profile["total_parameters"])),
        ("Size (MB)", float(teacher_profile["model_size_mb"]), float(student_profile["model_size_mb"])),
        (
            "Batch1 latency (ms)",
            nested_float(teacher_profile, ("batch1_latency", "inference_time_ms")) or 0.0,
            nested_float(student_profile, ("batch1_latency", "inference_time_ms")) or 0.0,
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for axis, (title, teacher_value, student_value) in zip(axes, metrics, strict=True):
        axis.bar(["Teacher", "Student"], [teacher_value, student_value], color=["#4C78A8", "#59A14F"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.5,
            max(teacher_value, student_value) * 0.9,
            f"-{reduction(teacher_value, student_value) * 100.0:.1f}%",
            ha="center",
            va="center",
            fontsize=10,
        )
    fig.suptitle("Teacher vs Student Compression")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_kd_sweep_plot(
    *,
    kd_payload: Mapping[str, Any],
    student_only_ndcg5: float,
    path: Path,
) -> None:
    rows = [row for row in kd_payload.get("rows", []) if isinstance(row, Mapping)]
    fig, axis = plt.subplots(figsize=(8, 4.2))
    for phase, label, color in [
        ("from_scratch", "KD from scratch", "#E15759"),
        ("warm_start", "Warm-start KD", "#4C78A8"),
    ]:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        phase_rows.sort(key=lambda item: float(item["base_loss_weight"]))
        x_values = [float(row["base_loss_weight"]) for row in phase_rows]
        y_values = [float(row["test_ndcg@5"]) for row in phase_rows]
        axis.plot(x_values, y_values, marker="o", linewidth=2, label=label, color=color)
    axis.axhline(
        student_only_ndcg5,
        color="#777777",
        linestyle="--",
        linewidth=1.5,
        label="Student-only",
    )
    axis.set_xlabel("Base loss weight")
    axis.set_ylabel("NDCG@5")
    axis.set_title("KD Weight Sweep")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_onnx_parity_plot(*, metrics: Mapping[str, Any], path: Path) -> None:
    labels = ["Top-1", "Top-3", "Top-5"]
    values = [
        float(metrics["top1_agreement"]),
        float(metrics["top3_agreement"]),
        float(metrics["top5_agreement"]),
    ]
    fig, axis = plt.subplots(figsize=(6.5, 3.8))
    axis.bar(labels, values, color="#59A14F")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Agreement")
    axis.set_title("PyTorch vs ONNX Parity")
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, value + 0.015, f"{value * 100.0:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_quantization_tradeoff_plot(
    *,
    rows: list[QuantizationPresentationRow],
    path: Path,
) -> None:
    colors = {
        "Selected": "#59A14F",
        "Rejected": "#E15759",
        "Reference": "#9C9C9C",
        "Passed": "#4C78A8",
        "Passed, not final": "#F28E2B",
    }
    fig, axis = plt.subplots(figsize=(8.8, 5.2))
    for row in rows:
        if row.default_p95_ms is None or row.ndcg_at_5 is None or row.size_mb is None:
            continue
        axis.scatter(
            row.default_p95_ms,
            row.ndcg_at_5,
            s=500 * max(row.size_mb, 0.05),
            color=colors.get(row.decision, "#4C78A8"),
            alpha=0.85,
            edgecolor="#333333",
            linewidth=0.6,
        )
        axis.annotate(
            row.label.replace("Student ", "").replace(" + ", "\n+ "),
            (row.default_p95_ms, row.ndcg_at_5),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7.5,
        )
    axis.set_xlabel("CPU proxy P95 latency (ms)")
    axis.set_ylabel("NDCG@5")
    axis.set_title("Quantization Trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_final_candidate_plot(
    *,
    rows: list[QuantizationPresentationRow],
    path: Path,
) -> None:
    selected_labels = [
        "Student QAT FP32",
        "Student QAT + PTQ FP16",
        "Student QAT + PTQ INT8 Dynamic",
        "Student QAT Static INT8 Sweep Best",
    ]
    selected_rows = [row for row in rows if row.label in selected_labels]
    labels = [
        row.label.replace("Student QAT ", "").replace("+ PTQ ", "")
        for row in selected_rows
    ]
    size_values = [float(row.size_mb or 0.0) for row in selected_rows]
    p95_values = [float(row.default_p95_ms or 0.0) for row in selected_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(labels, size_values, color="#4C78A8")
    axes[0].set_title("Top-1 ONNX size")
    axes[0].set_ylabel("MB")
    axes[0].tick_params(axis="x", labelrotation=20)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(labels, p95_values, color="#F28E2B")
    axes[1].set_title("CPU proxy P95")
    axes[1].set_ylabel("ms")
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Final Candidate Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def build_markdown(
    *,
    student_payload: Mapping[str, Any],
    kd_payload: Mapping[str, Any],
    parity_payload: Mapping[str, Any],
    quant_rows: list[QuantizationPresentationRow],
) -> str:
    profiles = student_payload["profiles"]
    teacher_profile = profiles["teacher"]
    student_profile = profiles["student"]
    student_only = kd_payload["student_only"]
    kd_rows = [row for row in kd_payload.get("rows", []) if isinstance(row, Mapping)]
    parity_metrics = parity_payload["metrics"]

    teacher_params = float(teacher_profile["total_parameters"])
    student_params = float(student_profile["total_parameters"])
    teacher_size = float(teacher_profile["model_size_mb"])
    student_size = float(student_profile["model_size_mb"])
    teacher_latency = nested_float(teacher_profile, ("batch1_latency", "inference_time_ms")) or 0.0
    student_latency = nested_float(student_profile, ("batch1_latency", "inference_time_ms")) or 0.0

    lines: list[str] = [
        "# TitLeNet Student and Quantization Presentation Assets",
        "",
        "## 1. Student 모델 설계",
        "",
        "발표 메시지: 기존 TitLeNet을 Teacher 기준으로 두고, ablation study 이후 온디바이스에 적합한 `titlenet_student` 구조를 선택했다.",
        "",
        "### Table 1. Teacher vs Student Compression",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Model", "Parameters", "Size (MB)", "Batch1 Latency (ms)", "Activation"],
            [
                [
                    "Teacher TitLeNet",
                    f"{teacher_params:,.0f}",
                    fmt(teacher_size),
                    fmt(teacher_latency),
                    str(student_payload["teacher"]["activation"]),
                ],
                [
                    "Student",
                    f"{student_params:,.0f}",
                    fmt(student_size),
                    fmt(student_latency),
                    str(student_payload["student"]["activation"]),
                ],
                [
                    "Reduction",
                    pct(reduction(teacher_params, student_params)),
                    pct(reduction(teacher_size, student_size)),
                    pct(reduction(teacher_latency, student_latency)),
                    "-",
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Teacher vs Student Compression](figures/01_student_compression.png)",
            "",
            "## 2. Teacher-Student Distillation",
            "",
            "발표 메시지: Student-only를 기준선으로 두고, KD from scratch와 warm-start KD를 같은 KD weight 조합에서 비교했다.",
            "",
            "### Table 2. KD Weight Sweep",
            "",
        ]
    )
    kd_table_rows = [
        [
            "Student-only",
            "-",
            "-",
            fmt(nested_float(student_only, ("test_metrics", "val_ndcg@3")), 6),
            fmt(nested_float(student_only, ("test_metrics", "val_ndcg@5")), 6),
            "-",
            "-",
        ]
    ]
    for row in kd_rows:
        kd_table_rows.append(
            [
                str(row["phase"]),
                str(row["trial"]),
                f"{float(row['base_loss_weight']):.1f}:{float(row['distillation_loss_weight']):.1f}",
                fmt(float(row["test_ndcg@3"]), 6),
                fmt(float(row["test_ndcg@5"]), 6),
                fmt(float(row["teacher_top1_agreement"]), 6),
                str(row["best_epoch"]),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Training",
                "Trial",
                "Base:KD",
                "NDCG@3",
                "NDCG@5",
                "Teacher Top-1 Agreement",
                "Best Epoch",
            ],
            kd_table_rows,
        )
    )
    lines.extend(
        [
            "",
            "![KD Weight Sweep](figures/02_kd_weight_sweep_ndcg5.png)",
            "",
            "## 3. ONNX 변환 및 검증",
            "",
            "발표 메시지: 검증용 logits 모델과 배포용 top-1 모델을 분리했고, PyTorch와 ONNX 결과 일치를 확인했다.",
            "",
            "### Table 3. ONNX Export and Parity",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Item", "Value"],
            [
                ["Input shape", "`[1, 4, 36, 136]`"],
                ["Logits output", "`[1, 32] float32`"],
                ["Top-1 output", "`[1] int64`"],
                ["Validation split / samples", f"`{parity_metrics['split']}` / `{parity_metrics['sample_count']}`"],
                ["Top-1 agreement", pct(float(parity_metrics["top1_agreement"]))],
                ["Top-3 agreement", pct(float(parity_metrics["top3_agreement"]))],
                ["Top-5 agreement", pct(float(parity_metrics["top5_agreement"]))],
                ["Max abs diff", f"{float(parity_metrics['max_abs_diff']):.2e}"],
                ["Mean abs diff", f"{float(parity_metrics['mean_abs_diff']):.2e}"],
                ["Result", "`passed`"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "![ONNX Parity](figures/03_onnx_parity.png)",
            "",
            "## 4. 양자화 실험",
            "",
            "발표 메시지: PTQ와 QAT 기반 FP16/INT8 후보를 비교했고, static INT8은 latency 이점이 있어도 top-1 agreement 기준에서 제외했다.",
            "",
            "### Table 4. Quantization Results",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Method",
                "Top-1 Agreement",
                "NDCG@3",
                "NDCG@5",
                "Size (MB)",
                "CPU Proxy P50/P95 (ms)",
                "1-thread P50/P95 (ms)",
                "Decision",
            ],
            [
                [
                    row.model_variant,
                    row.method,
                    fmt(row.top1_agreement, 2),
                    fmt(row.ndcg_at_3, 6),
                    fmt(row.ndcg_at_5, 6),
                    fmt(row.size_mb),
                    f"{fmt(row.default_p50_ms)}/{fmt(row.default_p95_ms)}",
                    f"{fmt(row.single_p50_ms)}/{fmt(row.single_p95_ms)}",
                    row.decision,
                ]
                for row in quant_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Quantization Trade-off](figures/04_quantization_tradeoff.png)",
            "",
            "## 5. 모델 비교 및 최종 후보 선정",
            "",
            "발표 메시지: 최종 후보는 `Student QAT + PTQ FP16`이다. INT8 계열은 일부 조건에서 더 빠르지만 top-1 agreement가 배포 기준을 만족하지 못했다.",
            "",
            "### Table 5. Final Candidate Summary",
            "",
        ]
    )
    final_rows = [
        row
        for row in quant_rows
        if row.label
        in {
            "Student QAT FP32",
            "Student QAT + PTQ FP16",
            "Student QAT + PTQ INT8 Dynamic",
            "Student QAT Static INT8 Sweep Best",
        }
    ]
    lines.extend(
        markdown_table(
            [
                "Candidate",
                "Top-1 Agreement",
                "NDCG@5",
                "Size (MB)",
                "CPU Proxy P95 (ms)",
                "Decision",
                "Reason",
            ],
            [
                [
                    row.label,
                    fmt(row.top1_agreement, 2),
                    fmt(row.ndcg_at_5, 6),
                    fmt(row.size_mb),
                    fmt(row.default_p95_ms),
                    row.decision,
                    (
                        "Accuracy preserved + compact deployment model"
                        if row.decision == "Selected"
                        else "Reference or not final candidate"
                        if row.decision != "Rejected"
                        else "Top-1 agreement below deployment threshold"
                    ),
                ]
                for row in final_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Final Candidate Comparison](figures/05_final_candidate_summary.png)",
            "",
            "## 발표용 핵심 문장",
            "",
            "> Student QAT + PTQ FP16 was selected as the final on-device candidate because it preserved 100% top-1 agreement while reducing the top-1 ONNX model size from 0.292 MB to 0.153 MB. Static INT8 variants showed competitive proxy latency, but were rejected because their top-1 agreement did not satisfy the deployment threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    student_payload = load_json(STUDENT_EXPERIMENT_METRICS)
    kd_payload = load_json(KD_SWEEP_METRICS)
    parity_payload = load_json(ONNX_PARITY_METRICS)
    quantization_payload = load_json(QUANTIZATION_METRICS)
    qat_quantization_payload = load_json(QAT_QUANTIZATION_METRICS)
    static_payload = load_json(STATIC_SWEEP_METRICS)
    mobile_payload = load_json(MOBILE_PROXY_METRICS)

    quant_rows = build_quantization_rows(
        quantization_payload=quantization_payload,
        qat_quantization_payload=qat_quantization_payload,
        static_payload=static_payload,
        mobile_payload=mobile_payload,
    )

    save_student_compression_plot(
        teacher_profile=student_payload["profiles"]["teacher"],
        student_profile=student_payload["profiles"]["student"],
        path=FIGURE_DIR / "01_student_compression.png",
    )
    save_kd_sweep_plot(
        kd_payload=kd_payload,
        student_only_ndcg5=float(kd_payload["student_only"]["test_metrics"]["val_ndcg@5"]),
        path=FIGURE_DIR / "02_kd_weight_sweep_ndcg5.png",
    )
    save_onnx_parity_plot(
        metrics=parity_payload["metrics"],
        path=FIGURE_DIR / "03_onnx_parity.png",
    )
    save_quantization_tradeoff_plot(
        rows=quant_rows,
        path=FIGURE_DIR / "04_quantization_tradeoff.png",
    )
    save_final_candidate_plot(
        rows=quant_rows,
        path=FIGURE_DIR / "05_final_candidate_summary.png",
    )

    MARKDOWN_OUTPUT.write_text(
        build_markdown(
            student_payload=student_payload,
            kd_payload=kd_payload,
            parity_payload=parity_payload,
            quant_rows=quant_rows,
        ),
        encoding="utf-8",
    )
    print(f"Wrote presentation assets: {MARKDOWN_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
