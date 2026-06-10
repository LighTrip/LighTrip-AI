from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.plot_utils import load_pyplot
from experiments.title_color_recommendation.make_titlenet_presentation_assets import (
    HEADER_NDCG_AT_5,
    HEADER_SIZE_MB,
    KD_SWEEP_METRICS,
    LABEL_STUDENT_QAT_FP32,
    LABEL_STUDENT_QAT_PTQ_FP16,
    LABEL_STUDENT_QAT_PTQ_INT8_DYNAMIC,
    LABEL_STUDENT_QAT_STATIC_INT8_SWEEP_BEST,
    MOBILE_PROXY_METRICS,
    ONNX_PARITY_METRICS,
    PREFIX_STUDENT_KD,
    PREFIX_STUDENT_QAT,
    QAT_QUANTIZATION_METRICS,
    QUANTIZATION_METRICS,
    STATIC_SWEEP_METRICS,
    STUDENT_EXPERIMENT_METRICS,
    QuantizationPresentationRow,
    build_quantization_rows,
    fmt,
    load_json,
    markdown_table,
    nested_float,
    pct,
    reduction,
)

plt = load_pyplot(PROJECT_ROOT)


OUTPUT_DIR = PROJECT_ROOT / (
    "outputs/reports/presentation/titlenet_student_quantization_paper_style"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
MARKDOWN_OUTPUT = OUTPUT_DIR / "paper_style_tables_and_figures.md"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "savefig.bbox": "tight",
        }
    )


def figure_caption(title: str, body: str) -> list[str]:
    return [
        f"**{title}.** {body}",
        "",
    ]


def paper_note(text: str) -> list[str]:
    return [
        f"*Note.* {text}",
        "",
    ]


def selected_rows(
    rows: list[QuantizationPresentationRow],
    labels: set[str],
) -> list[QuantizationPresentationRow]:
    return [row for row in rows if row.label in labels]


def save_resource_reduction_figure(
    *,
    teacher_params: float,
    student_params: float,
    teacher_size: float,
    student_size: float,
    teacher_latency: float,
    student_latency: float,
    path: Path,
) -> None:
    labels = ["Parameters", "Model size", "Latency"]
    teacher_values = [100.0, 100.0, 100.0]
    student_values = [
        student_params / teacher_params * 100.0,
        student_size / teacher_size * 100.0,
        student_latency / teacher_latency * 100.0,
    ]

    fig, axis = plt.subplots(figsize=(6.8, 3.8))
    x_positions = range(len(labels))
    width = 0.34
    axis.bar(
        [x - width / 2 for x in x_positions],
        teacher_values,
        width,
        label="Teacher",
        color="#B8B8B8",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axis.bar(
        [x + width / 2 for x in x_positions],
        student_values,
        width,
        label="Student",
        color="#3B6EA8",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axis.set_xticks(list(x_positions), labels)
    axis.set_ylabel("Relative value (%)")
    axis.set_ylim(0, 115)
    axis.set_title("Resource Reduction of the Student Model")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    for index, value in enumerate(student_values):
        axis.text(index + width / 2, value + 2.5, f"{value:.1f}%", ha="center", fontsize=9)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_kd_sweep_figure(
    *,
    kd_payload: dict,
    student_only_ndcg5: float,
    path: Path,
) -> None:
    rows = [row for row in kd_payload.get("rows", []) if isinstance(row, dict)]
    fig, axis = plt.subplots(figsize=(6.8, 4.0))
    for phase, label, marker, color in [
        ("from_scratch", "KD from scratch", "o", "#8C564B"),
        ("warm_start", "Warm-start KD", "s", "#2F6B4F"),
    ]:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        phase_rows.sort(key=lambda row: float(row["base_loss_weight"]))
        x_values = [float(row["base_loss_weight"]) for row in phase_rows]
        y_values = [float(row["test_ndcg@5"]) for row in phase_rows]
        axis.plot(
            x_values,
            y_values,
            marker=marker,
            markersize=5,
            linewidth=1.8,
            label=label,
            color=color,
        )
    axis.axhline(
        student_only_ndcg5,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label="Student-only",
    )
    axis.set_xlabel("Base loss weight")
    axis.set_ylabel(HEADER_NDCG_AT_5)
    axis.set_title("Effect of KD Weight and Initialization Strategy")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_parity_figure(*, parity_metrics: dict, path: Path) -> None:
    labels = ["Top-1", "Top-3", "Top-5"]
    values = [
        float(parity_metrics["top1_agreement"]),
        float(parity_metrics["top3_agreement"]),
        float(parity_metrics["top5_agreement"]),
    ]
    fig, axis = plt.subplots(figsize=(5.6, 3.4))
    bars = axis.bar(
        labels,
        values,
        color="#4F7CAC",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axis.set_ylim(0.95, 1.005)
    axis.set_ylabel("Agreement")
    axis.set_title("PyTorch-ONNX Output Consistency")
    axis.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.001,
            f"{value * 100.0:.1f}%",
            ha="center",
            fontsize=9,
        )
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_quantization_tradeoff_figure(
    *,
    rows: list[QuantizationPresentationRow],
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.6))
    color_map = {
        "Selected": "#1B7837",
        "Rejected": "#B2182B",
        "Reference": "#777777",
        "Passed": "#2166AC",
        "Passed, not final": "#F4A582",
    }
    for row in rows:
        if row.default_p95_ms is None or row.ndcg_at_5 is None or row.size_mb is None:
            continue
        axis.scatter(
            row.default_p95_ms,
            row.ndcg_at_5,
            s=850 * row.size_mb,
            color=color_map.get(row.decision, "#777777"),
            edgecolor="#222222",
            linewidth=0.6,
            alpha=0.86,
        )
        if row.decision in {"Selected", "Rejected", "Reference"}:
            short_label = (
                row.label.replace(PREFIX_STUDENT_QAT, "QAT ")
                .replace(PREFIX_STUDENT_KD, "KD ")
                .replace(" + ", "\n+ ")
            )
            axis.annotate(
                short_label,
                (row.default_p95_ms, row.ndcg_at_5),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
    axis.set_xlabel("CPU proxy P95 latency (ms)")
    axis.set_ylabel(HEADER_NDCG_AT_5)
    axis.set_title("Accuracy-Latency-Size Trade-off")
    axis.grid(alpha=0.25)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_final_candidate_figure(
    *,
    rows: list[QuantizationPresentationRow],
    path: Path,
) -> None:
    labels = {
        LABEL_STUDENT_QAT_FP32,
        LABEL_STUDENT_QAT_PTQ_FP16,
        LABEL_STUDENT_QAT_PTQ_INT8_DYNAMIC,
        LABEL_STUDENT_QAT_STATIC_INT8_SWEEP_BEST,
    }
    final_rows = selected_rows(rows, labels)
    x_labels = [
        row.label.replace(PREFIX_STUDENT_QAT, "").replace("+ PTQ ", "")
        for row in final_rows
    ]
    top1_values = [float(row.top1_agreement or 0.0) for row in final_rows]
    size_values = [float(row.size_mb or 0.0) for row in final_rows]
    p95_values = [float(row.default_p95_ms or 0.0) for row in final_rows]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    for axis, values, title, ylabel, color in [
        (axes[0], top1_values, "Top-1 agreement", "Agreement", "#4F7CAC"),
        (axes[1], size_values, "Top-1 ONNX size", "MB", "#6A994E"),
        (axes[2], p95_values, "CPU proxy P95", "ms", "#BC6C25"),
    ]:
        axis.bar(
            x_labels,
            values,
            color=color,
            edgecolor="#333333",
            linewidth=0.5,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0.90, 1.02)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_markdown(
    *,
    student_payload: dict,
    kd_payload: dict,
    parity_payload: dict,
    rows: list[QuantizationPresentationRow],
) -> str:
    profiles = student_payload["profiles"]
    teacher_profile = profiles["teacher"]
    student_profile = profiles["student"]
    teacher_params = float(teacher_profile["total_parameters"])
    student_params = float(student_profile["total_parameters"])
    teacher_size = float(teacher_profile["model_size_mb"])
    student_size = float(student_profile["model_size_mb"])
    teacher_latency = nested_float(teacher_profile, ("batch1_latency", "inference_time_ms")) or 0.0
    student_latency = nested_float(student_profile, ("batch1_latency", "inference_time_ms")) or 0.0
    student_only = kd_payload["student_only"]
    kd_rows = [row for row in kd_payload.get("rows", []) if isinstance(row, dict)]
    parity_metrics = parity_payload["metrics"]

    lines: list[str] = [
        "# Paper-Style Tables and Figures for TitLeNet Student Optimization",
        "",
        "이 문서는 발표 자료에 그대로 넣을 수 있도록 논문 결과 섹션 형식으로 정리한 표와 그림이다.",
        "",
        "## Student Architecture Compression",
        "",
    ]
    lines.extend(
        figure_caption(
            "Table 1",
            "Resource comparison between the original TitLeNet teacher and the lightweight student model.",
        )
    )
    lines.extend(
        markdown_table(
            ["Model", "Parameters", HEADER_SIZE_MB, "Batch1 Latency (ms)", "Activation"],
            [
                [
                    "TitLeNet Teacher",
                    f"{teacher_params:,.0f}",
                    fmt(teacher_size),
                    fmt(teacher_latency),
                    str(student_payload["teacher"]["activation"]),
                ],
                [
                    "TitLeNet Student",
                    f"{student_params:,.0f}",
                    fmt(student_size),
                    fmt(student_latency),
                    str(student_payload["student"]["activation"]),
                ],
                [
                    "Relative reduction",
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
            "![Figure 1](figures/fig_1_resource_reduction.png)",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Figure 1",
            "Relative resource usage of the student model compared with the teacher model.",
        )
    )
    lines.extend(
        paper_note(
            "Latency in Table 1 is PyTorch CPU batch1 latency from the student experiment profile."
        )
    )

    lines.extend(
        [
            "## Knowledge Distillation Study",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Table 2",
            "Comparison of Student-only, KD from scratch, and warm-start KD under different base/KD loss weights.",
        )
    )
    kd_table_rows = [
        [
            "Student-only",
            "-",
            "-",
            fmt(nested_float(student_only, ("test_metrics", "val_ndcg@3")), 6),
            fmt(nested_float(student_only, ("test_metrics", "val_ndcg@5")), 6),
            "-",
        ]
    ]
    for row in kd_rows:
        kd_table_rows.append(
            [
                str(row["phase"]).replace("_", " "),
                str(row["trial"]).replace("kd_", ""),
                f"{float(row['base_loss_weight']):.1f}:{float(row['distillation_loss_weight']):.1f}",
                fmt(float(row["test_ndcg@3"]), 6),
                fmt(float(row["test_ndcg@5"]), 6),
                fmt(float(row["teacher_top1_agreement"]), 6),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Training strategy",
                "KD ratio",
                "Base:KD",
                "NDCG@3",
                HEADER_NDCG_AT_5,
                "Teacher top-1 agreement",
            ],
            kd_table_rows,
        )
    )
    lines.extend(
        [
            "",
            "![Figure 2](figures/fig_2_kd_weight_sweep.png)",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Figure 2",
            f"{HEADER_NDCG_AT_5} trend according to KD weight and initialization strategy.",
        )
    )
    lines.extend(
        paper_note(
            "Warm-start KD consistently outperformed KD from scratch in this sweep, and `kd_90_10` was selected for the following ONNX and quantization experiments."
        )
    )

    lines.extend(
        [
            "## ONNX Export and Parity Validation",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Table 3",
            "ONNX export specification and PyTorch-ONNX parity validation.",
        )
    )
    lines.extend(
        markdown_table(
            ["Item", "Value"],
            [
                ["Input tensor", "`[1, 4, 36, 136]`, `float32`, NCHW"],
                ["Logits ONNX output", "`[1, 32]`, `float32`"],
                ["Deployment ONNX output", "`[1]`, `int64`, top-1 palette index"],
                ["Validation split / samples", f"`{parity_metrics['split']}` / `{parity_metrics['sample_count']}`"],
                ["Top-1 / Top-3 / Top-5 agreement", f"{pct(float(parity_metrics['top1_agreement']))} / {pct(float(parity_metrics['top3_agreement']))} / {pct(float(parity_metrics['top5_agreement']))}"],
                ["Max / mean absolute difference", f"{float(parity_metrics['max_abs_diff']):.2e} / {float(parity_metrics['mean_abs_diff']):.2e}"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Figure 3](figures/fig_3_onnx_parity.png)",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Figure 3",
            "Top-k agreement between PyTorch and ONNX Runtime outputs.",
        )
    )

    lines.extend(
        [
            "## Quantization Study",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Table 4",
            "Quantization results using top-1 ONNX models and ONNX Runtime CPU proxy latency.",
        )
    )
    lines.extend(
        markdown_table(
            [
                "Model",
                "Optimization",
                "Top-1 Agr.",
                "NDCG@3",
                HEADER_NDCG_AT_5,
                HEADER_SIZE_MB,
                "P50/P95 (ms)",
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
                    row.decision,
                ]
                for row in rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Figure 4](figures/fig_4_quantization_tradeoff.png)",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Figure 4",
            "Accuracy-latency-size trade-off of FP32, FP16, and INT8 candidates.",
        )
    )
    lines.extend(
        paper_note(
            "CPU proxy latency is measured with ONNX Runtime CPUExecutionProvider using top-1 ONNX models. It is intended as a proxy measurement, not a replacement for React Native release-build profiling."
        )
    )

    lines.extend(
        [
            "## Final Candidate Selection",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Table 5",
            "Final deployment candidate comparison among QAT-based models.",
        )
    )
    final_rows = selected_rows(
        rows,
        {
            LABEL_STUDENT_QAT_FP32,
            LABEL_STUDENT_QAT_PTQ_FP16,
            LABEL_STUDENT_QAT_PTQ_INT8_DYNAMIC,
            LABEL_STUDENT_QAT_STATIC_INT8_SWEEP_BEST,
        },
    )
    lines.extend(
        markdown_table(
            [
                "Candidate",
                "Top-1 Agr.",
                HEADER_NDCG_AT_5,
                HEADER_SIZE_MB,
                "P95 (ms)",
                "Decision",
                "Rationale",
            ],
            [
                [
                    row.label.replace(PREFIX_STUDENT_QAT, ""),
                    fmt(row.top1_agreement, 2),
                    fmt(row.ndcg_at_5, 6),
                    fmt(row.size_mb),
                    fmt(row.default_p95_ms),
                    row.decision,
                    (
                        "Selected due to preserved top-1 agreement and compact size"
                        if row.decision == "Selected"
                        else "Rejected due to top-1 degradation"
                        if row.decision == "Rejected"
                        else "Reference or secondary candidate"
                    ),
                ]
                for row in final_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "![Figure 5](figures/fig_5_final_candidate_comparison.png)",
            "",
        ]
    )
    lines.extend(
        figure_caption(
            "Figure 5",
            "Final QAT candidate comparison in terms of top-1 agreement, model size, and CPU proxy P95 latency.",
        )
    )
    lines.extend(
        [
            "### Slide-Ready Conclusion",
            "",
            f"The `{LABEL_STUDENT_QAT_PTQ_FP16}` model was selected as the final on-device candidate because it preserved 100% top-1 agreement while reducing the top-1 ONNX model size from 0.292 MB to 0.153 MB. Although static INT8 variants achieved competitive proxy latency, they were rejected because their top-1 agreement did not satisfy the deployment threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    student_payload = dict(load_json(STUDENT_EXPERIMENT_METRICS))
    kd_payload = dict(load_json(KD_SWEEP_METRICS))
    parity_payload = dict(load_json(ONNX_PARITY_METRICS))
    quantization_payload = load_json(QUANTIZATION_METRICS)
    qat_quantization_payload = load_json(QAT_QUANTIZATION_METRICS)
    static_payload = load_json(STATIC_SWEEP_METRICS)
    mobile_payload = load_json(MOBILE_PROXY_METRICS)
    rows = build_quantization_rows(
        quantization_payload=quantization_payload,
        qat_quantization_payload=qat_quantization_payload,
        static_payload=static_payload,
        mobile_payload=mobile_payload,
    )

    teacher_profile = student_payload["profiles"]["teacher"]
    student_profile = student_payload["profiles"]["student"]
    teacher_params = float(teacher_profile["total_parameters"])
    student_params = float(student_profile["total_parameters"])
    teacher_size = float(teacher_profile["model_size_mb"])
    student_size = float(student_profile["model_size_mb"])
    teacher_latency = nested_float(teacher_profile, ("batch1_latency", "inference_time_ms")) or 0.0
    student_latency = nested_float(student_profile, ("batch1_latency", "inference_time_ms")) or 0.0

    save_resource_reduction_figure(
        teacher_params=teacher_params,
        student_params=student_params,
        teacher_size=teacher_size,
        student_size=student_size,
        teacher_latency=teacher_latency,
        student_latency=student_latency,
        path=FIGURE_DIR / "fig_1_resource_reduction.png",
    )
    save_kd_sweep_figure(
        kd_payload=kd_payload,
        student_only_ndcg5=float(kd_payload["student_only"]["test_metrics"]["val_ndcg@5"]),
        path=FIGURE_DIR / "fig_2_kd_weight_sweep.png",
    )
    save_parity_figure(
        parity_metrics=parity_payload["metrics"],
        path=FIGURE_DIR / "fig_3_onnx_parity.png",
    )
    save_quantization_tradeoff_figure(
        rows=rows,
        path=FIGURE_DIR / "fig_4_quantization_tradeoff.png",
    )
    save_final_candidate_figure(
        rows=rows,
        path=FIGURE_DIR / "fig_5_final_candidate_comparison.png",
    )
    MARKDOWN_OUTPUT.write_text(
        build_markdown(
            student_payload=student_payload,
            kd_payload=kd_payload,
            parity_payload=parity_payload,
            rows=rows,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote paper-style assets: {MARKDOWN_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures: {FIGURE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
