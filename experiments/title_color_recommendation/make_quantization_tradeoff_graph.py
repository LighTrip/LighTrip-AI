from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.make_titlenet_presentation_assets import (
    MOBILE_PROXY_METRICS,
    QAT_QUANTIZATION_METRICS,
    QUANTIZATION_METRICS,
    STATIC_SWEEP_METRICS,
    QuantizationPresentationRow,
    build_quantization_rows,
    load_json,
)


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / (
    "outputs/reports/presentation/titlenet_student_quantization/figures/"
    "quantization_tradeoff_right.png"
)
PRACTICAL_CANDIDATES = {
    "Student KD FP32",
    "Student KD PTQ FP16",
    "Student KD PTQ INT8 Dynamic",
    "Student QAT FP32",
    "Student QAT + PTQ FP16",
    "Student QAT + PTQ INT8 Dynamic",
}
SELECTED_LABEL = "Student QAT + PTQ FP16"


def precision_for(row: QuantizationPresentationRow) -> str:
    if "FP16" in row.method:
        return "FP16"
    if "INT8" in row.method:
        return "INT8"
    return "FP32"


def marker_for(row: QuantizationPresentationRow) -> str:
    if row.label.startswith("Student QAT"):
        return "s"
    return "o"


def short_label(row: QuantizationPresentationRow) -> str:
    label = row.label
    replacements = {
        "Student KD FP32": "Student\nFP32",
        "Student KD PTQ FP16": "Student\nPTQ FP16",
        "Student KD PTQ INT8 Dynamic": "Student\nPTQ INT8",
        "Student QAT FP32": "QAT\nFP32",
        "Student QAT + PTQ FP16": "QAT+PTQ\nFP16",
        "Student QAT + PTQ INT8 Dynamic": "QAT+PTQ\nINT8",
    }
    return replacements.get(label, label)


def annotation_offset(row: QuantizationPresentationRow) -> tuple[int, int]:
    offsets = {
        "Student KD FP32": (-42, -26),
        "Student KD PTQ FP16": (-73, 8),
        "Student KD PTQ INT8 Dynamic": (8, 18),
        "Student QAT FP32": (8, -18),
        "Student QAT + PTQ FP16": (10, 12),
        "Student QAT + PTQ INT8 Dynamic": (12, -17),
    }
    return offsets.get(row.label, (6, 6))


def candidate_rows() -> list[QuantizationPresentationRow]:
    rows = build_quantization_rows(
        quantization_payload=load_json(QUANTIZATION_METRICS),
        qat_quantization_payload=load_json(QAT_QUANTIZATION_METRICS),
        static_payload=load_json(STATIC_SWEEP_METRICS),
        mobile_payload=load_json(MOBILE_PROXY_METRICS),
    )
    return [
        row
        for row in rows
        if row.label in PRACTICAL_CANDIDATES
        and row.default_p95_ms is not None
        and row.ndcg_at_5 is not None
    ]


def save_tradeoff_graph(rows: list[QuantizationPresentationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    precision_colors = {
        "FP32": "#7A7A7A",
        "FP16": "#1F77B4",
        "INT8": "#D55E00",
    }
    precision_faces = {
        "FP32": "#E1E1E1",
        "FP16": "#DCEBFA",
        "INT8": "#F8D8C7",
    }

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.5,
        }
    )

    fig, axis = plt.subplots(figsize=(6.0, 4.2))
    for row in rows:
        precision = precision_for(row)
        selected = row.label == SELECTED_LABEL
        axis.scatter(
            row.default_p95_ms,
            row.ndcg_at_5,
            s=165 if selected else 95,
            marker=marker_for(row),
            facecolor=precision_faces[precision],
            edgecolor="#111111" if selected else precision_colors[precision],
            linewidth=2.0 if selected else 1.2,
            zorder=4 if selected else 3,
        )
        if selected:
            axis.scatter(
                row.default_p95_ms,
                row.ndcg_at_5,
                s=55,
                marker="*",
                facecolor="#111111",
                edgecolor="#111111",
                linewidth=0.6,
                zorder=5,
            )
        axis.annotate(
            short_label(row),
            (row.default_p95_ms, row.ndcg_at_5),
            xytext=annotation_offset(row),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color="#111111" if selected else "#333333",
            weight="bold" if selected else "normal",
        )

    x_values = [float(row.default_p95_ms or 0.0) for row in rows]
    y_values = [float(row.ndcg_at_5 or 0.0) for row in rows]
    axis.set_xlim(min(x_values) - 0.05, max(x_values) + 0.08)
    axis.set_ylim(min(y_values) - 0.00055, max(y_values) + 0.00055)
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("NDCG@5")
    axis.set_title("Quantization Trade-off")
    axis.grid(axis="both", color="#D0D0D0", linewidth=0.6, alpha=0.65)
    axis.set_axisbelow(True)

    precision_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=precision_faces[name],
            markeredgecolor=color,
            markeredgewidth=1.2,
            markersize=7,
            label=name,
        )
        for name, color in precision_colors.items()
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#F5F5F5",
            markeredgecolor="#333333",
            markersize=7,
            label="PTQ",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#F5F5F5",
            markeredgecolor="#333333",
            markersize=7,
            label="QAT + PTQ",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#111111",
            markeredgecolor="#111111",
            markersize=8,
            label="Selected",
        ),
    ]
    legend = axis.legend(
        handles=[*precision_handles, *method_handles],
        frameon=False,
        ncols=2,
        loc="upper right",
    )
    axis.add_artist(legend)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    output_path = DEFAULT_OUTPUT_PATH
    rows = candidate_rows()
    save_tradeoff_graph(rows, output_path)
    print(f"Wrote figure: {output_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
