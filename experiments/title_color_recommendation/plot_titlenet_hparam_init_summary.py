from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.plot_utils import load_pyplot


DEFAULT_HPARAM_CSV = Path("outputs/reports/titlenet_hyperparameter_summary.csv")
DEFAULT_OUTPUT_PATH = Path("outputs/reports/titlenet_hparam_init_paper_summary.png")
TRIAL_GELU_BASE = "simple_cnn_m_res_se_gelu"
TRIAL_GELU_LR3E4 = "simple_cnn_m_res_se_gelu_lr3e-4"
TRIAL_HARDSWISH = "simple_cnn_m_res_se_hardswish"
BEST_TRIAL = "simple_cnn_m_res_se_gelu_wd1e-4_drop0.2"
LEARNING_CURVE_TRIALS = (
    TRIAL_GELU_BASE,
    BEST_TRIAL,
    TRIAL_GELU_LR3E4,
    TRIAL_HARDSWISH,
)
TRIAL_LABELS = {
    TRIAL_GELU_BASE: "GELU\nlr 5e-4\nwd 5e-4\ndrop .3",
    TRIAL_GELU_LR3E4: "GELU\nlr 3e-4\nwd 5e-4\ndrop .3",
    BEST_TRIAL: "GELU\nlr 5e-4\nwd 1e-4\ndrop .2",
    TRIAL_HARDSWISH: "Hardswish\nlr 5e-4\nwd 5e-4\ndrop .3",
}
CURVE_LABELS = {
    TRIAL_GELU_BASE: "GELU, wd 5e-4, drop .3",
    BEST_TRIAL: "GELU, wd 1e-4, drop .2",
    TRIAL_GELU_LR3E4: "GELU, lr 3e-4",
    TRIAL_HARDSWISH: "Hardswish",
}


@dataclass(frozen=True)
class TrialResult:
    trial: str
    best_val_loss: float
    best_val_ndcg5: float
    log_path: Path


@dataclass(frozen=True)
class EpochPoint:
    epoch: int
    val_ndcg5: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TitLeNet hyperparameter sweep summary figure."
    )
    parser.add_argument("--hparam-csv", type=Path, default=DEFAULT_HPARAM_CSV)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dpi", type=int, default=450)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_inside_project(PROJECT_ROOT, path, must_exist=must_exist)


def load_trial_results(path: Path) -> list[TrialResult]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [trial_result_from_row(row) for row in csv.DictReader(handle)]


def trial_result_from_row(row: dict[str, str]) -> TrialResult:
    return TrialResult(
        trial=row["trial"],
        best_val_loss=float(row["best_val_loss"]),
        best_val_ndcg5=float(row["best_val_ndcg5"]),
        log_path=resolve_path(row["log_path"], must_exist=True),
    )


def load_epoch_points(path: Path) -> list[EpochPoint]:
    points: list[EpochPoint] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                points.append(
                    EpochPoint(
                        epoch=int(payload["epoch"]),
                        val_ndcg5=float(payload["val_ndcg@5"]),
                    )
                )
    return points


def sort_trials_for_plot(trials: list[TrialResult]) -> list[TrialResult]:
    order = {trial_name: index for index, trial_name in enumerate(TRIAL_LABELS)}
    return sorted(trials, key=lambda trial: order.get(trial.trial, len(order)))


def trial_label(trial: TrialResult) -> str:
    return TRIAL_LABELS.get(trial.trial, trial.trial.replace("_", "\n"))


def bar_colors(trials: list[TrialResult]) -> list[str]:
    colors = []
    for trial in trials:
        if trial.trial == BEST_TRIAL:
            colors.append("#2A9D8F")
        else:
            colors.append("#9AA4B2")
    return colors


def configure_paper_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def add_panel_label(axis: Any, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def add_ndcg_panel(axis: Any, trials: list[TrialResult]) -> None:
    values = [trial.best_val_ndcg5 for trial in trials]
    labels = [trial_label(trial) for trial in trials]
    axis.bar(range(len(trials)), values, color=bar_colors(trials), width=0.72)
    axis.set_title("Validation ranking quality")
    axis.set_ylabel("Best val NDCG@5")
    axis.set_xticks(range(len(trials)), labels, rotation=0)
    axis.set_ylim(min(values) - 0.00045, max(values) + 0.00035)
    axis.grid(axis="y", alpha=0.25)
    annotate_bars(axis, values, fmt="{:.4f}", offset=0.00004)
    add_panel_label(axis, "(a)")


def add_loss_panel(axis: Any, trials: list[TrialResult]) -> None:
    values = [trial.best_val_loss for trial in trials]
    labels = [trial_label(trial) for trial in trials]
    axis.bar(range(len(trials)), values, color=bar_colors(trials), width=0.72)
    axis.set_title("Validation KL loss")
    axis.set_ylabel("Best val loss")
    axis.set_xticks(range(len(trials)), labels, rotation=0)
    axis.set_ylim(min(values) - 0.00025, max(values) + 0.00045)
    axis.grid(axis="y", alpha=0.25)
    annotate_bars(axis, values, fmt="{:.4f}", offset=0.00003)
    add_panel_label(axis, "(b)")


def annotate_bars(axis: Any, values: list[float], *, fmt: str, offset: float) -> None:
    for index, value in enumerate(values):
        axis.text(
            index,
            value + offset,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
        )


def add_curve_panel(axis: Any, trials: list[TrialResult]) -> None:
    trial_by_name = {trial.trial: trial for trial in trials}
    curve_colors = ["#2A9D8F", "#6C757D", "#8E44AD", "#D1495B"]
    for trial_name, color in zip(LEARNING_CURVE_TRIALS, curve_colors):
        trial = trial_by_name.get(trial_name)
        if trial is None:
            continue
        points = load_epoch_points(trial.log_path)
        epochs = [point.epoch for point in points]
        ndcg = [point.val_ndcg5 for point in points]
        axis.plot(epochs, ndcg, marker="o", markersize=2.6, linewidth=1.45, color=color)
        axis.plot([], [], label=CURVE_LABELS.get(trial_name, trial_name), color=color)
    axis.set_title("Validation NDCG@5 convergence")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Val NDCG@5")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    add_panel_label(axis, "(c)")


def save_figure(figure: Any, path: Path, *, dpi: int, save_pdf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def run(args: argparse.Namespace) -> Path:
    hparam_path = resolve_path(args.hparam_csv, must_exist=True)
    output_path = resolve_path(args.output_path)
    trials = [
        trial
        for trial in sort_trials_for_plot(load_trial_results(hparam_path))
        if trial.trial in TRIAL_LABELS
    ]

    plt = load_pyplot(PROJECT_ROOT)
    configure_paper_style(plt)
    figure = plt.figure(figsize=(12.8, 6.4))
    grid = figure.add_gridspec(2, 2, hspace=0.48, wspace=0.24)
    add_ndcg_panel(figure.add_subplot(grid[0, 0]), trials)
    add_loss_panel(figure.add_subplot(grid[0, 1]), trials)
    add_curve_panel(figure.add_subplot(grid[1, :]), trials)
    figure.suptitle("TitLeNet hyperparameter sweep summary", y=0.99)
    save_figure(figure, output_path, dpi=args.dpi, save_pdf=not args.no_pdf)
    plt.close(figure)
    return output_path


def main() -> None:
    path = run(parse_args())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
