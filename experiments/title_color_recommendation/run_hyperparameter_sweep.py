from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.run_full_training import (
    COLOR_DISTRIBUTION_KEY,
    TOP1_WCAG_PASS_RATE_KEY,
    TRAIN_LOSS_KEY,
    VAL_LOSS_KEY,
    VAL_NDCG_KEY,
    _color_distribution,
    _history_record_for_epoch,
    _markdown_image_path,
    _metric_is_better,
    resolve_project_path,
    run_training_loop,
    validate_training_config,
)
from src.models.fixed_palette_classifier import build_fixed_palette_resnet18
from src.title_color_recommendation.data.dataloader import (
    create_title_color_datasets,
    require_dataloader,
)
from src.title_color_recommendation.training.config import (
    TrainingConfig,
    training_config_from_mapping,
)
from src.title_color_recommendation.training.trainer import resolve_device

LOGGER = logging.getLogger(__name__)
DEFAULT_SWEEP_CONFIG = Path(
    "configs/title_color_recommendation/hyperparameter_sweep.json"
)
DEFAULT_OUTPUT_DIR = Path("outputs/hparam_sweep")
DEFAULT_REPORT_PATH = Path("outputs/reports/hyperparameter_sweep_report.md")
DEFAULT_RESULTS_CSV_PATH = Path("outputs/reports/hyperparameter_sweep_results.csv")
DEFAULT_VAL_LOSS_PLOT_PATH = Path("outputs/reports/hparam_sweep_val_loss.png")
DEFAULT_NDCG_PLOT_PATH = Path("outputs/reports/hparam_sweep_ndcg5.png")
DEFAULT_SAFETY_PLOT_PATH = Path("outputs/reports/hparam_sweep_safety.png")


@dataclass(frozen=True)
class SweepTrial:
    name: str
    config: TrainingConfig


@dataclass(frozen=True)
class SweepTrialResult:
    name: str
    config: TrainingConfig
    best_epoch: int
    selection_metric: str
    selection_metric_value: float
    best_val_loss: float
    best_val_ndcg_at_5: float
    best_top1_wcag_pass_rate: float
    best_max_color_share: float
    final_train_loss: float
    final_val_loss: float
    checkpoint_dir: Path
    log_path: Path


@dataclass(frozen=True)
class SweepResult:
    trials: list[SweepTrialResult]
    best_trial: SweepTrialResult
    report_path: Path
    results_csv_path: Path
    plot_paths: dict[str, Path]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a validation-only hyperparameter sweep."
    )
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP_CONFIG)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/title_color_recommendation"),
    )
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        default=DEFAULT_RESULTS_CSV_PATH,
    )
    parser.add_argument(
        "--val-loss-plot-path",
        type=Path,
        default=DEFAULT_VAL_LOSS_PLOT_PATH,
    )
    parser.add_argument(
        "--ndcg-plot-path",
        type=Path,
        default=DEFAULT_NDCG_PLOT_PATH,
    )
    parser.add_argument(
        "--safety-plot-path",
        type=Path,
        default=DEFAULT_SAFETY_PLOT_PATH,
    )
    parser.add_argument("--selection-metric", default=VAL_LOSS_KEY)
    parser.add_argument("--collapse-threshold", type=float, default=0.8)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def load_sweep_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    if not spec_path.exists():
        raise FileNotFoundError(f"sweep config not found: {spec_path}")
    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)
    if not isinstance(spec, Mapping):
        raise ValueError(f"sweep config must be a mapping: {spec_path}")
    if not isinstance(spec.get("trials"), list) or not spec["trials"]:
        raise ValueError("sweep config must contain a non-empty trials list")
    return dict(spec)


def _sanitize_trial_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    if not sanitized:
        raise ValueError("trial name must not be empty")
    return sanitized


def _limited_trials(
    trial_values: list[Mapping[str, Any]],
    *,
    max_trials: int | None,
) -> list[Mapping[str, Any]]:
    if max_trials is None:
        return trial_values
    if max_trials <= 0:
        raise ValueError(f"max_trials must be positive: {max_trials}")
    return trial_values[:max_trials]


def build_sweep_trials(
    spec: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[SweepTrial]:
    base_values = dict(spec.get("base") or {})
    trial_values = _limited_trials(spec["trials"], max_trials=args.max_trials)
    trials: list[SweepTrial] = []

    for index, raw_trial in enumerate(trial_values, start=1):
        values = dict(base_values)
        values.update(dict(raw_trial))
        raw_name = str(values.pop("name", f"trial_{index:02d}"))
        trial_name = _sanitize_trial_name(raw_name)

        for arg_name, config_name in (
            ("epochs", "epochs"),
            ("batch_size", "batch_size"),
            ("num_workers", "num_workers"),
            ("device", "device"),
            ("seed", "seed"),
        ):
            value = getattr(args, arg_name)
            if value is not None:
                values[config_name] = value

        values["best_metric"] = args.selection_metric
        trial_dir = resolve_project_path(args.output_dir) / trial_name
        values["checkpoint_dir"] = str(trial_dir / "checkpoints")
        values["log_path"] = str(trial_dir / "training_metrics.jsonl")

        config = training_config_from_mapping(values)
        validate_training_config(config)
        trials.append(SweepTrial(name=trial_name, config=config))

    return trials


def _dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    dataset_kwargs: dict[str, Any] = {}
    if args.labels_matrix is not None:
        dataset_kwargs["labels_matrix_path"] = args.labels_matrix
    if args.labels_soft is not None:
        dataset_kwargs["labels_soft_path"] = args.labels_soft
    return dataset_kwargs


def create_sweep_datasets(args: argparse.Namespace) -> dict[str, Any]:
    return create_title_color_datasets(
        splits=("train", "val"),
        data_root=args.data_root,
        project_root=PROJECT_ROOT,
        **_dataset_kwargs(args),
    )


def create_trial_loaders(
    datasets: Mapping[str, Any],
    config: TrainingConfig,
) -> dict[str, Any]:
    loader_cls = require_dataloader()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    device = resolve_device(config.device)
    return {
        "train": loader_cls(
            datasets["train"],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            generator=generator,
        ),
        "val": loader_cls(
            datasets["val"],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }


def _trial_result_from_history(
    *,
    trial: SweepTrial,
    history: list[dict[str, Any]],
    best_epoch: int,
    selection_metric_value: float,
) -> SweepTrialResult:
    best_record = _history_record_for_epoch(history, epoch=best_epoch)
    final_record = history[-1]
    best_distribution = _color_distribution(best_record)
    return SweepTrialResult(
        name=trial.name,
        config=trial.config,
        best_epoch=best_epoch,
        selection_metric=trial.config.best_metric,
        selection_metric_value=selection_metric_value,
        best_val_loss=float(best_record[VAL_LOSS_KEY]),
        best_val_ndcg_at_5=float(best_record[VAL_NDCG_KEY]),
        best_top1_wcag_pass_rate=float(best_record[TOP1_WCAG_PASS_RATE_KEY]),
        best_max_color_share=max(best_distribution),
        final_train_loss=float(final_record[TRAIN_LOSS_KEY]),
        final_val_loss=float(final_record[VAL_LOSS_KEY]),
        checkpoint_dir=resolve_project_path(trial.config.checkpoint_dir),
        log_path=resolve_project_path(trial.config.log_path),
    )


def run_trial(
    trial: SweepTrial,
    datasets: Mapping[str, Any],
    *,
    pretrained: bool,
) -> SweepTrialResult:
    LOGGER.info("starting trial=%s", trial.name)
    loaders = create_trial_loaders(datasets, trial.config)
    model = build_fixed_palette_resnet18(
        num_classes=trial.config.num_classes,
        pretrained=pretrained,
    )
    history, best_epoch, selection_metric_value, _best_state = run_training_loop(
        model,
        loaders["train"],
        loaders["val"],
        trial.config,
        checkpoint_dir=resolve_project_path(trial.config.checkpoint_dir),
        log_path=resolve_project_path(trial.config.log_path),
    )
    return _trial_result_from_history(
        trial=trial,
        history=history,
        best_epoch=best_epoch,
        selection_metric_value=selection_metric_value,
    )


def pick_best_trial(results: list[SweepTrialResult]) -> SweepTrialResult:
    if not results:
        raise ValueError("results must not be empty")
    best = results[0]
    for result in results[1:]:
        if _metric_is_better(
            candidate=result.selection_metric_value,
            best=best.selection_metric_value,
            metric_name=result.selection_metric,
        ):
            best = result
    return best


def sorted_results(results: list[SweepTrialResult]) -> list[SweepTrialResult]:
    if not results:
        return []
    metric_name = results[0].selection_metric
    reverse = not metric_name.endswith("loss")
    return sorted(
        results,
        key=lambda result: result.selection_metric_value,
        reverse=reverse,
    )


def write_results_csv(path: Path, results: list[SweepTrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "trial",
        "best_epoch",
        "selection_metric",
        "selection_metric_value",
        "best_val_loss",
        "best_val_ndcg@5",
        "best_top1_wcag_pass_rate",
        "best_max_color_share",
        "final_train_loss",
        "final_val_loss",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "scheduler",
        "checkpoint_dir",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(sorted_results(results), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "trial": result.name,
                    "best_epoch": result.best_epoch,
                    "selection_metric": result.selection_metric,
                    "selection_metric_value": result.selection_metric_value,
                    "best_val_loss": result.best_val_loss,
                    "best_val_ndcg@5": result.best_val_ndcg_at_5,
                    "best_top1_wcag_pass_rate": result.best_top1_wcag_pass_rate,
                    "best_max_color_share": result.best_max_color_share,
                    "final_train_loss": result.final_train_loss,
                    "final_val_loss": result.final_val_loss,
                    "learning_rate": result.config.learning_rate,
                    "weight_decay": result.config.weight_decay,
                    "batch_size": result.config.batch_size,
                    "scheduler": result.config.scheduler,
                    "checkpoint_dir": result.checkpoint_dir,
                    "log_path": result.log_path,
                }
            )


def _load_pyplot() -> Any:
    matplotlib_config_dir = PROJECT_ROOT / "outputs" / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_sweep_plots(
    results: list[SweepTrialResult],
    *,
    val_loss_plot_path: Path,
    ndcg_plot_path: Path,
    safety_plot_path: Path,
) -> dict[str, Path]:
    if not results:
        raise ValueError("results must not be empty")
    plt = _load_pyplot()
    ordered_results = sorted_results(results)
    labels = [result.name for result in ordered_results]
    positions = list(range(len(ordered_results)))

    val_loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    ndcg_plot_path.parent.mkdir(parents=True, exist_ok=True)
    safety_plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(positions, [result.best_val_loss for result in ordered_results])
    axis.set_title("Best Validation Loss by Trial")
    axis.set_ylabel("val_loss")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(val_loss_plot_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(
        positions,
        [result.best_val_ndcg_at_5 for result in ordered_results],
        color="#2563EB",
    )
    axis.set_title("Best Validation NDCG@5 by Trial")
    axis.set_ylabel("val_ndcg@5")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(ndcg_plot_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.8))
    width = 0.4
    left_positions = [position - width / 2 for position in positions]
    right_positions = [position + width / 2 for position in positions]
    axis.bar(
        left_positions,
        [result.best_top1_wcag_pass_rate for result in ordered_results],
        width=width,
        label="top1_wcag_pass_rate",
        color="#0F766E",
    )
    axis.bar(
        right_positions,
        [result.best_max_color_share for result in ordered_results],
        width=width,
        label="max_color_share",
        color="#C2410C",
    )
    axis.set_title("Validation Safety and Collapse Indicators")
    axis.set_ylabel("share")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(safety_plot_path, dpi=160)
    plt.close(figure)

    return {
        "val_loss": val_loss_plot_path,
        "ndcg": ndcg_plot_path,
        "safety": safety_plot_path,
    }


def _format_config_command(
    best_trial: SweepTrialResult,
    *,
    pretrained: bool,
) -> list[str]:
    config = best_trial.config
    lines = [
        "python experiments/title_color_recommendation/run_full_training.py \\",
        "  --config configs/title_color_recommendation/full_training.yaml \\",
        f"  --epochs {config.epochs} \\",
        f"  --learning-rate {config.learning_rate} \\",
        f"  --weight-decay {config.weight_decay} \\",
        f"  --batch-size {config.batch_size} \\",
        f"  --num-workers {config.num_workers} \\",
        f"  --device {config.device} \\",
        f"  --seed {config.seed} \\",
        f"  --scheduler {config.scheduler}",
    ]
    if pretrained:
        lines[-1] = f"{lines[-1]} \\"
        lines.append("  --pretrained")
    return lines


def write_sweep_report(
    path: Path,
    *,
    results: list[SweepTrialResult],
    best_trial: SweepTrialResult,
    plot_paths: Mapping[str, Path],
    results_csv_path: Path,
    collapse_threshold: float,
    pretrained: bool,
) -> None:
    ordered_results = sorted_results(results)
    lines = [
        "# Hyperparameter Sweep Report",
        "",
        "- selection_data: `val split only`",
        "- test_split_used: `False`",
        f"- selection_metric: `{best_trial.selection_metric}`",
        f"- best_trial: `{best_trial.name}`",
        f"- pretrained: `{pretrained}`",
        f"- collapse_threshold: `{collapse_threshold}`",
        f"- results_csv: `{results_csv_path}`",
        "",
        "## Best Trial",
        "",
        "| field | value |",
        "| --- | ---: |",
        f"| best_epoch | {best_trial.best_epoch} |",
        f"| best_val_loss | {best_trial.best_val_loss:.6f} |",
        f"| best_val_ndcg@5 | {best_trial.best_val_ndcg_at_5:.6f} |",
        (
            "| best_top1_wcag_pass_rate | "
            f"{best_trial.best_top1_wcag_pass_rate:.6f} |"
        ),
        f"| best_max_color_share | {best_trial.best_max_color_share:.6f} |",
        f"| learning_rate | {best_trial.config.learning_rate} |",
        f"| weight_decay | {best_trial.config.weight_decay} |",
        f"| batch_size | {best_trial.config.batch_size} |",
        f"| scheduler | {best_trial.config.scheduler} |",
        "",
        "## Plots",
        "",
        f"![Validation Loss]({_markdown_image_path(path, plot_paths['val_loss'])})",
        f"![Validation NDCG@5]({_markdown_image_path(path, plot_paths['ndcg'])})",
        f"![Safety]({_markdown_image_path(path, plot_paths['safety'])})",
        "",
        "## Trial Ranking",
        "",
        (
            "| rank | trial | lr | wd | batch | scheduler | best_epoch | "
            "val_loss | val_ndcg@5 | top1_wcag | max_color_share |"
        ),
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for rank, result in enumerate(ordered_results, start=1):
        lines.append(
            f"| {rank} | {result.name} | "
            f"{result.config.learning_rate} | "
            f"{result.config.weight_decay} | "
            f"{result.config.batch_size} | "
            f"{result.config.scheduler} | "
            f"{result.best_epoch} | "
            f"{result.best_val_loss:.6f} | "
            f"{result.best_val_ndcg_at_5:.6f} | "
            f"{result.best_top1_wcag_pass_rate:.6f} | "
            f"{result.best_max_color_share:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Run final training with the selected hyperparameters, then evaluate test.",
            "",
            "```bash",
            *_format_config_command(best_trial, pretrained=pretrained),
            "```",
            "",
            "## Notes",
            "",
            "- The test split is intentionally not used during hyperparameter selection.",
            "- Prefer low validation loss, then check WCAG pass rate and color collapse.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> SweepResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    spec = load_sweep_spec(resolve_project_path(args.sweep_config))
    trials = build_sweep_trials(spec, args)
    datasets = create_sweep_datasets(args)

    results = [
        run_trial(trial, datasets, pretrained=args.pretrained)
        for trial in trials
    ]
    best_trial = pick_best_trial(results)
    results_csv_path = resolve_project_path(args.results_csv_path)
    write_results_csv(results_csv_path, results)
    plot_paths = write_sweep_plots(
        results,
        val_loss_plot_path=resolve_project_path(args.val_loss_plot_path),
        ndcg_plot_path=resolve_project_path(args.ndcg_plot_path),
        safety_plot_path=resolve_project_path(args.safety_plot_path),
    )
    report_path = resolve_project_path(args.report_path)
    write_sweep_report(
        report_path,
        results=results,
        best_trial=best_trial,
        plot_paths=plot_paths,
        results_csv_path=results_csv_path,
        collapse_threshold=args.collapse_threshold,
        pretrained=args.pretrained,
    )
    return SweepResult(
        trials=results,
        best_trial=best_trial,
        report_path=report_path,
        results_csv_path=results_csv_path,
        plot_paths=plot_paths,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
