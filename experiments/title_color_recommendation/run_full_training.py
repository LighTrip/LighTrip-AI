from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    markdown_image_path,
    top_color_rows,
)
from src.models.fixed_palette_classifier import build_fixed_palette_resnet18
from src.title_color_recommendation.data.dataloader import (
    create_title_color_dataloaders,
)
from src.title_color_recommendation.training.config import (
    TrainingConfig,
    load_training_config,
    training_config_from_mapping,
)
from src.title_color_recommendation.training.metrics import ValidationMetrics
from src.title_color_recommendation.training.trainer import (
    append_jsonl_log,
    create_optimizer,
    create_scheduler,
    resolve_device,
    save_checkpoint,
    set_training_seed,
    train_one_epoch,
    validate,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints")
DEFAULT_LOG_PATH = Path("outputs/logs/training_metrics.jsonl")
DEFAULT_REPORT_PATH = Path("outputs/reports/full_training_report.md")
DEFAULT_LOSS_PLOT_PATH = Path("outputs/reports/loss_curve.png")
DEFAULT_NDCG_PLOT_PATH = Path("outputs/reports/ndcg5_curve.png")
DEFAULT_COLOR_PLOT_PATH = Path("outputs/reports/color_distribution.png")
TRAIN_LOSS_KEY = "train_loss"
VAL_LOSS_KEY = "val_loss"
VAL_NDCG_KEY = "val_ndcg@5"
TOP1_WCAG_PASS_RATE_KEY = "top1_wcag_pass_rate"
COLOR_DISTRIBUTION_KEY = "color_distribution"


@dataclass(frozen=True)
class FullTrainingResult:
    history: list[dict[str, Any]]
    test_metrics: ValidationMetrics
    best_epoch: int
    best_metric_value: float
    dataset_sizes: dict[str, int]
    checkpoint_paths: dict[str, Path]
    plot_paths: dict[str, Path]
    report_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the title color recommendation model on full splits."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/title_color_recommendation"),
    )
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--scheduler",
        choices=("none", "cosine", "plateau"),
        default=None,
    )
    parser.add_argument("--best-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--collapse-threshold", type=float, default=0.8)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        default=DEFAULT_LOSS_PLOT_PATH,
    )
    parser.add_argument(
        "--ndcg-plot-path",
        type=Path,
        default=DEFAULT_NDCG_PLOT_PATH,
    )
    parser.add_argument(
        "--color-plot-path",
        type=Path,
        default=DEFAULT_COLOR_PLOT_PATH,
    )
    return parser.parse_args(argv)


def build_training_config(args: argparse.Namespace) -> TrainingConfig:
    if args.config is None:
        defaults = TrainingConfig(
            batch_size=64,
            epochs=20,
            learning_rate=3e-4,
            weight_decay=1e-4,
            num_workers=4,
            device="cuda",
            scheduler="cosine",
            checkpoint_dir=str(DEFAULT_CHECKPOINT_DIR),
            log_path=str(DEFAULT_LOG_PATH),
            best_metric=VAL_LOSS_KEY,
            seed=42,
        )
    else:
        defaults = load_training_config(PROJECT_ROOT / args.config)

    overrides: dict[str, Any] = {}
    for arg_name, config_name in (
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("num_workers", "num_workers"),
        ("device", "device"),
        ("scheduler", "scheduler"),
        ("best_metric", "best_metric"),
        ("seed", "seed"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            overrides[config_name] = value

    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.log_path is not None:
        overrides["log_path"] = str(args.log_path)

    config = training_config_from_mapping(overrides, defaults=defaults)
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    if config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive: {config.batch_size}")
    if config.epochs <= 0:
        raise ValueError(f"epochs must be positive: {config.epochs}")
    if config.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive: {config.learning_rate}")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative: {config.weight_decay}")
    if config.num_classes <= 0:
        raise ValueError(f"num_classes must be positive: {config.num_classes}")
    if config.num_workers < 0:
        raise ValueError(f"num_workers must be non-negative: {config.num_workers}")


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def reset_jsonl_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def _scheduler_step(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    *,
    val_loss: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
        return
    scheduler.step()


def _metric_is_better(
    *,
    candidate: float,
    best: float | None,
    metric_name: str,
) -> bool:
    if best is None:
        return True
    if metric_name.endswith("loss"):
        return candidate < best
    return candidate > best


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _require_history(history: list[dict[str, Any]]) -> None:
    if not history:
        raise ValueError("history must not be empty")


def _color_distribution(record: Mapping[str, Any]) -> list[float]:
    distribution = [
        float(value)
        for value in record[COLOR_DISTRIBUTION_KEY]
    ]
    if not distribution:
        raise ValueError("color_distribution must not be empty")
    return distribution


def _validation_record(
    *,
    epoch: int,
    train_loss: float,
    metrics: ValidationMetrics,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        TRAIN_LOSS_KEY: train_loss,
        **metrics.as_dict(),
    }


def run_training_loop(
    model: nn.Module,
    train_loader: Any,
    val_loader: Any,
    config: TrainingConfig,
    *,
    checkpoint_dir: Path,
    log_path: Path,
) -> tuple[list[dict[str, Any]], int, float, dict[str, Tensor]]:
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reset_jsonl_log(log_path)

    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_epoch = 0
    best_model_state: dict[str, Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
        )
        validation = validate(
            model,
            val_loader,
            device=device,
            num_classes=config.num_classes,
        )
        _scheduler_step(scheduler, val_loss=validation.val_loss)

        record = _validation_record(
            epoch=epoch,
            train_loss=train_loss,
            metrics=validation,
        )
        history.append(record)
        append_jsonl_log(log_path, record)

        if config.best_metric not in record:
            raise ValueError(f"best_metric not found in metrics: {config.best_metric}")
        metric_value = float(record[config.best_metric])
        is_best = _metric_is_better(
            candidate=metric_value,
            best=best_metric_value,
            metric_name=config.best_metric,
        )
        if is_best:
            best_metric_value = metric_value
            best_epoch = epoch
            best_model_state = _clone_model_state(model)

        checkpoint_best_value = (
            metric_value if best_metric_value is None else best_metric_value
        )
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            metrics=record,
            best_metric_value=checkpoint_best_value,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                metrics=record,
                best_metric_value=checkpoint_best_value,
            )

        LOGGER.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_ndcg@5=%.6f "
            "top1_wcag_pass_rate=%.6f",
            epoch,
            train_loss,
            validation.val_loss,
            validation.val_ndcg_at_5,
            validation.top1_wcag_pass_rate,
        )

    if best_metric_value is None or best_model_state is None:
        raise RuntimeError("training did not produce a best checkpoint state")
    return history, best_epoch, best_metric_value, best_model_state


def evaluate_best_model(
    model: nn.Module,
    test_loader: Any,
    config: TrainingConfig,
    best_model_state: Mapping[str, Tensor],
) -> ValidationMetrics:
    device = resolve_device(config.device)
    model.load_state_dict(best_model_state)
    model.to(device)
    return validate(
        model,
        test_loader,
        device=device,
        num_classes=config.num_classes,
    )


def _history_record_for_epoch(
    history: list[dict[str, Any]],
    *,
    epoch: int,
) -> dict[str, Any]:
    for record in history:
        if int(record["epoch"]) == epoch:
            return record
    raise ValueError(f"epoch not found in history: {epoch}")


def write_training_plots(
    history: list[dict[str, Any]],
    *,
    best_epoch: int,
    test_metrics: ValidationMetrics,
    loss_plot_path: Path,
    ndcg_plot_path: Path,
    color_plot_path: Path,
) -> dict[str, Path]:
    _require_history(history)
    plt = load_pyplot(PROJECT_ROOT)

    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record[TRAIN_LOSS_KEY]) for record in history]
    val_loss = [float(record[VAL_LOSS_KEY]) for record in history]
    val_ndcg = [float(record[VAL_NDCG_KEY]) for record in history]
    best_record = _history_record_for_epoch(history, epoch=best_epoch)
    best_val_distribution = _color_distribution(best_record)
    test_distribution = list(test_metrics.color_distribution)

    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    ndcg_plot_path.parent.mkdir(parents=True, exist_ok=True)
    color_plot_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, train_loss, marker="o", label="train_loss")
    axis.plot(epochs, val_loss, marker="o", label="val_loss")
    axis.set_title("Full Training Loss Curve")
    axis.set_xlabel("epoch")
    axis.set_ylabel("KL loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_plot_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, val_ndcg, marker="o", color="#2563EB")
    axis.set_title("Validation NDCG@5 Curve")
    axis.set_xlabel("epoch")
    axis.set_ylabel("val_ndcg@5")
    axis.set_ylim(0.0, 1.05)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(ndcg_plot_path, dpi=160)
    plt.close(figure)

    palette_ids = list(range(len(test_distribution)))
    left_positions = [palette_id - 0.2 for palette_id in palette_ids]
    right_positions = [palette_id + 0.2 for palette_id in palette_ids]
    max_share = max(max(best_val_distribution), max(test_distribution), 0.01)
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(
        left_positions,
        best_val_distribution,
        width=0.4,
        label="best_val",
        color="#2563EB",
    )
    axis.bar(
        right_positions,
        test_distribution,
        width=0.4,
        label="test",
        color="#0F766E",
    )
    axis.set_title("Top-1 Color Distribution")
    axis.set_xlabel("palette_id")
    axis.set_ylabel("share")
    axis.set_ylim(0.0, max_share * 1.15)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(color_plot_path, dpi=160)
    plt.close(figure)

    return {
        "loss": loss_plot_path,
        "ndcg": ndcg_plot_path,
        "color_distribution": color_plot_path,
    }


def _not_collapsed(distribution: list[float], *, threshold: float) -> bool:
    max_share = max(distribution)
    active_colors = sum(1 for value in distribution if value > 0.0)
    return max_share < threshold and active_colors > 1


def success_checks(
    history: list[dict[str, Any]],
    test_metrics: ValidationMetrics,
    *,
    collapse_threshold: float,
) -> dict[str, bool]:
    _require_history(history)
    final_record = history[-1]
    final_val_distribution = _color_distribution(final_record)
    test_distribution = list(test_metrics.color_distribution)
    return {
        "train_loss_recorded": all(TRAIN_LOSS_KEY in record for record in history),
        "val_loss_recorded": all(VAL_LOSS_KEY in record for record in history),
        "val_ndcg_recorded": all(VAL_NDCG_KEY in record for record in history),
        "val_not_collapsed": _not_collapsed(
            final_val_distribution,
            threshold=collapse_threshold,
        ),
        "test_not_collapsed": _not_collapsed(
            test_distribution,
            threshold=collapse_threshold,
        ),
    }


def _metric_line(
    name: str,
    *,
    best_record: Mapping[str, Any],
    final_record: Mapping[str, Any],
    test_metrics: ValidationMetrics,
) -> str:
    test_values = test_metrics.as_dict()
    return (
        f"| {name} | {float(best_record[name]):.6f} | "
        f"{float(final_record[name]):.6f} | {float(test_values[name]):.6f} |"
    )


def write_full_training_report(
    path: Path,
    *,
    config: TrainingConfig,
    history: list[dict[str, Any]],
    test_metrics: ValidationMetrics,
    best_epoch: int,
    best_metric_value: float,
    dataset_sizes: Mapping[str, int],
    checkpoint_paths: Mapping[str, Path],
    plot_paths: Mapping[str, Path],
    collapse_threshold: float,
    pretrained: bool,
) -> None:
    _require_history(history)
    checks = success_checks(
        history,
        test_metrics,
        collapse_threshold=collapse_threshold,
    )
    status = "PASS" if all(checks.values()) else "REVIEW"
    final_record = history[-1]
    best_record = _history_record_for_epoch(history, epoch=best_epoch)
    test_distribution = list(test_metrics.color_distribution)
    max_test_color_share = max(test_distribution)

    lines = [
        "# Full Training Report",
        "",
        f"- status: `{status}`",
        f"- epochs: `{config.epochs}`",
        f"- batch_size: `{config.batch_size}`",
        f"- learning_rate: `{config.learning_rate}`",
        f"- weight_decay: `{config.weight_decay}`",
        f"- scheduler: `{config.scheduler}`",
        f"- best_metric: `{config.best_metric}`",
        f"- best_epoch: `{best_epoch}`",
        f"- best_metric_value: `{best_metric_value:.6f}`",
        f"- pretrained: `{pretrained}`",
        "",
        "## Datasets",
        "",
        "| split | samples |",
        "| --- | ---: |",
    ]
    for split in ("train", "val", "test"):
        lines.append(f"| {split} | {int(dataset_sizes[split])} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- best_checkpoint: `{checkpoint_paths['best']}`",
            f"- latest_checkpoint: `{checkpoint_paths['latest']}`",
            f"- log_path: `{config.log_path}`",
            "",
            "## Checks",
            "",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"- {name}: `{'PASS' if passed else 'REVIEW'}`")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| metric | best_val | final_val | test |",
            "| --- | ---: | ---: | ---: |",
            _metric_line(
                VAL_LOSS_KEY,
                best_record=best_record,
                final_record=final_record,
                test_metrics=test_metrics,
            ),
            _metric_line(
                VAL_NDCG_KEY,
                best_record=best_record,
                final_record=final_record,
                test_metrics=test_metrics,
            ),
            _metric_line(
                TOP1_WCAG_PASS_RATE_KEY,
                best_record=best_record,
                final_record=final_record,
                test_metrics=test_metrics,
            ),
            f"| max_color_share | - | - | {max_test_color_share:.6f} |",
            "",
            "## Plots",
            "",
            f"![Loss Curve]({markdown_image_path(path, plot_paths['loss'])})",
            f"![NDCG Curve]({markdown_image_path(path, plot_paths['ndcg'])})",
            (
                "![Color Distribution]("
                f"{markdown_image_path(path, plot_paths['color_distribution'])})"
            ),
            "",
            "## Test Top Colors",
            "",
            *top_color_rows(test_distribution),
            "",
            "## History",
            "",
            (
                "| epoch | train_loss | val_loss | val_ndcg@5 | "
                "top1_wcag_pass_rate | max_color_share |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for record in history:
        distribution = _color_distribution(record)
        lines.append(
            f"| {record['epoch']} | {float(record[TRAIN_LOSS_KEY]):.6f} | "
            f"{float(record[VAL_LOSS_KEY]):.6f} | "
            f"{float(record[VAL_NDCG_KEY]):.6f} | "
            f"{float(record[TOP1_WCAG_PASS_RATE_KEY]):.6f} | "
            f"{max(distribution):.6f} |"
        )

    if not all(checks.values()):
        lines.extend(
            [
                "",
                "## Review Notes",
                "",
                "- Check whether train/val/test split manifests overlap.",
                "- Check target_distribution row sums and palette id ordering.",
                "- Check whether color distribution collapsed to a single color.",
                "- Compare val/test curves before trusting final model quality.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    dataset_kwargs: dict[str, Any] = {}
    if args.labels_matrix is not None:
        dataset_kwargs["labels_matrix_path"] = args.labels_matrix
    if args.labels_soft is not None:
        dataset_kwargs["labels_soft_path"] = args.labels_soft
    return dataset_kwargs


def run(args: argparse.Namespace) -> FullTrainingResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = build_training_config(args)
    set_training_seed(config.seed)
    device = resolve_device(config.device)

    loaders = create_title_color_dataloaders(
        batch_size=config.batch_size,
        splits=("train", "val", "test"),
        data_root=args.data_root,
        project_root=PROJECT_ROOT,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        seed=config.seed,
        dataset_kwargs=_dataset_kwargs(args),
    )
    dataset_sizes = {
        split: len(loaders[split].dataset)
        for split in ("train", "val", "test")
    }

    model = build_fixed_palette_resnet18(
        num_classes=config.num_classes,
        pretrained=args.pretrained,
    )
    checkpoint_dir = resolve_project_path(config.checkpoint_dir)
    log_path = resolve_project_path(config.log_path)
    history, best_epoch, best_metric_value, best_state = run_training_loop(
        model,
        loaders["train"],
        loaders["val"],
        config,
        checkpoint_dir=checkpoint_dir,
        log_path=log_path,
    )
    test_metrics = evaluate_best_model(
        model,
        loaders["test"],
        config,
        best_state,
    )
    plot_paths = write_training_plots(
        history,
        best_epoch=best_epoch,
        test_metrics=test_metrics,
        loss_plot_path=resolve_project_path(args.loss_plot_path),
        ndcg_plot_path=resolve_project_path(args.ndcg_plot_path),
        color_plot_path=resolve_project_path(args.color_plot_path),
    )
    checkpoint_paths = {
        "best": checkpoint_dir / "checkpoint_best.pt",
        "latest": checkpoint_dir / "checkpoint_latest.pt",
    }
    report_path = resolve_project_path(args.report_path)
    write_full_training_report(
        report_path,
        config=config,
        history=history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        dataset_sizes=dataset_sizes,
        checkpoint_paths=checkpoint_paths,
        plot_paths=plot_paths,
        collapse_threshold=args.collapse_threshold,
        pretrained=args.pretrained,
    )
    return FullTrainingResult(
        history=history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        dataset_sizes=dataset_sizes,
        checkpoint_paths=checkpoint_paths,
        plot_paths=plot_paths,
        report_path=report_path,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
