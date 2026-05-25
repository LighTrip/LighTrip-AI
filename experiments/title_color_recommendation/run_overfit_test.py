from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.run_full_training import (
    _color_distribution,
    _require_history,
)
from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    save_configured_figure,
    top_color_rows,
)
from src.models.fixed_palette_classifier import build_fixed_palette_resnet18
from src.title_color_recommendation.data.dataset import (
    TitleColorDataset,
    load_label_matrix,
    load_soft_label_vectors,
    read_manifest_rows,
)
from src.title_color_recommendation.training.config import TrainingConfig
from src.title_color_recommendation.training.trainer import (
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    set_training_seed,
    train_one_epoch,
    validate,
)

LOGGER = logging.getLogger(__name__)
TRAIN_LOSS_KEY = "train_loss"
TRAIN_EVAL_LOSS_KEY = "train_eval_loss"
TRAIN_NDCG_KEY = "train_ndcg@5"
TOP1_WCAG_PASS_RATE_KEY = "top1_wcag_pass_rate"
TOP5_ANY_WCAG_PASS_RATE_KEY = "top5_any_wcag_pass_rate"
COLOR_DISTRIBUTION_KEY = "color_distribution"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a 128-sample overfit test for title color recommendation."
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("data/title_color_recommendation/splits/train.csv"),
    )
    parser.add_argument(
        "--labels-matrix",
        type=Path,
        default=Path(
            "data/title_color_recommendation/processed/labels/labels_matrix.npy"
        ),
    )
    parser.add_argument(
        "--labels-soft",
        type=Path,
        default=Path(
            "data/title_color_recommendation/processed/labels/labels_soft.csv"
        ),
    )
    parser.add_argument("--subset-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--scheduler", default="none", choices=("none", "cosine"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--collapse-threshold", type=float, default=0.8)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/reports/overfit_test_report.md"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("outputs/checkpoints/overfit_test.pt"),
    )
    parser.add_argument(
        "--subset-manifest-path",
        type=Path,
        default=Path("outputs/reports/overfit_train_subset_128.csv"),
    )
    parser.add_argument(
        "--loss-plot-path",
        type=Path,
        default=Path("outputs/reports/overfit_loss_curve.png"),
    )
    parser.add_argument(
        "--ndcg-plot-path",
        type=Path,
        default=Path("outputs/reports/overfit_ndcg_curve.png"),
    )
    parser.add_argument(
        "--color-plot-path",
        type=Path,
        default=Path("outputs/reports/overfit_color_distribution.png"),
    )
    return parser.parse_args(argv)


def select_subset_rows(
    rows: list[Mapping[str, Any]],
    *,
    subset_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    if subset_size <= 0:
        raise ValueError(f"subset_size must be positive: {subset_size}")
    if len(rows) < subset_size:
        raise ValueError(
            f"train manifest has fewer rows than requested: "
            f"rows={len(rows)}, subset_size={subset_size}"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(rows), generator=generator)[:subset_size].tolist()
    selected = [dict(rows[int(index)]) for index in indices]
    selected.sort(key=lambda row: str(row.get("id") or ""))
    return selected


def write_subset_manifest(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_overfit_loader(
    *,
    rows: list[Mapping[str, Any]],
    labels_matrix_path: Path,
    labels_soft_path: Path,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    labels_matrix = load_label_matrix(labels_matrix_path, mmap_mode="r")
    pseudo_scores_by_id, wcag_pass_by_id = load_soft_label_vectors(
        labels_soft_path,
        (str(row["id"]) for row in rows),
        num_classes=int(labels_matrix.shape[1]),
    )
    dataset = TitleColorDataset(
        "train",
        project_root=PROJECT_ROOT,
        rows=rows,
        labels_matrix=labels_matrix,
        pseudo_scores_by_id=pseudo_scores_by_id,
        wcag_pass_by_id=wcag_pass_by_id,
        augment=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )


def metric_record(
    *,
    epoch: int,
    train_loss: float | None,
    train_metrics: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "epoch": epoch,
        TRAIN_EVAL_LOSS_KEY: train_metrics.val_loss,
        TRAIN_NDCG_KEY: train_metrics.val_ndcg_at_5,
        TOP1_WCAG_PASS_RATE_KEY: train_metrics.top1_wcag_pass_rate,
        TOP5_ANY_WCAG_PASS_RATE_KEY: train_metrics.top5_any_wcag_pass_rate,
        COLOR_DISTRIBUTION_KEY: train_metrics.color_distribution,
    }
    if train_loss is not None:
        record[TRAIN_LOSS_KEY] = train_loss
    return record


def _record_value(
    record: Mapping[str, Any],
    key: str,
) -> float | None:
    value = record.get(key)
    return None if value is None else float(value)


def write_overfit_plots(
    history: list[dict[str, Any]],
    *,
    loss_plot_path: Path,
    ndcg_plot_path: Path,
    color_plot_path: Path,
) -> dict[str, Path]:
    _require_history(history)

    plt = load_pyplot(PROJECT_ROOT)

    epochs = [int(record["epoch"]) for record in history]
    train_loss = [_record_value(record, TRAIN_LOSS_KEY) for record in history]
    train_eval_loss = [
        float(record[TRAIN_EVAL_LOSS_KEY])
        for record in history
    ]
    train_ndcg = [
        float(record[TRAIN_NDCG_KEY])
        for record in history
    ]
    final_distribution = _color_distribution(history[-1])

    figure, axis = plt.subplots(figsize=(8, 4.5))
    train_loss_epochs = [
        epoch
        for epoch, value in zip(epochs, train_loss)
        if value is not None
    ]
    train_loss_values = [
        value
        for value in train_loss
        if value is not None
    ]
    if train_loss_values:
        axis.plot(
            train_loss_epochs,
            train_loss_values,
            marker="o",
            label="train_loss",
        )
    axis.plot(
        epochs,
        train_eval_loss,
        marker="o",
        label="train_eval_loss",
    )
    save_configured_figure(
        plt,
        figure,
        axis,
        loss_plot_path,
        title="Overfit Loss Curve",
        xlabel="epoch",
        ylabel="KL loss",
        legend=True,
    )

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, train_ndcg, marker="o", color="#2563EB")
    save_configured_figure(
        plt,
        figure,
        axis,
        ndcg_plot_path,
        title="Overfit NDCG@5 Curve",
        xlabel="epoch",
        ylabel="train_ndcg@5",
        ylim=(0.0, 1.05),
    )

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(final_distribution)), final_distribution, color="#0F766E")
    color_axis_limit = max(max(final_distribution) * 1.15, 0.01)
    save_configured_figure(
        plt,
        figure,
        axis,
        color_plot_path,
        title="Final Top-1 Color Distribution",
        xlabel="palette_id",
        ylabel="share",
        ylim=(0.0, color_axis_limit),
        grid_axis="y",
    )

    return {
        "loss": loss_plot_path,
        "ndcg": ndcg_plot_path,
        "color_distribution": color_plot_path,
    }


def _markdown_image_path(report_path: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return image_path.as_posix()


def success_checks(
    history: list[dict[str, Any]],
    *,
    collapse_threshold: float,
) -> dict[str, bool]:
    _require_history(history)

    train_loss_values = [
        float(record[TRAIN_LOSS_KEY])
        for record in history
        if TRAIN_LOSS_KEY in record
    ]
    ndcg_values = [float(record[TRAIN_NDCG_KEY]) for record in history]
    final_distribution = _color_distribution(history[-1])
    max_color_share = max(float(value) for value in final_distribution)
    active_colors = sum(1 for value in final_distribution if float(value) > 0.0)
    return {
        "train_loss_decreased": (
            len(train_loss_values) >= 2
            and train_loss_values[-1] < train_loss_values[0]
        ),
        "train_ndcg_increased": (
            len(ndcg_values) >= 2
            and ndcg_values[-1] > ndcg_values[0]
        ),
        "not_collapsed": max_color_share < collapse_threshold and active_colors > 1,
    }


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    history: list[dict[str, Any]],
    checks: dict[str, bool],
    checkpoint_path: Path,
    subset_manifest_path: Path,
    plot_paths: Mapping[str, Path] | None = None,
) -> None:
    _require_history(history)

    initial = history[0]
    final = history[-1]
    status = "PASS" if all(checks.values()) else "REVIEW"
    max_color_share = max(_color_distribution(final))

    lines = [
        "# Overfit Test Report",
        "",
        f"- status: `{status}`",
        f"- subset_size: `{args.subset_size}`",
        f"- epochs: `{args.epochs}`",
        f"- batch_size: `{args.batch_size}`",
        f"- learning_rate: `{args.learning_rate}`",
        f"- weight_decay: `{args.weight_decay}`",
        f"- pretrained: `{args.pretrained}`",
        f"- checkpoint: `{checkpoint_path}`",
        f"- subset_manifest: `{subset_manifest_path}`",
        "",
        "## Checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: `{'PASS' if passed else 'REVIEW'}`")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| metric | initial | final |",
            "| --- | ---: | ---: |",
            (
                f"| train_eval_loss | "
                f"{float(initial[TRAIN_EVAL_LOSS_KEY]):.6f} | "
                f"{float(final[TRAIN_EVAL_LOSS_KEY]):.6f} |"
            ),
            (
                f"| train_ndcg@5 | "
                f"{float(initial[TRAIN_NDCG_KEY]):.6f} | "
                f"{float(final[TRAIN_NDCG_KEY]):.6f} |"
            ),
            (
                f"| top1_wcag_pass_rate | "
                f"{float(initial[TOP1_WCAG_PASS_RATE_KEY]):.6f} | "
                f"{float(final[TOP1_WCAG_PASS_RATE_KEY]):.6f} |"
            ),
            (
                f"| top5_any_wcag_pass_rate | "
                f"{float(initial[TOP5_ANY_WCAG_PASS_RATE_KEY]):.6f} | "
                f"{float(final[TOP5_ANY_WCAG_PASS_RATE_KEY]):.6f} |"
            ),
            f"| max_color_share | - | {max_color_share:.6f} |",
            "",
        ]
    )

    if plot_paths:
        lines.extend(
            [
                "## Plots",
                "",
                (
                    "![Loss Curve]("
                    f"{_markdown_image_path(path, plot_paths['loss'])})"
                ),
                (
                    "![NDCG Curve]("
                    f"{_markdown_image_path(path, plot_paths['ndcg'])})"
                ),
                (
                    "![Color Distribution]("
                    f"{_markdown_image_path(path, plot_paths['color_distribution'])})"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Final Color Distribution",
            "",
            *top_color_rows(_color_distribution(final)),
            "",
            "## History",
            "",
            "| epoch | train_loss | train_eval_loss | train_ndcg@5 | max_color_share |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for record in history:
        distribution = _color_distribution(record)
        row_max_share = max(float(value) for value in distribution)
        train_loss = record.get(TRAIN_LOSS_KEY)
        train_loss_text = "-" if train_loss is None else f"{float(train_loss):.6f}"
        lines.append(
            f"| {record['epoch']} | {train_loss_text} | "
            f"{float(record[TRAIN_EVAL_LOSS_KEY]):.6f} | "
            f"{float(record[TRAIN_NDCG_KEY]):.6f} | {row_max_share:.6f} |"
        )

    if not all(checks.values()):
        lines.extend(
            [
                "",
                "## Review Notes",
                "",
                "- Check palette id ordering.",
                "- Check pseudo_scores and target_distribution shapes.",
                "- Check target_distribution row sums.",
                "- Check mask channel and input normalization.",
                "- Try reducing/increasing learning rate.",
                "- Recheck KL loss direction.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_training_seed(args.seed)

    rows = read_manifest_rows(PROJECT_ROOT / args.train_manifest)
    subset_rows = select_subset_rows(
        rows,
        subset_size=args.subset_size,
        seed=args.seed,
    )
    write_subset_manifest(PROJECT_ROOT / args.subset_manifest_path, subset_rows)

    train_loader = create_overfit_loader(
        rows=subset_rows,
        labels_matrix_path=PROJECT_ROOT / args.labels_matrix,
        labels_soft_path=PROJECT_ROOT / args.labels_soft,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        scheduler=args.scheduler,
        checkpoint_dir=str((PROJECT_ROOT / args.checkpoint_path).parent),
        best_metric="train_eval_loss",
        seed=args.seed,
    )
    device = torch.device(
        "cuda"
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    model = build_fixed_palette_resnet18(pretrained=args.pretrained)
    model.to(device)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    history: list[dict[str, Any]] = []
    initial_metrics = validate(
        model,
        train_loader,
        device=device,
        num_classes=config.num_classes,
    )
    history.append(
        metric_record(epoch=0, train_loss=None, train_metrics=initial_metrics)
    )

    best_loss = initial_metrics.val_loss
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
        )
        train_metrics = validate(
            model,
            train_loader,
            device=device,
            num_classes=config.num_classes,
        )
        if scheduler is not None:
            scheduler.step()

        record = metric_record(
            epoch=epoch,
            train_loss=train_loss,
            train_metrics=train_metrics,
        )
        history.append(record)
        if train_metrics.val_loss < best_loss:
            best_loss = train_metrics.val_loss
            save_checkpoint(
                PROJECT_ROOT / args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=config,
                metrics=record,
                best_metric_value=best_loss,
            )

        LOGGER.info(
            "epoch=%s train_loss=%.6f train_eval_loss=%.6f train_ndcg@5=%.6f",
            epoch,
            train_loss,
            train_metrics.val_loss,
            train_metrics.val_ndcg_at_5,
        )

    if not (PROJECT_ROOT / args.checkpoint_path).exists():
        final_record = history[-1]
        save_checkpoint(
            PROJECT_ROOT / args.checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=args.epochs,
            config=config,
            metrics=final_record,
            best_metric_value=best_loss,
        )

    checks = success_checks(history, collapse_threshold=args.collapse_threshold)
    plot_paths = write_overfit_plots(
        history,
        loss_plot_path=PROJECT_ROOT / args.loss_plot_path,
        ndcg_plot_path=PROJECT_ROOT / args.ndcg_plot_path,
        color_plot_path=PROJECT_ROOT / args.color_plot_path,
    )
    write_report(
        PROJECT_ROOT / args.report_path,
        args=args,
        history=history,
        checks=checks,
        checkpoint_path=args.checkpoint_path,
        subset_manifest_path=args.subset_manifest_path,
        plot_paths=plot_paths,
    )
    return history


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
