from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import run_full_training
from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    markdown_image_path,
)
from src.models.fixed_palette_classifier import (
    DEFAULT_INPUT_SHAPE,
    count_total_parameters,
    count_trainable_parameters,
)
from src.models.title_color_model_registry import (
    available_model_names,
    build_title_color_model,
    normalize_model_name,
)
from src.title_color_recommendation.training.trainer import resolve_device


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/title_color_recommendation/model_comparison.yaml")
DEFAULT_RESULTS_CSV = Path("outputs/reports/model_comparison_results.csv")
DEFAULT_REPORT_PATH = Path("outputs/reports/model_comparison_report.md")
DEFAULT_LATENCY_PLOT = Path("outputs/reports/model_comparison_latency.png")
DEFAULT_LOSS_PLOT = Path("outputs/reports/model_comparison_loss_curve.png")
DEFAULT_NDCG_PLOT = Path("outputs/reports/model_comparison_ndcg5_curve.png")
RESULT_FIELDS = [
    "model_name",
    "dropout",
    "weight_init",
    "activation",
    "total_parameters",
    "trainable_parameters",
    "model_size_mb",
    "latency_device",
    "batch_size",
    "inference_time_ms",
    "images_per_second",
    "trained",
    "best_epoch",
    "test_loss",
    "test_ndcg@3",
    "test_ndcg@5",
    "top1_wcag_pass_rate",
    "top5_any_wcag_pass_rate",
    "max_color_share",
    "report_path",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare title color recommendation model families."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--models", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=None)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight-init", default=None)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--latency-plot-path", type=Path, default=DEFAULT_LATENCY_PLOT)
    parser.add_argument("--loss-plot-path", type=Path, default=DEFAULT_LOSS_PLOT)
    parser.add_argument("--ndcg-plot-path", type=Path, default=DEFAULT_NDCG_PLOT)
    return parser.parse_args(argv)


def is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_project_path(
    value: str | Path,
    *,
    must_exist: bool = False,
    description: str = "path",
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve(strict=False)
    project_root = PROJECT_ROOT.resolve()
    if not is_relative_to(resolved, project_root):
        raise ValueError(f"{description} must be inside project root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {value}")
    return resolved


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def configured_models(config: Mapping[str, Any], override: str) -> list[str]:
    if override.strip():
        raw_models = [name.strip() for name in override.split(",") if name.strip()]
    else:
        raw_models = list(config.get("models") or available_model_names())
    models = [normalize_model_name(name) for name in raw_models]
    if not models:
        raise ValueError("at least one model is required")
    return models


def nested_value(
    config: Mapping[str, Any],
    section: str,
    key: str,
    default: Any,
) -> Any:
    section_payload = config.get(section) or {}
    if not isinstance(section_payload, Mapping):
        return default
    return section_payload.get(key, default)


def model_size_mb(model: nn.Module) -> float:
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (parameter_bytes + buffer_bytes) / (1024 * 1024)


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    warmup_steps: int,
    benchmark_steps: int,
) -> dict[str, float]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive: {batch_size}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative: {warmup_steps}")
    if benchmark_steps <= 0:
        raise ValueError(f"benchmark_steps must be positive: {benchmark_steps}")

    model = model.to(device)
    model.eval()
    example = torch.zeros((batch_size, *DEFAULT_INPUT_SHAPE), device=device)
    with torch.no_grad():
        for _index in range(warmup_steps):
            model(example)
        synchronize_if_needed(device)
        start = time.perf_counter()
        for _index in range(benchmark_steps):
            model(example)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start

    return {
        "inference_time_ms": (elapsed * 1000.0) / benchmark_steps,
        "images_per_second": (batch_size * benchmark_steps) / max(elapsed, 1e-12),
    }


def training_args_for_model(
    *,
    model_name: str,
    config: Mapping[str, Any],
    epochs_override: int | None,
    pretrained: bool,
    dropout: float,
    weight_init: str,
    activation: str,
) -> argparse.Namespace:
    training = config.get("training") or {}
    if not isinstance(training, Mapping):
        training = {}
    epochs = int(epochs_override or training.get("epochs", 20))
    batch_size = int(training.get("batch_size", 64))
    learning_rate = float(training.get("learning_rate", 5e-4))
    weight_decay = float(training.get("weight_decay", 1e-4))
    scheduler = str(training.get("scheduler", "cosine"))
    num_workers = int(training.get("num_workers", 4))
    device = str(training.get("device", "cuda"))

    output_root = Path("outputs")
    report_root = output_root / "reports" / "model_comparison" / model_name
    return run_full_training.parse_args(
        [
            "--config",
            "configs/title_color_recommendation/full_training.yaml",
            "--model-name",
            model_name,
            "--learning-rate",
            str(learning_rate),
            "--weight-decay",
            str(weight_decay),
            "--batch-size",
            str(batch_size),
            "--dropout",
            str(dropout),
            "--weight-init",
            weight_init,
            "--activation",
            activation,
            "--epochs",
            str(epochs),
            "--scheduler",
            scheduler,
            "--num-workers",
            str(num_workers),
            "--device",
            device,
            "--checkpoint-dir",
            str(output_root / "checkpoints" / "model_comparison" / model_name),
            "--log-path",
            str(output_root / "logs" / "model_comparison" / f"{model_name}.jsonl"),
            "--report-path",
            str(report_root / "full_training_report.md"),
            "--loss-plot-path",
            str(report_root / "loss_curve.png"),
            "--ndcg-plot-path",
            str(report_root / "ndcg5_curve.png"),
            "--color-plot-path",
            str(report_root / "color_distribution.png"),
            *(["--pretrained"] if pretrained else []),
        ]
    )


def row_from_training_result(
    *,
    row: dict[str, Any],
    result: run_full_training.FullTrainingResult,
) -> dict[str, Any]:
    metrics = result.test_metrics.as_dict()
    distribution = list(result.test_metrics.color_distribution)
    row.update(
        {
            "trained": True,
            "best_epoch": result.best_epoch,
            "test_loss": metrics["val_loss"],
            "test_ndcg@3": metrics["val_ndcg@3"],
            "test_ndcg@5": metrics["val_ndcg@5"],
            "top1_wcag_pass_rate": metrics["top1_wcag_pass_rate"],
            "top5_any_wcag_pass_rate": metrics["top5_any_wcag_pass_rate"],
            "max_color_share": max(distribution),
            "report_path": result.report_path.as_posix(),
        }
    )
    return row


def write_results_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latency_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt = load_pyplot(PROJECT_ROOT)
    labels = [str(row["model_name"]) for row in rows]
    values = [float(row["inference_time_ms"]) for row in rows]
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(labels, values, color="#2563EB")
    axis.set_title("Model Inference Latency")
    axis.set_xlabel("model")
    axis.set_ylabel("ms / batch")
    axis.tick_params(axis="x", rotation=35)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_training_curve_plots(
    *,
    loss_plot_path: Path,
    ndcg_plot_path: Path,
    histories: Mapping[str, list[dict[str, Any]]],
) -> None:
    if not histories:
        return
    loss_plot_path.parent.mkdir(parents=True, exist_ok=True)
    ndcg_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt = load_pyplot(PROJECT_ROOT)

    figure, axis = plt.subplots(figsize=(9, 5))
    for model_name, history in histories.items():
        epochs = [int(record["epoch"]) for record in history]
        values = [float(record["val_loss"]) for record in history]
        axis.plot(epochs, values, marker="o", label=model_name)
    axis.set_title("Validation Loss by Model")
    axis.set_xlabel("epoch")
    axis.set_ylabel("val_loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(loss_plot_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for model_name, history in histories.items():
        epochs = [int(record["epoch"]) for record in history]
        values = [float(record["val_ndcg@5"]) for record in history]
        axis.plot(epochs, values, marker="o", label=model_name)
    axis.set_title("Validation NDCG@5 by Model")
    axis.set_xlabel("epoch")
    axis.set_ylabel("val_ndcg@5")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(ndcg_plot_path, dpi=160)
    plt.close(figure)


def format_optional_float(value: Any) -> str:
    if value in {"", None}:
        return "-"
    return f"{float(value):.6f}"


def write_report(
    path: Path,
    *,
    rows: list[Mapping[str, Any]],
    latency_plot_path: Path,
    loss_plot_path: Path,
    ndcg_plot_path: Path,
    trained: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Model Comparison Report",
        "",
        f"- trained: `{trained}`",
        f"- latency_plot: `{latency_plot_path}`",
        "",
        "## Summary",
        "",
        (
            "| model | init | act | dropout | params | size_mb | latency_ms | img_per_sec | "
            "test_ndcg@3 | test_ndcg@5 | top1_wcag | top5_any_wcag | "
            "max_color_share |"
        ),
        (
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {row['weight_init']} | "
            f"{row['activation']} | "
            f"{float(row['dropout']):.3f} | "
            f"{int(row['total_parameters'])} | "
            f"{float(row['model_size_mb']):.3f} | "
            f"{float(row['inference_time_ms']):.3f} | "
            f"{float(row['images_per_second']):.2f} | "
            f"{format_optional_float(row.get('test_ndcg@3'))} | "
            f"{format_optional_float(row.get('test_ndcg@5'))} | "
            f"{format_optional_float(row.get('top1_wcag_pass_rate'))} | "
            f"{format_optional_float(row.get('top5_any_wcag_pass_rate'))} | "
            f"{format_optional_float(row.get('max_color_share'))} |"
        )

    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![Latency]({markdown_image_path(path, latency_plot_path)})",
        ]
    )
    if trained:
        lines.extend(
            [
                f"![Validation Loss]({markdown_image_path(path, loss_plot_path)})",
                f"![Validation NDCG@5]({markdown_image_path(path, ndcg_plot_path)})",
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_models(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    config_path = resolve_project_path(args.config, must_exist=True)
    config = load_config(config_path)
    model_names = configured_models(config, args.models)
    device_name = args.device or str(nested_value(config, "training", "device", "cuda"))
    device = resolve_device(device_name)
    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else nested_value(config, "latency", "batch_size", 64)
    )
    warmup_steps = int(
        args.warmup_steps
        if args.warmup_steps is not None
        else nested_value(config, "latency", "warmup_steps", 5)
    )
    benchmark_steps = int(
        args.benchmark_steps
        if args.benchmark_steps is not None
        else nested_value(config, "latency", "benchmark_steps", 20)
    )
    dropout = float(
        args.dropout
        if args.dropout is not None
        else nested_value(config, "training", "dropout", 0.2)
    )
    weight_init = str(
        args.weight_init
        if args.weight_init is not None
        else nested_value(config, "training", "weight_init", "pytorch_default")
    )
    activation = str(
        args.activation
        if args.activation is not None
        else nested_value(config, "training", "activation", "silu")
    )

    rows: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    for model_name in model_names:
        LOGGER.info("building model=%s", model_name)
        model = build_title_color_model(
            model_name,
            pretrained=args.pretrained,
            dropout=dropout,
            weight_init=weight_init,
            activation=activation,
        )
        latency = measure_latency(
            model,
            device=device,
            batch_size=batch_size,
            warmup_steps=warmup_steps,
            benchmark_steps=benchmark_steps,
        )
        row: dict[str, Any] = {
            "model_name": model_name,
            "dropout": dropout,
            "weight_init": weight_init,
            "activation": activation,
            "total_parameters": count_total_parameters(model),
            "trainable_parameters": count_trainable_parameters(model),
            "model_size_mb": model_size_mb(model),
            "latency_device": device.type,
            "batch_size": batch_size,
            **latency,
            "trained": False,
            "best_epoch": "",
            "test_loss": "",
            "test_ndcg@3": "",
            "test_ndcg@5": "",
            "top1_wcag_pass_rate": "",
            "top5_any_wcag_pass_rate": "",
            "max_color_share": "",
            "report_path": "",
        }
        if args.train:
            LOGGER.info("training model=%s", model_name)
            train_args = training_args_for_model(
                model_name=model_name,
                config=config,
                epochs_override=args.epochs,
                pretrained=args.pretrained,
                dropout=dropout,
                weight_init=weight_init,
                activation=activation,
            )
            result = run_full_training.run(train_args)
            histories[model_name] = result.history
            row = row_from_training_result(row=row, result=result)
        rows.append(row)
    return rows, histories


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows, histories = compare_models(args)
    results_csv = resolve_project_path(args.results_csv)
    report_path = resolve_project_path(args.report_path)
    latency_plot_path = resolve_project_path(args.latency_plot_path)
    loss_plot_path = resolve_project_path(args.loss_plot_path)
    ndcg_plot_path = resolve_project_path(args.ndcg_plot_path)
    write_results_csv(results_csv, rows)
    write_latency_plot(latency_plot_path, rows)
    write_training_curve_plots(
        loss_plot_path=loss_plot_path,
        ndcg_plot_path=ndcg_plot_path,
        histories=histories,
    )
    write_report(
        report_path,
        rows=rows,
        latency_plot_path=latency_plot_path,
        loss_plot_path=loss_plot_path,
        ndcg_plot_path=ndcg_plot_path,
        trained=bool(histories),
    )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
