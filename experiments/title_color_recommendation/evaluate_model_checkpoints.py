from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.run_model_comparison import model_size_mb
from src.models.fixed_palette_classifier import (
    count_total_parameters,
    count_trainable_parameters,
)
from src.models.title_color_model_registry import (
    build_title_color_model,
    normalize_model_name,
)
from src.title_color_recommendation.data.dataloader import (
    create_title_color_dataloaders,
)
from src.title_color_recommendation.training.config import (
    TrainingConfig,
    training_config_from_mapping,
)
from src.title_color_recommendation.training.metrics import ValidationMetrics
from src.title_color_recommendation.training.trainer import resolve_device, validate


DEFAULT_MODELS = (
    "titlenet",
    "resnet18",
    "resnet34",
    "vit_tiny",
    "convnext_tiny",
    "efficientnet_b0",
    "flatten_mlp",
    "swin_tiny",
)
DEFAULT_CHECKPOINTS = {
    "titlenet": Path("outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt"),
    "resnet18": Path("outputs/checkpoints/resnet18_ndcg3_eval/checkpoint_best.pt"),
    "resnet34": Path("outputs/checkpoints/resnet34_ndcg3_eval/checkpoint_best.pt"),
    "vit_tiny": Path("outputs/checkpoints/vit_tiny_ndcg3_eval/checkpoint_best.pt"),
    "convnext_tiny": Path(
        "outputs/checkpoints/convnext_tiny_ndcg3_eval/checkpoint_best.pt"
    ),
    "efficientnet_b0": Path(
        "outputs/checkpoints/efficientnet_b0_ndcg3_eval/checkpoint_best.pt"
    ),
    "flatten_mlp": Path(
        "outputs/checkpoints/flatten_mlp_ndcg3_eval/checkpoint_best.pt"
    ),
    "swin_tiny": Path("outputs/checkpoints/swin_tiny_ndcg3_eval/checkpoint_best.pt"),
}
DEFAULT_RESULTS_CSV = Path(
    "outputs/reports/model_evaluation/checkpoint_eval_results.csv"
)
DEFAULT_REPORT_PATH = Path("outputs/reports/ndcg_model_eval_comparison.md")
DEFAULT_LATENCY_CSV = Path(
    "outputs/reports/model_evaluation/latency/existing_models_latency_ndcg3_eval.csv"
)
RESULT_FIELDS = [
    "rank",
    "model_name",
    "checkpoint_path",
    "test_loss",
    "test_ndcg@3",
    "test_ndcg@5",
    "top1_wcag_pass_rate",
    "top5_any_wcag_pass_rate",
    "max_color_share",
    "total_parameters",
    "trainable_parameters",
    "model_size_mb",
    "latency_b1_ms",
    "latency_b64_ms",
    "ms_per_image_b64",
]


@dataclass(frozen=True)
class EvaluatedModel:
    model_name: str
    checkpoint_path: Path
    config: TrainingConfig
    metrics: ValidationMetrics
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    latency_b1_ms: float | None
    latency_b64_ms: float | None
    ms_per_image_b64: float | None

    @property
    def max_color_share(self) -> float:
        return max(self.metrics.color_distribution)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved title color recommendation checkpoints."
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/title_color_recommendation"),
    )
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--latency-csv", type=Path, default=DEFAULT_LATENCY_CSV)
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


def parse_model_names(raw_models: str) -> list[str]:
    models = [
        normalize_model_name(name)
        for name in raw_models.split(",")
        if name.strip()
    ]
    if not models:
        raise ValueError("at least one model is required")
    return models


def checkpoint_path_for_model(model_name: str) -> Path:
    try:
        relative_path = DEFAULT_CHECKPOINTS[model_name]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_CHECKPOINTS))
        raise ValueError(
            f"no default checkpoint path for model={model_name!r}; known={known}"
        ) from exc
    return resolve_project_path(relative_path, must_exist=True, description="checkpoint")


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    # Checkpoint paths are fixed internally and constrained to PROJECT_ROOT.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"checkpoint must be a mapping: {path}")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"checkpoint missing model_state_dict: {path}")
    return checkpoint


def training_config_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    fallback_model_name: str,
    batch_size: int,
    device: str,
    num_workers: int,
) -> TrainingConfig:
    raw_config = checkpoint.get("config") or {}
    if not isinstance(raw_config, Mapping):
        raw_config = {}
    values = {
        **dict(raw_config),
        "batch_size": batch_size,
        "device": device,
        "num_workers": num_workers,
        "model_name": normalize_model_name(
            str(raw_config.get("model_name") or fallback_model_name)
        ),
    }
    return training_config_from_mapping(values)


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    config: TrainingConfig,
) -> nn.Module:
    model = build_title_color_model(
        config.model_name,
        num_classes=config.num_classes,
        pretrained=False,
        dropout=config.dropout,
        weight_init=config.weight_init,
        activation=config.activation,
    )
    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint model_state_dict must be a mapping")
    model.load_state_dict(state_dict)
    return model


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.labels_matrix is not None:
        kwargs["labels_matrix_path"] = args.labels_matrix
    if args.labels_soft is not None:
        kwargs["labels_soft_path"] = args.labels_soft
    return kwargs


def load_latency_rows(path: Path) -> dict[tuple[str, int], Mapping[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {
        (normalize_model_name(row["model"]), int(row["batch_size"])): row
        for row in rows
    }


def optional_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def latency_value(
    latency_rows: Mapping[tuple[str, int], Mapping[str, str]],
    *,
    model_name: str,
    batch_size: int,
    field: str,
) -> float | None:
    row = latency_rows.get((model_name, batch_size))
    if row is None:
        return None
    return optional_float(row.get(field))


def evaluate_model(
    *,
    model_name: str,
    checkpoint_path: Path,
    test_loader: Any,
    args: argparse.Namespace,
    latency_rows: Mapping[tuple[str, int], Mapping[str, str]],
) -> EvaluatedModel:
    checkpoint = load_checkpoint(checkpoint_path)
    config = training_config_from_checkpoint(
        checkpoint,
        fallback_model_name=model_name,
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
    )
    model = build_model_from_checkpoint(checkpoint, config)
    device = resolve_device(config.device)
    model.to(device)
    metrics = validate(
        model,
        test_loader,
        device=device,
        num_classes=config.num_classes,
    )
    return EvaluatedModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        config=config,
        metrics=metrics,
        total_parameters=count_total_parameters(model),
        trainable_parameters=count_trainable_parameters(model),
        model_size_mb=model_size_mb(model),
        latency_b1_ms=latency_value(
            latency_rows,
            model_name=model_name,
            batch_size=1,
            field="median_latency_ms",
        ),
        latency_b64_ms=latency_value(
            latency_rows,
            model_name=model_name,
            batch_size=64,
            field="median_latency_ms",
        ),
        ms_per_image_b64=latency_value(
            latency_rows,
            model_name=model_name,
            batch_size=64,
            field="ms_per_image_median",
        ),
    )


def ranked_results(results: list[EvaluatedModel]) -> list[EvaluatedModel]:
    return sorted(results, key=lambda result: result.metrics.val_ndcg_at_3, reverse=True)


def format_float(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_results_csv(path: Path, results: list[EvaluatedModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for rank, result in enumerate(ranked_results(results), start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "model_name": result.model_name,
                    "checkpoint_path": result.checkpoint_path.as_posix(),
                    "test_loss": result.metrics.val_loss,
                    "test_ndcg@3": result.metrics.val_ndcg_at_3,
                    "test_ndcg@5": result.metrics.val_ndcg_at_5,
                    "top1_wcag_pass_rate": result.metrics.top1_wcag_pass_rate,
                    "top5_any_wcag_pass_rate": (
                        result.metrics.top5_any_wcag_pass_rate
                    ),
                    "max_color_share": result.max_color_share,
                    "total_parameters": result.total_parameters,
                    "trainable_parameters": result.trainable_parameters,
                    "model_size_mb": result.model_size_mb,
                    "latency_b1_ms": format_float(result.latency_b1_ms),
                    "latency_b64_ms": format_float(result.latency_b64_ms),
                    "ms_per_image_b64": format_float(result.ms_per_image_b64),
                }
            )


def compact_params(parameter_count: int) -> str:
    return f"{parameter_count / 1_000_000:.3f}M"


def table_float(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_report(
    path: Path,
    *,
    results: list[EvaluatedModel],
    results_csv: Path,
    latency_csv: Path,
) -> None:
    ordered = ranked_results(results)
    by_batch1 = sorted(results, key=lambda result: result.latency_b1_ms or float("inf"))
    by_batch64 = sorted(
        results,
        key=lambda result: result.latency_b64_ms or float("inf"),
    )
    lines = [
        "# NDCG@3 Model Evaluation and Latency Comparison",
        "",
        "## Runs",
        "",
        "All listed models were evaluated from saved best checkpoints.",
        "",
        "```text",
        *[result.model_name for result in ordered],
        "```",
        "",
        "## Test Metrics With Latency",
        "",
        (
            "| rank | model | test_loss | test_ndcg@3 | test_ndcg@5 | "
            "max_color_share | params | latency_b1_ms | latency_b64_ms |"
        ),
        (
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for rank, result in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | `{result.model_name}` | "
            f"{result.metrics.val_loss:.6f} | "
            f"{result.metrics.val_ndcg_at_3:.6f} | "
            f"{result.metrics.val_ndcg_at_5:.6f} | "
            f"{result.max_color_share:.6f} | "
            f"{compact_params(result.total_parameters)} | "
            f"{table_float(result.latency_b1_ms)} | "
            f"{table_float(result.latency_b64_ms)} |"
        )

    lines.extend(
        [
            "",
            "## Latency Ranking",
            "",
            "### Batch 1",
            "",
            "| rank | model | latency_b1_ms | test_ndcg@3 |",
            "| ---: | --- | ---: | ---: |",
        ]
    )
    for rank, result in enumerate(by_batch1, start=1):
        lines.append(
            f"| {rank} | `{result.model_name}` | "
            f"{table_float(result.latency_b1_ms)} | "
            f"{result.metrics.val_ndcg_at_3:.6f} |"
        )

    lines.extend(
        [
            "",
            "### Batch 64",
            "",
            "| rank | model | latency_b64_ms | ms_per_image | test_ndcg@3 |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for rank, result in enumerate(by_batch64, start=1):
        lines.append(
            f"| {rank} | `{result.model_name}` | "
            f"{table_float(result.latency_b64_ms)} | "
            f"{table_float(result.ms_per_image_b64, digits=4)} | "
            f"{result.metrics.val_ndcg_at_3:.6f} |"
        )

    best_quality = ordered[0]
    best_batch1 = by_batch1[0]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- `{best_quality.model_name}` is the best model by "
                "`test_ndcg@3` in this checkpoint evaluation."
            ),
            (
                f"- `{best_batch1.model_name}` is the fastest model at batch 1 "
                "among the evaluated checkpoints."
            ),
            "- `max_color_share` is still useful for detecting collapse toward one "
            "dominant color.",
            "",
            "## Recommendation",
            "",
            "```text",
            "Use the top-ranked quality model when recommendation quality is primary.",
            "Use the batch-1 latency winner only if single-image response time is primary.",
            "Reject models with low NDCG or high max_color_share for final service use.",
            "```",
            "",
            "## Artifacts",
            "",
            "| artifact | path |",
            "| --- | --- |",
            f"| checkpoint evaluation CSV | `{display_path(results_csv)}` |",
            f"| latency benchmark CSV | `{display_path(latency_csv)}` |",
        ]
    )
    for result in ordered:
        lines.append(
            f"| `{result.model_name}` checkpoint | "
            f"`{display_path(result.checkpoint_path)}` |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[EvaluatedModel]:
    device = resolve_device(args.device)
    loaders = create_title_color_dataloaders(
        batch_size=args.batch_size,
        splits=("test",),
        data_root=args.data_root,
        project_root=PROJECT_ROOT,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        dataset_kwargs=dataset_kwargs(args),
    )
    test_loader = loaders["test"]
    latency_csv = resolve_project_path(args.latency_csv)
    latency_rows = load_latency_rows(latency_csv)
    results: list[EvaluatedModel] = []
    for model_name in parse_model_names(args.models):
        checkpoint_path = checkpoint_path_for_model(model_name)
        print(f"evaluating {model_name} from {checkpoint_path}")
        results.append(
            evaluate_model(
                model_name=model_name,
                checkpoint_path=checkpoint_path,
                test_loader=test_loader,
                args=args,
                latency_rows=latency_rows,
            )
        )

    results_csv = resolve_project_path(args.results_csv, description="results CSV")
    report_path = resolve_project_path(args.report_path, description="report path")
    write_results_csv(results_csv, results)
    write_report(
        report_path,
        results=results,
        results_csv=results_csv,
        latency_csv=latency_csv,
    )
    print(f"wrote {results_csv}")
    print(f"wrote {report_path}")
    return results


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
