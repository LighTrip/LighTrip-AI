from __future__ import annotations

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import run_full_training
from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    markdown_image_path,
)
from experiments.title_color_recommendation.run_model_comparison import (
    measure_latency,
    model_size_mb,
)
from src.models.fixed_palette_classifier import (
    count_total_parameters,
    count_trainable_parameters,
)
from src.models.title_color_model_registry import build_title_color_model
from src.title_color_recommendation.training.trainer import resolve_device


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path("configs/title_color_recommendation/titlenet_ablation.yaml")
DEFAULT_FULL_TRAINING_CONFIG = Path(
    "configs/title_color_recommendation/full_training.yaml"
)
VAL_LOSS_KEY = "val_loss"
VAL_NDCG_AT_3_KEY = "val_ndcg@3"
VAL_NDCG_AT_5_KEY = "val_ndcg@5"
TEST_LOSS_KEY = "test_loss"
TEST_NDCG_AT_3_KEY = "test_ndcg@3"
TEST_NDCG_AT_5_KEY = "test_ndcg@5"
BATCH1_LATENCY_KEY = "batch1_latency_ms"
BATCH64_LATENCY_KEY = "batch64_latency_ms"
BASELINE_TRIAL = "titlenet"
TRIAL_NO_STEM = "titlenet_no_stem"
TRIAL_NO_STAGE1 = "titlenet_no_stage1"
TRIAL_NO_STAGE2 = "titlenet_no_stage2"
TRIAL_NO_STAGE3 = "titlenet_no_stage3"
TRIAL_NO_SE = "titlenet_no_se"
TRIAL_NO_RESIDUAL = "titlenet_no_residual"
TRIAL_NO_FIRST_RESIDUAL = "titlenet_no_first_residual"
TRIAL_NO_MIDDLE_RESIDUAL = "titlenet_no_middle_residual"
TRIAL_NO_LAST_RESIDUAL = "titlenet_no_last_residual"
TRIAL_NO_LAST_EXTRA_RESIDUAL = "titlenet_no_last_extra_residual"
DEFAULT_TRIAL_ACTIVATION = "gelu"
DEFAULT_TRAINING_DEVICE = "cuda"
DEFAULT_TRIAL_GROUP = "ablation"
DEFAULT_TRIAL_WEIGHT_INIT = "small_head"
ACTION_STORE_TRUE = "store_true"
ALIGN_CENTER = "center"
ALIGN_RIGHT = "right"
ALIGN_TOP = "top"
FIELD_TRIAL = "trial"
FIELD_GROUP = "group"
FIELD_MODEL_NAME = "model_name"
FIELD_DESCRIPTION = "description"
FIELD_ACTIVATION = "activation"
FIELD_WEIGHT_INIT = "weight_init"
FIELD_DROPOUT = "dropout"
FIELD_LEARNING_RATE = "learning_rate"
FIELD_WEIGHT_DECAY = "weight_decay"
FIELD_TOTAL_PARAMETERS = "total_parameters"
FIELD_TRAINABLE_PARAMETERS = "trainable_parameters"
FIELD_MODEL_SIZE_MB = "model_size_mb"
FIELD_LATENCY_DEVICE = "latency_device"
FIELD_BATCH1_IMAGES_PER_SECOND = "batch1_images_per_second"
FIELD_BATCH64_IMAGES_PER_SECOND = "batch64_images_per_second"
FIELD_TRAINED = "trained"
FIELD_BEST_EPOCH = "best_epoch"
FIELD_MAX_COLOR_SHARE = "max_color_share"
FIELD_CHECKPOINT_DIR = "checkpoint_dir"
FIELD_LOG_PATH = "log_path"
FIELD_REPORT_PATH = "report_path"
FIELD_EPOCHS = "epochs"
FIELD_DEVICE = "device"
FIELD_NUM_WORKERS = "num_workers"
FIELD_WARMUP_STEPS = "warmup_steps"
FIELD_BENCHMARK_STEPS = "benchmark_steps"
FIELD_TRIALS = "trials"
RESULT_FIELDS = [
    FIELD_TRIAL,
    FIELD_GROUP,
    FIELD_MODEL_NAME,
    FIELD_DESCRIPTION,
    FIELD_ACTIVATION,
    FIELD_WEIGHT_INIT,
    FIELD_DROPOUT,
    FIELD_LEARNING_RATE,
    FIELD_WEIGHT_DECAY,
    FIELD_TOTAL_PARAMETERS,
    FIELD_TRAINABLE_PARAMETERS,
    FIELD_MODEL_SIZE_MB,
    FIELD_LATENCY_DEVICE,
    BATCH1_LATENCY_KEY,
    BATCH64_LATENCY_KEY,
    FIELD_BATCH1_IMAGES_PER_SECOND,
    FIELD_BATCH64_IMAGES_PER_SECOND,
    FIELD_TRAINED,
    FIELD_BEST_EPOCH,
    TEST_LOSS_KEY,
    TEST_NDCG_AT_3_KEY,
    TEST_NDCG_AT_5_KEY,
    FIELD_MAX_COLOR_SHARE,
    FIELD_CHECKPOINT_DIR,
    FIELD_LOG_PATH,
    FIELD_REPORT_PATH,
]


@dataclass(frozen=True)
class AblationTrial:
    name: str
    group: str
    model_name: str
    description: str
    activation: str
    weight_init: str
    dropout: float
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class AblationOutputs:
    checkpoint_root: Path
    log_root: Path
    report_root: Path
    results_csv: Path
    report_path: Path
    ndcg_plot_path: Path
    ndcg_delta_plot_path: Path
    paper_summary_plot_path: Path
    residual_paper_plot_path: Path
    stage_paper_plot_path: Path
    latency_plot_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TitLeNet ablation study.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trials", default="")
    parser.add_argument("--train", action=ACTION_STORE_TRUE)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=None)
    parser.add_argument("--pretrained", action=ACTION_STORE_TRUE)
    parser.add_argument("--merge-existing-results", action=ACTION_STORE_TRUE)
    return parser.parse_args(argv)


def load_ablation_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"ablation config must be a mapping: {path}")
    if not isinstance(payload.get(FIELD_TRIALS), list) or not payload[FIELD_TRIALS]:
        raise ValueError("ablation config must contain a non-empty trials list")
    return dict(payload)


def config_section(config: Mapping[str, Any], section_name: str) -> Mapping[str, Any]:
    section = config.get(section_name) or {}
    if not isinstance(section, Mapping):
        raise ValueError(f"config section must be a mapping: {section_name}")
    return section


def output_paths(config: Mapping[str, Any]) -> AblationOutputs:
    output = config_section(config, "output")

    def resolve(key: str, default: str) -> Path:
        return resolve_inside_project(PROJECT_ROOT, output.get(key, default))

    return AblationOutputs(
        checkpoint_root=resolve(
            "checkpoint_root",
            "outputs/checkpoints/titlenet_ablation",
        ),
        log_root=resolve("log_root", "outputs/logs/titlenet_ablation"),
        report_root=resolve("report_root", "outputs/reports/titlenet_ablation"),
        results_csv=resolve(
            "results_csv",
            "outputs/reports/titlenet_ablation_results.csv",
        ),
        report_path=resolve(
            FIELD_REPORT_PATH,
            "outputs/reports/titlenet_ablation_report.md",
        ),
        ndcg_plot_path=resolve(
            "ndcg_plot_path",
            "outputs/reports/titlenet_ablation_ndcg_curve.png",
        ),
        ndcg_delta_plot_path=resolve(
            "ndcg_delta_plot_path",
            "outputs/reports/titlenet_ablation_ndcg5_delta.png",
        ),
        paper_summary_plot_path=resolve(
            "paper_summary_plot_path",
            "outputs/reports/titlenet_ablation_paper_summary.png",
        ),
        residual_paper_plot_path=resolve(
            "residual_paper_plot_path",
            "outputs/reports/titlenet_residual_ablation_paper_summary.png",
        ),
        stage_paper_plot_path=resolve(
            "stage_paper_plot_path",
            "outputs/reports/titlenet_stage_ablation_paper_summary.png",
        ),
        latency_plot_path=resolve(
            "latency_plot_path",
            "outputs/reports/titlenet_ablation_latency.png",
        ),
    )


def merged_trial_values(
    base: Mapping[str, Any],
    trial: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(base)
    values.update(dict(trial))
    return values


def selected_trials(
    config: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[AblationTrial]:
    names = {name.strip() for name in args.trials.split(",") if name.strip()}
    base = config_section(config, "base")
    raw_trials = config[FIELD_TRIALS]
    limited_trials = raw_trials[: args.max_trials] if args.max_trials else raw_trials
    trials: list[AblationTrial] = []
    for index, raw_trial in enumerate(limited_trials, start=1):
        if not isinstance(raw_trial, Mapping):
            raise ValueError(f"trial must be a mapping: index={index}")
        values = merged_trial_values(base, raw_trial)
        name = str(values["name"])
        if names and name not in names:
            continue
        trials.append(
            AblationTrial(
                name=name,
                group=str(values.get(FIELD_GROUP, DEFAULT_TRIAL_GROUP)),
                model_name=str(values[FIELD_MODEL_NAME]),
                description=str(values.get(FIELD_DESCRIPTION, "")),
                activation=str(values.get(FIELD_ACTIVATION, DEFAULT_TRIAL_ACTIVATION)),
                weight_init=str(values.get(FIELD_WEIGHT_INIT, DEFAULT_TRIAL_WEIGHT_INIT)),
                dropout=float(values.get(FIELD_DROPOUT, 0.2)),
                learning_rate=float(values.get(FIELD_LEARNING_RATE, 5e-4)),
                weight_decay=float(values.get(FIELD_WEIGHT_DECAY, 1e-4)),
            )
        )
    if not trials:
        raise ValueError("no ablation trials selected")
    return trials


def training_value(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    key: str,
    default: Any,
) -> Any:
    training = config_section(config, "training")
    if key == FIELD_EPOCHS and args.epochs is not None:
        return args.epochs
    if key == FIELD_DEVICE and args.device is not None:
        return args.device
    if key == FIELD_NUM_WORKERS and args.num_workers is not None:
        return args.num_workers
    return training.get(key, default)


def latency_value(
    config: Mapping[str, Any],
    args: argparse.Namespace,
    key: str,
    default: Any,
) -> Any:
    latency = config_section(config, "latency")
    if key == FIELD_WARMUP_STEPS and args.warmup_steps is not None:
        return args.warmup_steps
    if key == FIELD_BENCHMARK_STEPS and args.benchmark_steps is not None:
        return args.benchmark_steps
    return latency.get(key, default)


def full_training_args(
    *,
    trial: AblationTrial,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    outputs: AblationOutputs,
) -> argparse.Namespace:
    report_dir = outputs.report_root / trial.name
    cli_args = [
        "--config",
        str(DEFAULT_FULL_TRAINING_CONFIG),
        "--model-name",
        trial.model_name,
        "--epochs",
        str(training_value(config, args, FIELD_EPOCHS, 20)),
        "--learning-rate",
        str(trial.learning_rate),
        "--weight-decay",
        str(trial.weight_decay),
        "--batch-size",
        str(training_value(config, args, "batch_size", 64)),
        "--dropout",
        str(trial.dropout),
        "--weight-init",
        trial.weight_init,
        "--activation",
        trial.activation,
        "--num-workers",
        str(training_value(config, args, FIELD_NUM_WORKERS, 4)),
        "--device",
        str(training_value(config, args, FIELD_DEVICE, DEFAULT_TRAINING_DEVICE)),
        "--seed",
        str(training_value(config, args, "seed", 42)),
        "--best-metric",
        str(training_value(config, args, "best_metric", VAL_NDCG_AT_5_KEY)),
        "--scheduler",
        str(training_value(config, args, "scheduler", "cosine")),
        "--checkpoint-dir",
        str(outputs.checkpoint_root / trial.name),
        "--log-path",
        str(outputs.log_root / f"{trial.name}.jsonl"),
        "--report-path",
        str(report_dir / "full_training_report.md"),
        "--loss-plot-path",
        str(report_dir / "loss_curve.png"),
        "--ndcg-plot-path",
        str(report_dir / "ndcg_curve.png"),
        "--color-plot-path",
        str(report_dir / "color_distribution.png"),
    ]
    if args.pretrained:
        cli_args.append("--pretrained")
    return run_full_training.parse_args(cli_args)


def benchmark_row(
    *,
    trial: AblationTrial,
    config: Mapping[str, Any],
    args: argparse.Namespace,
    outputs: AblationOutputs,
) -> dict[str, Any]:
    device_name = str(training_value(config, args, FIELD_DEVICE, DEFAULT_TRAINING_DEVICE))
    device = resolve_device(device_name)
    warmup_steps = int(latency_value(config, args, FIELD_WARMUP_STEPS, 10))
    benchmark_steps = int(latency_value(config, args, FIELD_BENCHMARK_STEPS, 50))
    model = build_title_color_model(
        trial.model_name,
        pretrained=args.pretrained,
        dropout=trial.dropout,
        weight_init=trial.weight_init,
        activation=trial.activation,
    )
    batch1_latency = measure_latency(
        model,
        device=device,
        batch_size=1,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )
    batch64_latency = measure_latency(
        model,
        device=device,
        batch_size=64,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )
    return {
        FIELD_TRIAL: trial.name,
        FIELD_GROUP: trial.group,
        FIELD_MODEL_NAME: trial.model_name,
        FIELD_DESCRIPTION: trial.description,
        FIELD_ACTIVATION: trial.activation,
        FIELD_WEIGHT_INIT: trial.weight_init,
        FIELD_DROPOUT: trial.dropout,
        FIELD_LEARNING_RATE: trial.learning_rate,
        FIELD_WEIGHT_DECAY: trial.weight_decay,
        FIELD_TOTAL_PARAMETERS: count_total_parameters(model),
        FIELD_TRAINABLE_PARAMETERS: count_trainable_parameters(model),
        FIELD_MODEL_SIZE_MB: model_size_mb(model),
        FIELD_LATENCY_DEVICE: device.type,
        BATCH1_LATENCY_KEY: batch1_latency["inference_time_ms"],
        BATCH64_LATENCY_KEY: batch64_latency["inference_time_ms"],
        FIELD_BATCH1_IMAGES_PER_SECOND: batch1_latency["images_per_second"],
        FIELD_BATCH64_IMAGES_PER_SECOND: batch64_latency["images_per_second"],
        FIELD_TRAINED: False,
        FIELD_BEST_EPOCH: "",
        TEST_LOSS_KEY: "",
        TEST_NDCG_AT_3_KEY: "",
        TEST_NDCG_AT_5_KEY: "",
        FIELD_MAX_COLOR_SHARE: "",
        FIELD_CHECKPOINT_DIR: str(outputs.checkpoint_root / trial.name),
        FIELD_LOG_PATH: str(outputs.log_root / f"{trial.name}.jsonl"),
        FIELD_REPORT_PATH: str(outputs.report_root / trial.name / "full_training_report.md"),
    }


def add_training_metrics(
    row: dict[str, Any],
    result: run_full_training.FullTrainingResult,
) -> dict[str, Any]:
    test_metrics = result.test_metrics.as_dict()
    row.update(
        {
            FIELD_TRAINED: True,
            FIELD_BEST_EPOCH: result.best_epoch,
            TEST_LOSS_KEY: test_metrics[VAL_LOSS_KEY],
            TEST_NDCG_AT_3_KEY: test_metrics[VAL_NDCG_AT_3_KEY],
            TEST_NDCG_AT_5_KEY: test_metrics[VAL_NDCG_AT_5_KEY],
            FIELD_MAX_COLOR_SHARE: max(result.test_metrics.color_distribution),
            FIELD_REPORT_PATH: str(result.report_path),
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


def load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def merge_result_rows(
    existing_rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_by_trial = {
        str(row[FIELD_TRIAL]): row
        for row in existing_rows
    }
    for row in new_rows:
        rows_by_trial[str(row[FIELD_TRIAL])] = row

    merged_rows: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    for row in [*existing_rows, *new_rows]:
        trial = str(row[FIELD_TRIAL])
        if trial in seen_trials:
            continue
        merged_rows.append(rows_by_trial[trial])
        seen_trials.add(trial)
    return merged_rows


def write_latency_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    plt = load_pyplot(PROJECT_ROOT)
    labels = [str(row[FIELD_TRIAL]) for row in rows]
    positions = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        [position - 0.2 for position in positions],
        [float(row[BATCH1_LATENCY_KEY]) for row in rows],
        width=0.4,
        label="batch1",
        color="#2563EB",
    )
    axis.bar(
        [position + 0.2 for position in positions],
        [float(row[BATCH64_LATENCY_KEY]) for row in rows],
        width=0.4,
        label="batch64",
        color="#0F766E",
    )
    axis.set_title("TitLeNet Ablation Latency")
    axis.set_xlabel(FIELD_TRIAL)
    axis.set_ylabel("ms / batch")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha=ALIGN_RIGHT)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_ndcg_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    trained_rows = [row for row in rows if row.get(TEST_NDCG_AT_5_KEY) not in {"", None}]
    if not trained_rows:
        return
    plt = load_pyplot(PROJECT_ROOT)
    labels = [str(row[FIELD_TRIAL]) for row in trained_rows]
    positions = list(range(len(trained_rows)))
    ndcg_values = [
        value
        for row in trained_rows
        for value in (
            float(row[TEST_NDCG_AT_3_KEY]),
            float(row[TEST_NDCG_AT_5_KEY]),
        )
    ]
    ndcg_range = max(ndcg_values) - min(ndcg_values)
    padding = max(ndcg_range * 0.03, 0.0001)
    lower_bound = max(0.0, min(ndcg_values) - padding)
    upper_bound = min(1.0, max(ndcg_values) + padding)
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        [position - 0.2 for position in positions],
        [float(row[TEST_NDCG_AT_3_KEY]) for row in trained_rows],
        width=0.4,
        label=TEST_NDCG_AT_3_KEY,
        color="#7C3AED",
    )
    axis.bar(
        [position + 0.2 for position in positions],
        [float(row[TEST_NDCG_AT_5_KEY]) for row in trained_rows],
        width=0.4,
        label=TEST_NDCG_AT_5_KEY,
        color="#C2410C",
    )
    axis.set_title("TitLeNet Ablation Test NDCG")
    axis.set_xlabel(FIELD_TRIAL)
    axis.set_ylabel("NDCG")
    axis.set_ylim(lower_bound, upper_bound)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=30, ha=ALIGN_RIGHT)
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


PAPER_LABELS = {
    BASELINE_TRIAL: "Full TitLeNet",
    TRIAL_NO_STEM: "Without Stem",
    TRIAL_NO_STAGE1: "Without Stage 1",
    TRIAL_NO_STAGE2: "Without Stage 2",
    TRIAL_NO_STAGE3: "Without Stage 3",
    TRIAL_NO_SE: "Without SE",
    TRIAL_NO_RESIDUAL: "Without all residual",
    TRIAL_NO_FIRST_RESIDUAL: "Without first residual",
    TRIAL_NO_MIDDLE_RESIDUAL: "Without middle residual",
    TRIAL_NO_LAST_RESIDUAL: "Without last residuals",
    TRIAL_NO_LAST_EXTRA_RESIDUAL: "Without last extra residual",
}
RESIDUAL_ABLATION_TRIALS = frozenset(
    {
        TRIAL_NO_SE,
        TRIAL_NO_RESIDUAL,
        TRIAL_NO_FIRST_RESIDUAL,
        TRIAL_NO_MIDDLE_RESIDUAL,
        TRIAL_NO_LAST_RESIDUAL,
        TRIAL_NO_LAST_EXTRA_RESIDUAL,
    }
)
STAGE_ABLATION_TRIALS = frozenset(
    {
        TRIAL_NO_STEM,
        TRIAL_NO_STAGE1,
        TRIAL_NO_STAGE2,
        TRIAL_NO_STAGE3,
    }
)


def write_ndcg_delta_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    trained_rows = [row for row in rows if row.get(TEST_NDCG_AT_5_KEY) not in {"", None}]
    baseline = next(
        (row for row in trained_rows if row[FIELD_TRIAL] == BASELINE_TRIAL),
        None,
    )
    if baseline is None:
        return

    baseline_score = float(baseline[TEST_NDCG_AT_5_KEY])
    ablation_rows = [
        row
        for row in trained_rows
        if row[FIELD_TRIAL] != BASELINE_TRIAL
    ]
    if not ablation_rows:
        return

    ordered_rows = sorted(
        ablation_rows,
        key=lambda row: baseline_score - float(row[TEST_NDCG_AT_5_KEY]),
        reverse=True,
    )
    labels = [
        PAPER_LABELS.get(str(row[FIELD_TRIAL]), str(row[FIELD_TRIAL]))
        for row in ordered_rows
    ]
    drops = [
        baseline_score - float(row[TEST_NDCG_AT_5_KEY])
        for row in ordered_rows
    ]
    positions = list(range(len(ordered_rows)))

    plt = load_pyplot(PROJECT_ROOT)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.barh(positions, drops, color="#334155")
    axis.invert_yaxis()
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.set_xlabel("Test NDCG@5 drop from TitLeNet")
    axis.set_title(f"TitLeNet Ablation Effect (baseline={baseline_score:.6f})")
    axis.grid(axis="x", alpha=0.25)
    axis.axvline(0.0, color="#111827", linewidth=0.8)
    for position, drop in zip(positions, drops):
        axis.text(
            drop + 0.00005,
            position,
            f"{drop:.4f}",
            va=ALIGN_CENTER,
            fontsize=9,
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def trained_rows_for_metric(
    rows: list[Mapping[str, Any]],
    metric_key: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get(metric_key) not in {"", None}
    ]


def baseline_row(
    rows: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    return next((row for row in rows if row[FIELD_TRIAL] == BASELINE_TRIAL), None)


def ordered_ablation_rows(
    rows: list[Mapping[str, Any]],
    *,
    metric_key: str,
    baseline_score: float,
) -> list[Mapping[str, Any]]:
    return sorted(
        (row for row in rows if row[FIELD_TRIAL] != BASELINE_TRIAL),
        key=lambda row: baseline_score - float(row[metric_key]),
        reverse=True,
    )


def paper_label(row: Mapping[str, Any]) -> str:
    return PAPER_LABELS.get(str(row[FIELD_TRIAL]), str(row[FIELD_TRIAL]))


def draw_ndcg_drop_axis(
    axis: Any,
    rows: list[Mapping[str, Any]],
    *,
    metric_key: str,
    title: str,
    xlabel: str,
) -> None:
    baseline = baseline_row(rows)
    if baseline is None:
        return
    baseline_score = float(baseline[metric_key])
    ordered_rows = ordered_ablation_rows(
        rows,
        metric_key=metric_key,
        baseline_score=baseline_score,
    )
    drops = [baseline_score - float(row[metric_key]) for row in ordered_rows]
    positions = list(range(len(ordered_rows)))

    axis.barh(positions, drops, color="#475569")
    axis.invert_yaxis()
    axis.set_yticks(positions)
    axis.set_yticklabels([paper_label(row) for row in ordered_rows])
    axis.set_xlabel(xlabel)
    axis.set_title(title, loc=ALIGN_CENTER)
    axis.grid(axis="x", alpha=0.25)
    axis.set_xlim(0.0, max(drops) * 1.18)
    for position, drop in zip(positions, drops):
        axis.text(drop + 0.00004, position, f"{drop:.4f}", va="center", fontsize=8.5)


def draw_latency_ndcg_axis(
    axis: Any,
    rows: list[Mapping[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
) -> None:
    baseline = baseline_row(rows)
    if baseline is None:
        return
    ablation_rows = [row for row in rows if row[FIELD_TRIAL] != BASELINE_TRIAL]
    metric_values = [float(row[metric_key]) for row in rows]
    latency_values = [float(row[BATCH1_LATENCY_KEY]) for row in rows]
    colors = [
        "#64748B"
        for _row in ablation_rows
    ]
    sizes = [
        max(55.0, float(row[FIELD_TOTAL_PARAMETERS]) / 900.0)
        for row in ablation_rows
    ]

    axis.scatter(
        [float(row[BATCH1_LATENCY_KEY]) for row in ablation_rows],
        [float(row[metric_key]) for row in ablation_rows],
        s=sizes,
        c=colors,
        alpha=0.88,
        edgecolors="white",
        linewidth=0.8,
    )
    axis.scatter(
        [float(baseline[BATCH1_LATENCY_KEY])],
        [float(baseline[metric_key])],
        s=300,
        marker="*",
        c="#DC2626",
        edgecolors="#7F1D1D",
        linewidth=0.8,
        zorder=3,
    )
    axis.set_xlabel("Batch-1 latency (ms)")
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc=ALIGN_CENTER)
    axis.grid(True, alpha=0.25)
    axis.set_ylim(min(metric_values) - 0.00045, max(metric_values) + 0.00035)
    axis.set_xlim(0.0, max(latency_values) + 1.0)
    annotate_latency_axis(axis, rows, metric_key=metric_key)


def annotate_latency_axis(
    axis: Any,
    rows: list[Mapping[str, Any]],
    *,
    metric_key: str,
) -> None:
    highlighted = {
        BASELINE_TRIAL,
        TRIAL_NO_SE,
        TRIAL_NO_RESIDUAL,
        TRIAL_NO_LAST_EXTRA_RESIDUAL,
        TRIAL_NO_STAGE1,
        TRIAL_NO_STAGE3,
    }
    annotation_offsets = {
        BASELINE_TRIAL: (8, -16),
        TRIAL_NO_SE: (8, -10),
        TRIAL_NO_RESIDUAL: (8, 10),
        TRIAL_NO_LAST_EXTRA_RESIDUAL: (8, 12),
        TRIAL_NO_STAGE1: (-12, 14),
        TRIAL_NO_STAGE3: (8, -10),
    }
    right_aligned_trials = {
        TRIAL_NO_STAGE1,
    }
    for row in rows:
        trial = str(row[FIELD_TRIAL])
        if trial not in highlighted:
            continue
        axis.annotate(
            paper_label(row),
            (float(row[BATCH1_LATENCY_KEY]), float(row[metric_key])),
            xytext=annotation_offsets[trial],
            textcoords="offset points",
            fontsize=8.0,
            ha=ALIGN_RIGHT if trial in right_aligned_trials else "left",
            va=ALIGN_TOP if trial == BASELINE_TRIAL else ALIGN_CENTER,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.4,
            },
        )


def write_paper_summary_plot(
    path: Path,
    rows: list[Mapping[str, Any]],
    *,
    figure_title: str = (
        "TitLeNet ablation: component deletion consistently reduces ranking quality"
    ),
) -> None:
    trained_rows = trained_rows_for_metric(rows, TEST_NDCG_AT_5_KEY)
    baseline = baseline_row(trained_rows)
    if baseline is None:
        return

    plt = load_pyplot(PROJECT_ROOT)
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
    draw_ndcg_drop_axis(
        axes[0][0],
        trained_rows,
        metric_key=TEST_NDCG_AT_3_KEY,
        title="(a) Test NDCG@3 drop from full TitLeNet",
        xlabel="Test NDCG@3 drop",
    )
    draw_latency_ndcg_axis(
        axes[0][1],
        trained_rows,
        metric_key=TEST_NDCG_AT_3_KEY,
        title="(b) Latency vs Test NDCG@3",
        ylabel="Test NDCG@3",
    )
    draw_ndcg_drop_axis(
        axes[1][0],
        trained_rows,
        metric_key=TEST_NDCG_AT_5_KEY,
        title="(c) Test NDCG@5 drop from full TitLeNet",
        xlabel="Test NDCG@5 drop",
    )
    draw_latency_ndcg_axis(
        axes[1][1],
        trained_rows,
        metric_key=TEST_NDCG_AT_5_KEY,
        title="(d) Latency vs Test NDCG@5",
        ylabel="Test NDCG@5",
    )

    figure.suptitle(
        figure_title,
        fontsize=13,
        fontweight="bold",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=600)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def filter_rows_for_trials(
    rows: list[Mapping[str, Any]],
    trial_names: frozenset[str],
) -> list[Mapping[str, Any]]:
    selected_trials = {BASELINE_TRIAL, *trial_names}
    return [
        row
        for row in rows
        if str(row[FIELD_TRIAL]) in selected_trials
    ]


def write_residual_paper_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    write_paper_summary_plot(
        path,
        filter_rows_for_trials(rows, RESIDUAL_ABLATION_TRIALS),
        figure_title=(
            "TitLeNet residual/SE ablation: removing design elements "
            "reduces ranking quality"
        ),
    )


def write_stage_paper_plot(path: Path, rows: list[Mapping[str, Any]]) -> None:
    write_paper_summary_plot(
        path,
        filter_rows_for_trials(rows, STAGE_ABLATION_TRIALS),
        figure_title=(
            "TitLeNet stage-level ablation: each stage contributes to "
            "ranking quality"
        ),
    )


def format_float(value: Any, *, digits: int = 6) -> str:
    if value in {"", None}:
        return "-"
    return f"{float(value):.{digits}f}"


def write_report(
    path: Path,
    *,
    rows: list[Mapping[str, Any]],
    results_csv: Path,
    ndcg_plot_path: Path,
    ndcg_delta_plot_path: Path,
    paper_summary_plot_path: Path,
    residual_paper_plot_path: Path,
    stage_paper_plot_path: Path,
    latency_plot_path: Path,
    trained: bool,
) -> None:
    lines = [
        "# TitLeNet Ablation Study",
        "",
        f"- trained: `{trained}`",
        f"- results_csv: `{results_csv}`",
        "",
        "## Summary",
        "",
        (
            "| trial | group | model | act | init | params | size_mb | "
            "b1_ms | b64_ms | test_loss | test_ndcg@3 | test_ndcg@5 | "
            "max_color_share |"
        ),
        (
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: |"
        ),
    ]
    for row in rows:
        lines.append(
            f"| {row[FIELD_TRIAL]} | {row[FIELD_GROUP]} | "
            f"{row[FIELD_MODEL_NAME]} | "
            f"{row[FIELD_ACTIVATION]} | {row[FIELD_WEIGHT_INIT]} | "
            f"{int(row[FIELD_TOTAL_PARAMETERS])} | "
            f"{float(row[FIELD_MODEL_SIZE_MB]):.3f} | "
            f"{float(row[BATCH1_LATENCY_KEY]):.3f} | "
            f"{float(row[BATCH64_LATENCY_KEY]):.3f} | "
            f"{format_float(row.get(TEST_LOSS_KEY))} | "
            f"{format_float(row.get(TEST_NDCG_AT_3_KEY))} | "
            f"{format_float(row.get(TEST_NDCG_AT_5_KEY))} | "
            f"{format_float(row.get(FIELD_MAX_COLOR_SHARE))} |"
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
        lines.append(f"![NDCG]({markdown_image_path(path, ndcg_plot_path)})")
        lines.append(
            "![NDCG Delta]("
            f"{markdown_image_path(path, ndcg_delta_plot_path)})"
        )
        lines.append(
            "![Paper Summary]("
            f"{markdown_image_path(path, paper_summary_plot_path)})"
        )
        lines.append(
            "![Residual Ablation]("
            f"{markdown_image_path(path, residual_paper_plot_path)})"
        )
        lines.append(
            "![Stage Ablation]("
            f"{markdown_image_path(path, stage_paper_plot_path)})"
        )
    else:
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Run the same command with `--train` to fill test metrics.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config_path = resolve_inside_project(PROJECT_ROOT, args.config, must_exist=True)
    config = load_ablation_config(config_path)
    outputs = output_paths(config)
    new_rows: list[dict[str, Any]] = []
    for trial in selected_trials(config, args):
        LOGGER.info("running ablation trial=%s", trial.name)
        row = benchmark_row(trial=trial, config=config, args=args, outputs=outputs)
        if args.train:
            train_args = full_training_args(
                trial=trial,
                config=config,
                args=args,
                outputs=outputs,
            )
            row = add_training_metrics(row, run_full_training.run(train_args))
        new_rows.append(row)

    rows = new_rows
    if args.merge_existing_results:
        rows = merge_result_rows(
            load_existing_results(outputs.results_csv),
            new_rows,
        )
    write_results_csv(outputs.results_csv, rows)
    write_latency_plot(outputs.latency_plot_path, rows)
    write_ndcg_plot(outputs.ndcg_plot_path, rows)
    write_ndcg_delta_plot(outputs.ndcg_delta_plot_path, rows)
    write_paper_summary_plot(outputs.paper_summary_plot_path, rows)
    write_residual_paper_plot(outputs.residual_paper_plot_path, rows)
    write_stage_paper_plot(outputs.stage_paper_plot_path, rows)
    write_report(
        outputs.report_path,
        rows=rows,
        results_csv=outputs.results_csv,
        ndcg_plot_path=outputs.ndcg_plot_path,
        ndcg_delta_plot_path=outputs.ndcg_delta_plot_path,
        paper_summary_plot_path=outputs.paper_summary_plot_path,
        residual_paper_plot_path=outputs.residual_paper_plot_path,
        stage_paper_plot_path=outputs.stage_paper_plot_path,
        latency_plot_path=outputs.latency_plot_path,
        trained=args.train,
    )
    return rows


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
