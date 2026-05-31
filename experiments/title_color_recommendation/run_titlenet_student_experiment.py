from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import run_full_training
from experiments.title_color_recommendation import (
    train_titlenet_student_distillation as student_distillation,
)
from experiments.title_color_recommendation.path_utils import (
    load_yaml_mapping,
    require_mapping,
    resolve_project_path as resolve_inside_project,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = student_distillation.DEFAULT_CONFIG
DEFAULT_STUDENT_ONLY_CHECKPOINT_DIR = Path(
    "outputs/checkpoints/titlenet_student_only"
)
DEFAULT_STUDENT_ONLY_LOG_PATH = Path("outputs/logs/titlenet_student_only.jsonl")
DEFAULT_STUDENT_ONLY_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_only_report.md"
)
DEFAULT_STUDENT_ONLY_LOSS_PLOT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_only_loss.png"
)
DEFAULT_STUDENT_ONLY_NDCG_PLOT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_only_ndcg.png"
)
DEFAULT_STUDENT_ONLY_COLOR_PLOT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_only_colors.png"
)
DEFAULT_EXPERIMENT_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_experiment_report.md"
)
DEFAULT_EXPERIMENT_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_experiment_metrics.json"
)


@dataclass(frozen=True)
class StudentOnlyOutputPaths:
    checkpoint_dir: Path
    log_path: Path
    report_path: Path
    loss_plot_path: Path
    ndcg_plot_path: Path
    color_plot_path: Path


@dataclass(frozen=True)
class ExperimentOutputPaths:
    report_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class TitLeNetStudentExperimentResult:
    student_only: run_full_training.FullTrainingResult
    distillation: student_distillation.StudentDistillationResult
    report_path: Path
    metrics_path: Path
    metrics_payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Student-only and Teacher-Student distillation training in one "
            "TitLeNet Student experiment."
        )
    )
    student_distillation.add_distillation_training_args(parser)
    parser.add_argument("--distillation-epochs", type=int, default=None)
    parser.add_argument("--distillation-learning-rate", type=float, default=None)
    parser.add_argument("--student-only-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--student-only-log-path", type=Path, default=None)
    parser.add_argument("--student-only-report-path", type=Path, default=None)
    parser.add_argument("--student-only-loss-plot-path", type=Path, default=None)
    parser.add_argument("--student-only-ndcg-plot-path", type=Path, default=None)
    parser.add_argument("--student-only-color-plot-path", type=Path, default=None)
    parser.add_argument("--distillation-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--distillation-log-path", type=Path, default=None)
    parser.add_argument("--distillation-report-path", type=Path, default=None)
    parser.add_argument("--distillation-metrics-path", type=Path, default=None)
    parser.add_argument("--experiment-report-path", type=Path, default=None)
    parser.add_argument("--experiment-metrics-path", type=Path, default=None)
    return parser.parse_args(argv)


def _load_raw_config(path: Path) -> Mapping[str, Any]:
    return load_yaml_mapping(
        PROJECT_ROOT,
        path,
        description="student experiment config",
    )


def _output_path(
    section: Mapping[str, Any],
    key: str,
    default: Path,
    *,
    override: Path | None,
) -> Path:
    value = override if override is not None else section.get(key, default)
    return resolve_inside_project(PROJECT_ROOT, Path(value), description=key)


def load_student_only_output_paths(args: argparse.Namespace) -> StudentOnlyOutputPaths:
    raw_config = _load_raw_config(args.config)
    section = require_mapping(raw_config.get("student_only"), description="student_only")
    return StudentOnlyOutputPaths(
        checkpoint_dir=_output_path(
            section,
            "checkpoint_dir",
            DEFAULT_STUDENT_ONLY_CHECKPOINT_DIR,
            override=args.student_only_checkpoint_dir,
        ),
        log_path=_output_path(
            section,
            "log_path",
            DEFAULT_STUDENT_ONLY_LOG_PATH,
            override=args.student_only_log_path,
        ),
        report_path=_output_path(
            section,
            "report_path",
            DEFAULT_STUDENT_ONLY_REPORT_PATH,
            override=args.student_only_report_path,
        ),
        loss_plot_path=_output_path(
            section,
            "loss_plot_path",
            DEFAULT_STUDENT_ONLY_LOSS_PLOT_PATH,
            override=args.student_only_loss_plot_path,
        ),
        ndcg_plot_path=_output_path(
            section,
            "ndcg_plot_path",
            DEFAULT_STUDENT_ONLY_NDCG_PLOT_PATH,
            override=args.student_only_ndcg_plot_path,
        ),
        color_plot_path=_output_path(
            section,
            "color_plot_path",
            DEFAULT_STUDENT_ONLY_COLOR_PLOT_PATH,
            override=args.student_only_color_plot_path,
        ),
    )


def load_experiment_output_paths(args: argparse.Namespace) -> ExperimentOutputPaths:
    raw_config = _load_raw_config(args.config)
    section = require_mapping(
        raw_config.get("experiment_outputs"),
        description="experiment_outputs",
    )
    return ExperimentOutputPaths(
        report_path=_output_path(
            section,
            "report_path",
            DEFAULT_EXPERIMENT_REPORT_PATH,
            override=args.experiment_report_path,
        ),
        metrics_path=_output_path(
            section,
            "metrics_path",
            DEFAULT_EXPERIMENT_METRICS_PATH,
            override=args.experiment_metrics_path,
        ),
    )


def _append_optional_arg(
    cli_args: list[str],
    flag: str,
    value: Any,
) -> None:
    if value is not None:
        cli_args.extend([flag, str(value)])


def load_distillation_finetune_overrides(
    args: argparse.Namespace,
) -> dict[str, Any]:
    raw_config = _load_raw_config(args.config)
    section = require_mapping(
        raw_config.get("distillation_finetune"),
        description="distillation_finetune",
    )
    overrides: dict[str, Any] = {}
    learning_rate = (
        args.distillation_learning_rate
        if args.distillation_learning_rate is not None
        else section.get("learning_rate")
    )
    epochs = (
        args.distillation_epochs
        if args.distillation_epochs is not None
        else section.get("epochs")
    )
    if learning_rate is not None:
        overrides["learning_rate"] = float(learning_rate)
    if epochs is not None:
        overrides["epochs"] = int(epochs)
    return overrides


def distillation_args_from_experiment_args(
    args: argparse.Namespace,
    *,
    student_init_checkpoint: Path | None = None,
    finetune_overrides: Mapping[str, Any] | None = None,
) -> argparse.Namespace:
    cli_args = ["--config", str(args.config)]
    _append_optional_arg(cli_args, "--data-root", args.data_root)
    _append_optional_arg(cli_args, "--labels-matrix", args.labels_matrix)
    _append_optional_arg(cli_args, "--labels-soft", args.labels_soft)
    _append_optional_arg(cli_args, "--teacher-checkpoint", args.teacher_checkpoint)
    _append_optional_arg(
        cli_args,
        "--student-init-checkpoint",
        student_init_checkpoint or args.student_init_checkpoint,
    )
    overrides = dict(finetune_overrides or {})
    _append_optional_arg(cli_args, "--epochs", overrides.get("epochs", args.epochs))
    _append_optional_arg(cli_args, "--batch-size", args.batch_size)
    _append_optional_arg(
        cli_args,
        "--learning-rate",
        overrides.get("learning_rate", args.learning_rate),
    )
    _append_optional_arg(cli_args, "--weight-decay", args.weight_decay)
    _append_optional_arg(cli_args, "--num-workers", args.num_workers)
    _append_optional_arg(cli_args, "--device", args.device)
    _append_optional_arg(cli_args, "--scheduler", args.scheduler)
    _append_optional_arg(cli_args, "--best-metric", args.best_metric)
    _append_optional_arg(cli_args, "--seed", args.seed)
    _append_optional_arg(cli_args, "--temperature", args.temperature)
    _append_optional_arg(cli_args, "--base-loss-weight", args.base_loss_weight)
    _append_optional_arg(
        cli_args,
        "--distillation-loss-weight",
        args.distillation_loss_weight,
    )
    _append_optional_arg(
        cli_args,
        "--checkpoint-dir",
        args.distillation_checkpoint_dir,
    )
    _append_optional_arg(cli_args, "--log-path", args.distillation_log_path)
    _append_optional_arg(cli_args, "--report-path", args.distillation_report_path)
    _append_optional_arg(cli_args, "--metrics-path", args.distillation_metrics_path)
    return student_distillation.parse_args(cli_args)


def full_training_args_for_student_only(
    *,
    config: student_distillation.StudentDistillationConfig,
    output_paths: StudentOnlyOutputPaths,
) -> argparse.Namespace:
    cli_args = [
        "--data-root",
        str(config.data_root),
        "--model-name",
        config.student.model_name,
        "--epochs",
        str(config.training.epochs),
        "--learning-rate",
        str(config.training.learning_rate),
        "--weight-decay",
        str(config.training.weight_decay),
        "--batch-size",
        str(config.training.batch_size),
        "--dropout",
        str(config.student.dropout),
        "--weight-init",
        config.student.weight_init,
        "--activation",
        config.student.activation,
        "--num-workers",
        str(config.training.num_workers),
        "--device",
        config.training.device,
        "--seed",
        str(config.training.seed),
        "--best-metric",
        config.training.best_metric,
        "--scheduler",
        config.training.scheduler,
        "--checkpoint-dir",
        str(output_paths.checkpoint_dir),
        "--log-path",
        str(output_paths.log_path),
        "--report-path",
        str(output_paths.report_path),
        "--loss-plot-path",
        str(output_paths.loss_plot_path),
        "--ndcg-plot-path",
        str(output_paths.ndcg_plot_path),
        "--color-plot-path",
        str(output_paths.color_plot_path),
    ]
    if config.labels_matrix is not None:
        cli_args.extend(["--labels-matrix", str(config.labels_matrix)])
    if config.labels_soft is not None:
        cli_args.extend(["--labels-soft", str(config.labels_soft)])
    return run_full_training.parse_args(cli_args)


def run_student_only(
    *,
    config: student_distillation.StudentDistillationConfig,
    output_paths: StudentOnlyOutputPaths,
) -> run_full_training.FullTrainingResult:
    LOGGER.info("running Student-only training: model=%s", config.student.model_name)
    return run_full_training.run(
        full_training_args_for_student_only(
            config=config,
            output_paths=output_paths,
        )
    )


def build_comparison_payload(
    *,
    config: student_distillation.StudentDistillationConfig,
    student_only: run_full_training.FullTrainingResult,
    distillation: student_distillation.StudentDistillationResult,
) -> dict[str, Any]:
    student_only_metrics = student_only.test_metrics.as_dict()
    distilled_metrics = distillation.test_metrics.as_dict()
    ndcg5_delta = (
        float(distilled_metrics[student_distillation.VAL_NDCG_AT_5_KEY])
        - float(student_only_metrics[student_distillation.VAL_NDCG_AT_5_KEY])
    )
    ndcg3_delta = (
        float(distilled_metrics[student_distillation.VAL_NDCG_AT_3_KEY])
        - float(student_only_metrics[student_distillation.VAL_NDCG_AT_3_KEY])
    )
    loss_delta = (
        float(distilled_metrics[student_distillation.VAL_LOSS_KEY])
        - float(student_only_metrics[student_distillation.VAL_LOSS_KEY])
    )
    outcome_checks = {
        "ndcg@3_improved_or_equal": ndcg3_delta >= 0.0,
        "ndcg@5_improved_or_equal": ndcg5_delta >= 0.0,
        "loss_improved_or_equal": loss_delta <= 0.0,
    }
    outcome_checks["ndcg@3_and_ndcg@5_improved_or_equal"] = (
        outcome_checks["ndcg@3_improved_or_equal"]
        and outcome_checks["ndcg@5_improved_or_equal"]
    )
    return {
        "teacher": distillation.metrics_payload["teacher"],
        "student": distillation.metrics_payload["student"],
        "student_init_checkpoint": distillation.metrics_payload[
            "student_init_checkpoint"
        ],
        "training": config.training.as_dict(),
        "distillation_training": distillation.metrics_payload["training"],
        "distillation": distillation.metrics_payload["distillation"],
        "dataset_sizes": distillation.dataset_sizes,
        "student_only": {
            "best_epoch": student_only.best_epoch,
            "best_metric_value": student_only.best_metric_value,
            "test_metrics": student_only_metrics,
            "report_path": str(student_only.report_path),
            "checkpoint_paths": {
                name: str(path)
                for name, path in student_only.checkpoint_paths.items()
            },
        },
        "student_distilled": {
            "best_epoch": distillation.best_epoch,
            "best_metric_value": distillation.best_metric_value,
            "test_metrics": distilled_metrics,
            "teacher_agreement": distillation.test_agreement,
            "report_path": str(distillation.report_path),
            "checkpoint_paths": {
                name: str(path)
                for name, path in distillation.checkpoint_paths.items()
            },
        },
        "profiles": {
            "teacher": distillation.metrics_payload["teacher_profile"],
            "student": distillation.metrics_payload["student_profile"],
        },
        "comparison": {
            "ndcg@3_delta_distilled_vs_student_only": ndcg3_delta,
            "ndcg@5_delta_distilled_vs_student_only": ndcg5_delta,
            "loss_delta_distilled_vs_student_only": loss_delta,
        },
        "outcome_checks": outcome_checks,
    }


def write_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_experiment_report(path: Path, payload: Mapping[str, Any]) -> None:
    student_only = payload["student_only"]
    distilled = payload["student_distilled"]
    profiles = payload["profiles"]
    comparison = payload["comparison"]
    outcome_checks = payload["outcome_checks"]
    student_only_metrics = student_only["test_metrics"]
    distilled_metrics = distilled["test_metrics"]
    teacher_agreement = distilled["teacher_agreement"]
    student_init_checkpoint = payload["student_init_checkpoint"]
    distillation_training = payload["distillation_training"]
    distillation_config = payload["distillation"]

    lines = [
        "# TitLeNet Student Experiment Report",
        "",
        "## Distillation Setup",
        "",
        f"- student_init_checkpoint: `{student_init_checkpoint}`",
        f"- distillation_epochs: `{distillation_training['epochs']}`",
        f"- distillation_learning_rate: `{distillation_training['learning_rate']}`",
        f"- temperature: `{distillation_config['temperature']}`",
        f"- base_loss_weight: `{distillation_config['base_loss_weight']}`",
        (
            f"- distillation_loss_weight: "
            f"`{distillation_config['distillation_loss_weight']}`"
        ),
        "",
        "## Runs",
        "",
        "| run | checkpoint | report |",
        "| --- | --- | --- |",
        (
            f"| Student-only | `{student_only['checkpoint_paths']['best']}` | "
            f"`{student_only['report_path']}` |"
        ),
        (
            f"| Student-distilled | `{distilled['checkpoint_paths']['best']}` | "
            f"`{distilled['report_path']}` |"
        ),
        "",
        "## Test Metrics",
        "",
        "| metric | Student-only | Student-distilled | delta |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| val_loss | "
            f"{float(student_only_metrics[student_distillation.VAL_LOSS_KEY]):.6f} | "
            f"{float(distilled_metrics[student_distillation.VAL_LOSS_KEY]):.6f} | "
            f"{float(comparison['loss_delta_distilled_vs_student_only']):.6f} |"
        ),
        (
            f"| val_ndcg@3 | "
            f"{float(student_only_metrics[student_distillation.VAL_NDCG_AT_3_KEY]):.6f} | "
            f"{float(distilled_metrics[student_distillation.VAL_NDCG_AT_3_KEY]):.6f} | "
            f"{float(comparison['ndcg@3_delta_distilled_vs_student_only']):.6f} |"
        ),
        (
            f"| val_ndcg@5 | "
            f"{float(student_only_metrics[student_distillation.VAL_NDCG_AT_5_KEY]):.6f} | "
            f"{float(distilled_metrics[student_distillation.VAL_NDCG_AT_5_KEY]):.6f} | "
            f"{float(comparison['ndcg@5_delta_distilled_vs_student_only']):.6f} |"
        ),
        "",
        "## Outcome Checks",
        "",
        "| check | result |",
        "| --- | --- |",
        (
            f"| NDCG@3 improved or equal | "
            f"`{'PASS' if outcome_checks['ndcg@3_improved_or_equal'] else 'REVIEW'}` |"
        ),
        (
            f"| NDCG@5 improved or equal | "
            f"`{'PASS' if outcome_checks['ndcg@5_improved_or_equal'] else 'REVIEW'}` |"
        ),
        (
            f"| loss improved or equal | "
            f"`{'PASS' if outcome_checks['loss_improved_or_equal'] else 'REVIEW'}` |"
        ),
        (
            f"| NDCG@3 and NDCG@5 improved or equal | "
            f"`{'PASS' if outcome_checks['ndcg@3_and_ndcg@5_improved_or_equal'] else 'REVIEW'}` |"
        ),
        "",
        "## Teacher Agreement",
        "",
        "| metric | Student-distilled |",
        "| --- | ---: |",
        (
            f"| teacher_top1_agreement | "
            f"{float(teacher_agreement[student_distillation.TEACHER_TOP1_AGREEMENT_KEY]):.6f} |"
        ),
        (
            f"| teacher_top3_overlap | "
            f"{float(teacher_agreement[student_distillation.TEACHER_TOP3_OVERLAP_KEY]):.6f} |"
        ),
        (
            f"| teacher_top5_overlap | "
            f"{float(teacher_agreement[student_distillation.TEACHER_TOP5_OVERLAP_KEY]):.6f} |"
        ),
        "",
        "## Model Profile",
        "",
        "| model | params | size_mb | batch1_ms | batch64_ms |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Teacher | {profiles['teacher']['total_parameters']} | "
            f"{float(profiles['teacher']['model_size_mb']):.6f} | "
            f"{float(profiles['teacher']['batch1_latency']['inference_time_ms']):.6f} | "
            f"{float(profiles['teacher']['batch64_latency']['inference_time_ms']):.6f} |"
        ),
        (
            f"| Student | {profiles['student']['total_parameters']} | "
            f"{float(profiles['student']['model_size_mb']):.6f} | "
            f"{float(profiles['student']['batch1_latency']['inference_time_ms']):.6f} | "
            f"{float(profiles['student']['batch64_latency']['inference_time_ms']):.6f} |"
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> TitLeNetStudentExperimentResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    base_distillation_args = distillation_args_from_experiment_args(args)
    config = student_distillation.load_student_distillation_config(
        base_distillation_args
    )
    student_distillation.validate_distillation_config(config)
    student_only_paths = load_student_only_output_paths(args)
    experiment_paths = load_experiment_output_paths(args)
    finetune_overrides = load_distillation_finetune_overrides(args)

    student_only_result = run_student_only(
        config=config,
        output_paths=student_only_paths,
    )
    distillation_args = distillation_args_from_experiment_args(
        args,
        student_init_checkpoint=(
            args.student_init_checkpoint
            or student_only_result.checkpoint_paths["best"]
        ),
        finetune_overrides=finetune_overrides,
    )
    distillation_result = student_distillation.run(distillation_args)
    payload = build_comparison_payload(
        config=config,
        student_only=student_only_result,
        distillation=distillation_result,
    )
    write_metrics_json(experiment_paths.metrics_path, payload)
    write_experiment_report(experiment_paths.report_path, payload)

    return TitLeNetStudentExperimentResult(
        student_only=student_only_result,
        distillation=distillation_result,
        report_path=experiment_paths.report_path,
        metrics_path=experiment_paths.metrics_path,
        metrics_payload=payload,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
