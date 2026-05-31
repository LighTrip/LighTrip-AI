from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import (
    run_titlenet_student_experiment as student_experiment,
)
from experiments.title_color_recommendation import (
    train_titlenet_student_distillation as student_distillation,
)
from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = student_distillation.DEFAULT_CONFIG
SWEEP_SECTION = "kd_weight_sweep"
PHASE_FROM_SCRATCH = "from_scratch"
PHASE_WARM_START = "warm_start"
DEFAULT_TRIALS = (
    ("kd_50_50", 0.5, 0.5),
    ("kd_70_30", 0.7, 0.3),
    ("kd_80_20", 0.8, 0.2),
    ("kd_90_10", 0.9, 0.1),
)
DEFAULT_FROM_SCRATCH_EPOCHS = 20
DEFAULT_FROM_SCRATCH_LR = 5e-4
DEFAULT_WARM_START_EPOCHS = 10
DEFAULT_WARM_START_LR = 1e-4
DEFAULT_CHECKPOINT_ROOT = Path("outputs/checkpoints/titlenet_student_kd_weight_sweep")
DEFAULT_LOG_ROOT = Path("outputs/logs/titlenet_student_kd_weight_sweep")
DEFAULT_REPORT_ROOT = Path(
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep"
)
DEFAULT_SWEEP_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_report.md"
)
DEFAULT_SWEEP_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_metrics.json"
)
DEFAULT_SWEEP_RESULTS_CSV = Path(
    "outputs/reports/model_evaluation/titlenet_student_kd_weight_sweep_results.csv"
)
RESULT_FIELDS = (
    "phase",
    "trial",
    "base_loss_weight",
    "distillation_loss_weight",
    "temperature",
    "student_init_checkpoint",
    "epochs",
    "learning_rate",
    "best_epoch",
    "best_metric_value",
    "test_loss",
    "test_ndcg@3",
    "test_ndcg@5",
    "teacher_top1_agreement",
    "teacher_top3_overlap",
    "teacher_top5_overlap",
    "checkpoint_best",
    "report_path",
)


@dataclass(frozen=True)
class WeightTrial:
    name: str
    base_loss_weight: float
    distillation_loss_weight: float


@dataclass(frozen=True)
class PhaseConfig:
    epochs: int
    learning_rate: float
    checkpoint_root: Path
    log_root: Path
    report_root: Path


@dataclass(frozen=True)
class SweepOutputPaths:
    report_path: Path
    metrics_path: Path
    results_csv: Path


@dataclass(frozen=True)
class KDWeightSweepConfig:
    trials: tuple[WeightTrial, ...]
    from_scratch: PhaseConfig
    warm_start: PhaseConfig
    outputs: SweepOutputPaths


@dataclass(frozen=True)
class KDWeightSweepResult:
    student_only: student_experiment.run_full_training.FullTrainingResult
    from_scratch: list[student_distillation.StudentDistillationResult]
    warm_start: list[student_distillation.StudentDistillationResult]
    report_path: Path
    metrics_path: Path
    results_csv: Path
    rows: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a rigorous TitLeNet Student KD weight sweep for both "
            "from-scratch and warm-start distillation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--student-only-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--student-only-log-path", type=Path, default=None)
    parser.add_argument("--student-only-report-path", type=Path, default=None)
    parser.add_argument("--student-only-loss-plot-path", type=Path, default=None)
    parser.add_argument("--student-only-ndcg-plot-path", type=Path, default=None)
    parser.add_argument("--student-only-color-plot-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--best-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--from-scratch-epochs", type=int, default=None)
    parser.add_argument("--from-scratch-learning-rate", type=float, default=None)
    parser.add_argument("--warm-start-epochs", type=int, default=None)
    parser.add_argument("--warm-start-learning-rate", type=float, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    return parser.parse_args(argv)


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _load_raw_config(path: Path) -> Mapping[str, Any]:
    config_path = resolve_inside_project(
        PROJECT_ROOT,
        path,
        must_exist=True,
        description="student kd weight sweep config",
    )
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return _require_mapping(payload, description="student kd weight sweep config")


def _resolve_output_path(value: str | Path, *, description: str) -> Path:
    return resolve_inside_project(PROJECT_ROOT, Path(value), description=description)


def _section_path(
    section: Mapping[str, Any],
    key: str,
    default: Path,
    *,
    override: Path | None = None,
) -> Path:
    value = override if override is not None else section.get(key, default)
    return _resolve_output_path(value, description=key)


def _load_trials(section: Mapping[str, Any]) -> tuple[WeightTrial, ...]:
    raw_trials = section.get("trials")
    if raw_trials is None:
        return tuple(
            WeightTrial(
                name=name,
                base_loss_weight=base_weight,
                distillation_loss_weight=kd_weight,
            )
            for name, base_weight, kd_weight in DEFAULT_TRIALS
        )
    if not isinstance(raw_trials, list) or not raw_trials:
        raise ValueError("kd_weight_sweep.trials must be a non-empty list")

    trials: list[WeightTrial] = []
    for raw_trial in raw_trials:
        trial = _require_mapping(raw_trial, description="kd weight trial")
        trials.append(
            WeightTrial(
                name=str(trial["name"]),
                base_loss_weight=float(trial["base_loss_weight"]),
                distillation_loss_weight=float(trial["distillation_loss_weight"]),
            )
        )
    return tuple(trials)


def _phase_config(
    section: Mapping[str, Any],
    phase: str,
    *,
    default_epochs: int,
    default_learning_rate: float,
    epochs_override: int | None,
    learning_rate_override: float | None,
) -> PhaseConfig:
    phase_section = _require_mapping(section.get(phase), description=phase)
    checkpoint_default = DEFAULT_CHECKPOINT_ROOT / phase
    log_default = DEFAULT_LOG_ROOT / phase
    report_default = DEFAULT_REPORT_ROOT / phase
    return PhaseConfig(
        epochs=int(
            epochs_override
            if epochs_override is not None
            else phase_section.get("epochs", default_epochs)
        ),
        learning_rate=float(
            learning_rate_override
            if learning_rate_override is not None
            else phase_section.get("learning_rate", default_learning_rate)
        ),
        checkpoint_root=_section_path(
            phase_section,
            "checkpoint_root",
            checkpoint_default,
        ),
        log_root=_section_path(phase_section, "log_root", log_default),
        report_root=_section_path(phase_section, "report_root", report_default),
    )


def load_sweep_config(args: argparse.Namespace) -> KDWeightSweepConfig:
    raw_config = _load_raw_config(args.config)
    section = _require_mapping(raw_config.get(SWEEP_SECTION), description=SWEEP_SECTION)
    outputs = _require_mapping(section.get("outputs"), description="sweep outputs")
    return KDWeightSweepConfig(
        trials=_load_trials(section),
        from_scratch=_phase_config(
            section,
            PHASE_FROM_SCRATCH,
            default_epochs=DEFAULT_FROM_SCRATCH_EPOCHS,
            default_learning_rate=DEFAULT_FROM_SCRATCH_LR,
            epochs_override=args.from_scratch_epochs,
            learning_rate_override=args.from_scratch_learning_rate,
        ),
        warm_start=_phase_config(
            section,
            PHASE_WARM_START,
            default_epochs=DEFAULT_WARM_START_EPOCHS,
            default_learning_rate=DEFAULT_WARM_START_LR,
            epochs_override=args.warm_start_epochs,
            learning_rate_override=args.warm_start_learning_rate,
        ),
        outputs=SweepOutputPaths(
            report_path=_section_path(
                outputs,
                "report_path",
                DEFAULT_SWEEP_REPORT_PATH,
                override=args.report_path,
            ),
            metrics_path=_section_path(
                outputs,
                "metrics_path",
                DEFAULT_SWEEP_METRICS_PATH,
                override=args.metrics_path,
            ),
            results_csv=_section_path(
                outputs,
                "results_csv",
                DEFAULT_SWEEP_RESULTS_CSV,
                override=args.results_csv,
            ),
        ),
    )


def _append_optional_arg(cli_args: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cli_args.extend([flag, str(value)])


def base_distillation_args(args: argparse.Namespace) -> list[str]:
    cli_args = ["--config", str(args.config)]
    _append_optional_arg(cli_args, "--data-root", args.data_root)
    _append_optional_arg(cli_args, "--labels-matrix", args.labels_matrix)
    _append_optional_arg(cli_args, "--labels-soft", args.labels_soft)
    _append_optional_arg(cli_args, "--teacher-checkpoint", args.teacher_checkpoint)
    _append_optional_arg(cli_args, "--batch-size", args.batch_size)
    _append_optional_arg(cli_args, "--weight-decay", args.weight_decay)
    _append_optional_arg(cli_args, "--num-workers", args.num_workers)
    _append_optional_arg(cli_args, "--device", args.device)
    _append_optional_arg(cli_args, "--scheduler", args.scheduler)
    _append_optional_arg(cli_args, "--best-metric", args.best_metric)
    _append_optional_arg(cli_args, "--seed", args.seed)
    _append_optional_arg(cli_args, "--temperature", args.temperature)
    return cli_args


def distillation_args_for_trial(
    *,
    args: argparse.Namespace,
    phase: str,
    phase_config: PhaseConfig,
    trial: WeightTrial,
    student_init_checkpoint: Path | None,
) -> argparse.Namespace:
    checkpoint_dir = phase_config.checkpoint_root / trial.name
    log_path = phase_config.log_root / f"{trial.name}.jsonl"
    report_path = phase_config.report_root / trial.name / "report.md"
    metrics_path = phase_config.report_root / trial.name / "metrics.json"
    cli_args = [
        *base_distillation_args(args),
        "--epochs",
        str(phase_config.epochs),
        "--learning-rate",
        str(phase_config.learning_rate),
        "--base-loss-weight",
        str(trial.base_loss_weight),
        "--distillation-loss-weight",
        str(trial.distillation_loss_weight),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--log-path",
        str(log_path),
        "--report-path",
        str(report_path),
        "--metrics-path",
        str(metrics_path),
    ]
    if student_init_checkpoint is not None:
        cli_args.extend(["--student-init-checkpoint", str(student_init_checkpoint)])
    LOGGER.info(
        "running %s trial=%s base=%.2f kd=%.2f",
        phase,
        trial.name,
        trial.base_loss_weight,
        trial.distillation_loss_weight,
    )
    return student_distillation.parse_args(cli_args)


def student_only_args(args: argparse.Namespace) -> argparse.Namespace:
    cli_args = ["--config", str(args.config)]
    _append_optional_arg(cli_args, "--data-root", args.data_root)
    _append_optional_arg(cli_args, "--labels-matrix", args.labels_matrix)
    _append_optional_arg(cli_args, "--labels-soft", args.labels_soft)
    _append_optional_arg(cli_args, "--teacher-checkpoint", args.teacher_checkpoint)
    _append_optional_arg(cli_args, "--batch-size", args.batch_size)
    _append_optional_arg(cli_args, "--weight-decay", args.weight_decay)
    _append_optional_arg(cli_args, "--num-workers", args.num_workers)
    _append_optional_arg(cli_args, "--device", args.device)
    _append_optional_arg(cli_args, "--scheduler", args.scheduler)
    _append_optional_arg(cli_args, "--best-metric", args.best_metric)
    _append_optional_arg(cli_args, "--seed", args.seed)
    _append_optional_arg(
        cli_args,
        "--student-only-checkpoint-dir",
        args.student_only_checkpoint_dir,
    )
    _append_optional_arg(cli_args, "--student-only-log-path", args.student_only_log_path)
    _append_optional_arg(
        cli_args,
        "--student-only-report-path",
        args.student_only_report_path,
    )
    _append_optional_arg(
        cli_args,
        "--student-only-loss-plot-path",
        args.student_only_loss_plot_path,
    )
    _append_optional_arg(
        cli_args,
        "--student-only-ndcg-plot-path",
        args.student_only_ndcg_plot_path,
    )
    _append_optional_arg(
        cli_args,
        "--student-only-color-plot-path",
        args.student_only_color_plot_path,
    )
    return student_experiment.parse_args(cli_args)


def run_student_only(
    args: argparse.Namespace,
) -> student_experiment.run_full_training.FullTrainingResult:
    experiment_args = student_only_args(args)
    distillation_args = student_experiment.distillation_args_from_experiment_args(
        experiment_args
    )
    config = student_distillation.load_student_distillation_config(distillation_args)
    paths = student_experiment.load_student_only_output_paths(experiment_args)
    return student_experiment.run_student_only(config=config, output_paths=paths)


def result_row(
    *,
    phase: str,
    trial: WeightTrial,
    result: student_distillation.StudentDistillationResult,
) -> dict[str, Any]:
    metrics = result.test_metrics.as_dict()
    agreement = result.test_agreement
    training = result.metrics_payload["training"]
    distillation = result.metrics_payload["distillation"]
    return {
        "phase": phase,
        "trial": trial.name,
        "base_loss_weight": trial.base_loss_weight,
        "distillation_loss_weight": trial.distillation_loss_weight,
        "temperature": distillation["temperature"],
        "student_init_checkpoint": result.metrics_payload["student_init_checkpoint"],
        "epochs": training["epochs"],
        "learning_rate": training["learning_rate"],
        "best_epoch": result.best_epoch,
        "best_metric_value": result.best_metric_value,
        "test_loss": metrics[student_distillation.VAL_LOSS_KEY],
        "test_ndcg@3": metrics[student_distillation.VAL_NDCG_AT_3_KEY],
        "test_ndcg@5": metrics[student_distillation.VAL_NDCG_AT_5_KEY],
        "teacher_top1_agreement": agreement[
            student_distillation.TEACHER_TOP1_AGREEMENT_KEY
        ],
        "teacher_top3_overlap": agreement[
            student_distillation.TEACHER_TOP3_OVERLAP_KEY
        ],
        "teacher_top5_overlap": agreement[
            student_distillation.TEACHER_TOP5_OVERLAP_KEY
        ],
        "checkpoint_best": str(result.checkpoint_paths["best"]),
        "report_path": str(result.report_path),
    }


def write_results_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics_json(
    path: Path,
    *,
    student_only: student_experiment.run_full_training.FullTrainingResult,
    rows: list[Mapping[str, Any]],
    sweep_config: KDWeightSweepConfig,
) -> None:
    payload = {
        "student_only": {
            "best_epoch": student_only.best_epoch,
            "best_metric_value": student_only.best_metric_value,
            "test_metrics": student_only.test_metrics.as_dict(),
            "checkpoint_paths": {
                name: str(path)
                for name, path in student_only.checkpoint_paths.items()
            },
            "report_path": str(student_only.report_path),
        },
        "sweep_config": {
            "trials": [asdict(trial) for trial in sweep_config.trials],
            "from_scratch": {
                "epochs": sweep_config.from_scratch.epochs,
                "learning_rate": sweep_config.from_scratch.learning_rate,
                "checkpoint_root": str(sweep_config.from_scratch.checkpoint_root),
                "log_root": str(sweep_config.from_scratch.log_root),
                "report_root": str(sweep_config.from_scratch.report_root),
            },
            "warm_start": {
                "epochs": sweep_config.warm_start.epochs,
                "learning_rate": sweep_config.warm_start.learning_rate,
                "checkpoint_root": str(sweep_config.warm_start.checkpoint_root),
                "log_root": str(sweep_config.warm_start.log_root),
                "report_root": str(sweep_config.warm_start.report_root),
            },
        },
        "rows": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _format_float(value: Any) -> str:
    return f"{float(value):.6f}"


def write_report(
    path: Path,
    *,
    student_only: student_experiment.run_full_training.FullTrainingResult,
    rows: list[Mapping[str, Any]],
    results_csv: Path,
    metrics_path: Path,
) -> None:
    lines = [
        "# TitLeNet Student KD Weight Sweep",
        "",
        "## Student-Only Baseline",
        "",
        "| metric | value |",
        "| --- | ---: |",
        (
            f"| test_loss | "
            f"{student_only.test_metrics.val_loss:.6f} |"
        ),
        (
            f"| NDCG@3 | "
            f"{student_only.test_metrics.val_ndcg_at_3:.6f} |"
        ),
        (
            f"| NDCG@5 | "
            f"{student_only.test_metrics.val_ndcg_at_5:.6f} |"
        ),
        "",
        "## KD Sweep Results",
        "",
        (
            "| phase | trial | base | kd | epochs | lr | "
            "NDCG@3 | NDCG@5 | teacher top-1 | report |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['phase']} | {row['trial']} | "
            f"{float(row['base_loss_weight']):.2f} | "
            f"{float(row['distillation_loss_weight']):.2f} | "
            f"{int(row['epochs'])} | {float(row['learning_rate']):.6f} | "
            f"{_format_float(row['test_ndcg@3'])} | "
            f"{_format_float(row['test_ndcg@5'])} | "
            f"{_format_float(row['teacher_top1_agreement'])} | "
            f"`{row['report_path']}` |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- results_csv: `{results_csv}`",
            f"- metrics_path: `{metrics_path}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> KDWeightSweepResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sweep_config = load_sweep_config(args)
    student_only = run_student_only(args)
    student_init_checkpoint = student_only.checkpoint_paths["best"]

    from_scratch_results: list[student_distillation.StudentDistillationResult] = []
    warm_start_results: list[student_distillation.StudentDistillationResult] = []
    rows: list[dict[str, Any]] = []

    for trial in sweep_config.trials:
        result = student_distillation.run(
            distillation_args_for_trial(
                args=args,
                phase=PHASE_FROM_SCRATCH,
                phase_config=sweep_config.from_scratch,
                trial=trial,
                student_init_checkpoint=None,
            )
        )
        from_scratch_results.append(result)
        rows.append(result_row(phase=PHASE_FROM_SCRATCH, trial=trial, result=result))

    for trial in sweep_config.trials:
        result = student_distillation.run(
            distillation_args_for_trial(
                args=args,
                phase=PHASE_WARM_START,
                phase_config=sweep_config.warm_start,
                trial=trial,
                student_init_checkpoint=student_init_checkpoint,
            )
        )
        warm_start_results.append(result)
        rows.append(result_row(phase=PHASE_WARM_START, trial=trial, result=result))

    write_results_csv(sweep_config.outputs.results_csv, rows)
    write_metrics_json(
        sweep_config.outputs.metrics_path,
        student_only=student_only,
        rows=rows,
        sweep_config=sweep_config,
    )
    write_report(
        sweep_config.outputs.report_path,
        student_only=student_only,
        rows=rows,
        results_csv=sweep_config.outputs.results_csv,
        metrics_path=sweep_config.outputs.metrics_path,
    )

    return KDWeightSweepResult(
        student_only=student_only,
        from_scratch=from_scratch_results,
        warm_start=warm_start_results,
        report_path=sweep_config.outputs.report_path,
        metrics_path=sweep_config.outputs.metrics_path,
        results_csv=sweep_config.outputs.results_csv,
        rows=rows,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
