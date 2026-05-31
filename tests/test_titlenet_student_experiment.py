from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def experiment_module() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_titlenet_student_experiment"
    )


def test_student_experiment_loads_output_paths(
    experiment_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(experiment_module, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "student_experiment.yaml"
    config_path.write_text(
        "\n".join(
            [
                "student_only:",
                "  checkpoint_dir: outputs/checkpoints/student_only",
                "  log_path: outputs/logs/student_only.jsonl",
                "  report_path: outputs/reports/student_only.md",
                "  loss_plot_path: outputs/reports/student_only_loss.png",
                "  ndcg_plot_path: outputs/reports/student_only_ndcg.png",
                "  color_plot_path: outputs/reports/student_only_colors.png",
                "experiment_outputs:",
                "  report_path: outputs/reports/comparison.md",
                "  metrics_path: outputs/reports/comparison.json",
            ]
        ),
        encoding="utf-8",
    )

    args = experiment_module.parse_args(["--config", str(config_path)])
    student_only = experiment_module.load_student_only_output_paths(args)
    experiment = experiment_module.load_experiment_output_paths(args)

    assert student_only.checkpoint_dir == tmp_path / "outputs/checkpoints/student_only"
    assert student_only.log_path == tmp_path / "outputs/logs/student_only.jsonl"
    assert experiment.report_path == tmp_path / "outputs/reports/comparison.md"
    assert experiment.metrics_path == tmp_path / "outputs/reports/comparison.json"


def test_full_training_args_for_student_only_uses_student_config(
    experiment_module: Any,
    tmp_path: Path,
) -> None:
    distillation = experiment_module.student_distillation
    training = distillation.TrainingConfig(
        batch_size=4,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.01,
        num_workers=0,
        device="cpu",
        scheduler="none",
        best_metric="val_ndcg@5",
        seed=7,
        model_name="titlenet_student",
        dropout=0.2,
        weight_init="small_head",
        activation="hardswish",
    )
    config = distillation.StudentDistillationConfig(
        teacher=distillation.ModelBuildConfig(
            model_name="titlenet",
            dropout=0.2,
            weight_init="small_head",
            activation="gelu",
        ),
        student=distillation.ModelBuildConfig(
            model_name="titlenet_student",
            dropout=0.2,
            weight_init="small_head",
            activation="hardswish",
        ),
        training=training,
        loss=distillation.DistillationLossConfig(
            temperature=2.0,
            base_loss_weight=0.5,
            distillation_loss_weight=0.5,
        ),
        latency=distillation.LatencyConfig(warmup_steps=0, benchmark_steps=1),
        data_root=tmp_path / "data",
        labels_matrix=None,
        labels_soft=None,
        teacher_checkpoint=tmp_path / "teacher.pt",
        student_init_checkpoint=None,
        report_path=tmp_path / "distill.md",
        metrics_path=tmp_path / "distill.json",
    )
    output_paths = experiment_module.StudentOnlyOutputPaths(
        checkpoint_dir=tmp_path / "checkpoints",
        log_path=tmp_path / "student.jsonl",
        report_path=tmp_path / "student.md",
        loss_plot_path=tmp_path / "loss.png",
        ndcg_plot_path=tmp_path / "ndcg.png",
        color_plot_path=tmp_path / "colors.png",
    )

    args = experiment_module.full_training_args_for_student_only(
        config=config,
        output_paths=output_paths,
    )

    assert args.model_name == "titlenet_student"
    assert args.activation == "hardswish"
    assert args.epochs == 2
    assert args.batch_size == 4
    assert args.device == "cpu"
    assert args.checkpoint_dir == tmp_path / "checkpoints"


def test_student_experiment_report_and_metrics_are_written(
    experiment_module: Any,
    tmp_path: Path,
) -> None:
    payload = {
        "student_only": {
            "checkpoint_paths": {"best": "student_only_best.pt"},
            "report_path": "student_only.md",
            "test_metrics": {
                "val_loss": 0.3,
                "val_ndcg@3": 0.91,
                "val_ndcg@5": 0.92,
            },
        },
        "student_init_checkpoint": "student_only_best.pt",
        "distillation": {
            "temperature": 2.0,
            "base_loss_weight": 0.8,
            "distillation_loss_weight": 0.2,
        },
        "distillation_training": {
            "epochs": 10,
            "learning_rate": 0.0001,
        },
        "student_distilled": {
            "checkpoint_paths": {"best": "student_distilled_best.pt"},
            "report_path": "student_distilled.md",
            "test_metrics": {
                "val_loss": 0.25,
                "val_ndcg@3": 0.93,
                "val_ndcg@5": 0.94,
            },
            "teacher_agreement": {
                "teacher_top1_agreement": 0.8,
                "teacher_top3_overlap": 0.9,
                "teacher_top5_overlap": 1.0,
            },
        },
        "profiles": {
            "teacher": {
                "total_parameters": 183732,
                "model_size_mb": 0.714,
                "batch1_latency": {"inference_time_ms": 2.4},
                "batch64_latency": {"inference_time_ms": 3.0},
            },
            "student": {
                "total_parameters": 73542,
                "model_size_mb": 0.288,
                "batch1_latency": {"inference_time_ms": 1.2},
                "batch64_latency": {"inference_time_ms": 2.0},
            },
        },
        "comparison": {
            "loss_delta_distilled_vs_student_only": -0.05,
            "ndcg@3_delta_distilled_vs_student_only": 0.02,
            "ndcg@5_delta_distilled_vs_student_only": 0.02,
        },
        "outcome_checks": {
            "ndcg@3_improved_or_equal": True,
            "ndcg@5_improved_or_equal": True,
            "loss_improved_or_equal": True,
            "ndcg@3_and_ndcg@5_improved_or_equal": True,
        },
    }
    report_path = tmp_path / "report.md"
    metrics_path = tmp_path / "metrics.json"

    experiment_module.write_experiment_report(report_path, payload)
    experiment_module.write_metrics_json(metrics_path, payload)

    report = report_path.read_text(encoding="utf-8")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "TitLeNet Student Experiment Report" in report
    assert "Student-only" in report
    assert "Student-distilled" in report
    assert "NDCG@3 improved or equal" in report
    assert metrics["comparison"]["ndcg@5_delta_distilled_vs_student_only"] == 0.02
