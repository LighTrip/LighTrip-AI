from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def sweep_module() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_titlenet_student_kd_weight_sweep"
    )


def test_kd_weight_sweep_loads_trials_and_outputs(
    sweep_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sweep_module, "PROJECT_ROOT", tmp_path)
    config_path = tmp_path / "student_distillation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "kd_weight_sweep:",
                "  trials:",
                "    - name: kd_test",
                "      base_loss_weight: 0.8",
                "      distillation_loss_weight: 0.2",
                "  from_scratch:",
                "    epochs: 3",
                "    learning_rate: 0.001",
                "    checkpoint_root: outputs/checkpoints/from_scratch",
                "    log_root: outputs/logs/from_scratch",
                "    report_root: outputs/reports/from_scratch",
                "  warm_start:",
                "    epochs: 2",
                "    learning_rate: 0.0001",
                "    checkpoint_root: outputs/checkpoints/warm_start",
                "    log_root: outputs/logs/warm_start",
                "    report_root: outputs/reports/warm_start",
                "  outputs:",
                "    report_path: outputs/reports/sweep.md",
                "    metrics_path: outputs/reports/sweep.json",
                "    results_csv: outputs/reports/sweep.csv",
            ]
        ),
        encoding="utf-8",
    )

    args = sweep_module.parse_args(["--config", str(config_path)])
    config = sweep_module.load_sweep_config(args)

    assert len(config.trials) == 1
    assert config.trials[0].name == "kd_test"
    assert config.from_scratch.epochs == 3
    assert config.warm_start.epochs == 2
    assert config.outputs.report_path == tmp_path / "outputs/reports/sweep.md"


def test_distillation_args_for_trial_sets_phase_outputs(
    sweep_module: Any,
    tmp_path: Path,
) -> None:
    args = sweep_module.parse_args(["--config", "config.yaml", "--num-workers", "0"])
    trial = sweep_module.WeightTrial(
        name="kd_80_20",
        base_loss_weight=0.8,
        distillation_loss_weight=0.2,
    )
    phase_config = sweep_module.PhaseConfig(
        epochs=5,
        learning_rate=0.0001,
        checkpoint_root=tmp_path / "checkpoints",
        log_root=tmp_path / "logs",
        report_root=tmp_path / "reports",
    )
    init_checkpoint = tmp_path / "student_only.pt"

    distill_args = sweep_module.distillation_args_for_trial(
        args=args,
        phase=sweep_module.PHASE_WARM_START,
        phase_config=phase_config,
        trial=trial,
        student_init_checkpoint=init_checkpoint,
    )

    assert distill_args.epochs == 5
    assert distill_args.learning_rate == 0.0001
    assert distill_args.base_loss_weight == 0.8
    assert distill_args.distillation_loss_weight == 0.2
    assert distill_args.student_init_checkpoint == init_checkpoint
    assert distill_args.checkpoint_dir == tmp_path / "checkpoints" / "kd_80_20"


def test_sweep_report_csv_and_metrics_are_written(
    sweep_module: Any,
    tmp_path: Path,
) -> None:
    metrics = sweep_module.student_distillation.ValidationMetrics(
        val_loss=0.3,
        val_ndcg_at_3=0.91,
        val_ndcg_at_5=0.92,
        top1_wcag_pass_rate=0.8,
        top5_any_wcag_pass_rate=0.9,
        color_distribution=[1.0] + [0.0] * 31,
    )

    class StudentOnlyResult:
        best_epoch = 2
        best_metric_value = 0.92
        test_metrics = metrics
        checkpoint_paths = {"best": tmp_path / "student.pt"}
        report_path = tmp_path / "student.md"

    rows = [
        {
            "phase": "warm_start",
            "trial": "kd_80_20",
            "base_loss_weight": 0.8,
            "distillation_loss_weight": 0.2,
            "temperature": 2.0,
            "student_init_checkpoint": str(tmp_path / "student.pt"),
            "epochs": 10,
            "learning_rate": 0.0001,
            "best_epoch": 5,
            "best_metric_value": 0.94,
            "test_loss": 0.25,
            "test_ndcg@3": 0.93,
            "test_ndcg@5": 0.94,
            "teacher_top1_agreement": 0.88,
            "teacher_top3_overlap": 0.99,
            "teacher_top5_overlap": 1.0,
            "checkpoint_best": str(tmp_path / "best.pt"),
            "report_path": str(tmp_path / "report.md"),
        }
    ]
    csv_path = tmp_path / "results.csv"
    metrics_path = tmp_path / "metrics.json"
    report_path = tmp_path / "report.md"
    sweep_config = sweep_module.KDWeightSweepConfig(
        trials=(
            sweep_module.WeightTrial(
                name="kd_80_20",
                base_loss_weight=0.8,
                distillation_loss_weight=0.2,
            ),
        ),
        from_scratch=sweep_module.PhaseConfig(
            epochs=20,
            learning_rate=0.0005,
            checkpoint_root=tmp_path / "from_scratch",
            log_root=tmp_path / "logs" / "from_scratch",
            report_root=tmp_path / "reports" / "from_scratch",
        ),
        warm_start=sweep_module.PhaseConfig(
            epochs=10,
            learning_rate=0.0001,
            checkpoint_root=tmp_path / "warm_start",
            log_root=tmp_path / "logs" / "warm_start",
            report_root=tmp_path / "reports" / "warm_start",
        ),
        outputs=sweep_module.SweepOutputPaths(
            report_path=report_path,
            metrics_path=metrics_path,
            results_csv=csv_path,
        ),
    )

    sweep_module.write_results_csv(csv_path, rows)
    sweep_module.write_metrics_json(
        metrics_path,
        student_only=StudentOnlyResult(),
        rows=rows,
        sweep_config=sweep_config,
    )
    sweep_module.write_report(
        report_path,
        student_only=StudentOnlyResult(),
        rows=rows,
        results_csv=csv_path,
        metrics_path=metrics_path,
    )

    csv_rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert csv_rows[0]["trial"] == "kd_80_20"
    assert metrics_payload["rows"][0]["test_ndcg@5"] == 0.94
    assert "KD Sweep Results" in report
