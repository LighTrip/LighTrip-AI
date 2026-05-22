from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.title_color_experiment_helpers import (
    tiny_classifier,
    tiny_split_loaders,
)


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def nn_module() -> Any:
    return pytest.importorskip("torch.nn")


@pytest.fixture()
def full_training_module() -> Any:
    pytest.importorskip("torch")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_full_training"
    )


def test_build_training_config_uses_full_training_defaults(
    full_training_module: Any,
) -> None:
    args = full_training_module.parse_args([])

    config = full_training_module.build_training_config(args)

    assert config.batch_size == 64
    assert config.epochs == 20
    assert math.isclose(config.learning_rate, 3e-4, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(config.weight_decay, 1e-4, rel_tol=0.0, abs_tol=1e-12)
    assert config.checkpoint_dir == "outputs/checkpoints"
    assert config.log_path == "outputs/logs/training_metrics.jsonl"


def test_full_training_report_records_test_metrics_and_plots(
    full_training_module: Any,
    tmp_path: Path,
) -> None:
    config = full_training_module.TrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        device="cpu",
        scheduler="none",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "logs" / "training_metrics.jsonl"),
    )
    history = [
        {
            "epoch": 1,
            "train_loss": 0.8,
            "val_loss": 0.7,
            "val_ndcg@5": 0.4,
            "top1_wcag_pass_rate": 0.5,
            "color_distribution": [0.4, 0.6] + [0.0] * 30,
        }
    ]
    test_metrics = full_training_module.ValidationMetrics(
        val_loss=0.6,
        val_ndcg_at_5=0.5,
        top1_wcag_pass_rate=0.7,
        color_distribution=[0.3, 0.7] + [0.0] * 30,
    )
    plot_paths = {
        "loss": tmp_path / "reports" / "loss_curve.png",
        "ndcg": tmp_path / "reports" / "ndcg5_curve.png",
        "color_distribution": tmp_path / "reports" / "color_distribution.png",
    }
    report_path = tmp_path / "reports" / "full_training_report.md"

    full_training_module.write_full_training_report(
        report_path,
        config=config,
        history=history,
        test_metrics=test_metrics,
        best_epoch=1,
        best_metric_value=0.7,
        dataset_sizes={"train": 4, "val": 2, "test": 2},
        checkpoint_paths={
            "best": tmp_path / "checkpoints" / "checkpoint_best.pt",
            "latest": tmp_path / "checkpoints" / "checkpoint_latest.pt",
        },
        plot_paths=plot_paths,
        collapse_threshold=0.8,
        pretrained=False,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "status: `PASS`" in report
    assert "checkpoint_best.pt" in report
    assert "![Loss Curve](loss_curve.png)" in report
    assert "![NDCG Curve](ndcg5_curve.png)" in report
    assert "![Color Distribution](color_distribution.png)" in report


def test_run_executes_one_epoch_with_stubbed_loaders(
    full_training_module: Any,
    torch_module: Any,
    nn_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))

    monkeypatch.setattr(
        full_training_module,
        "create_title_color_dataloaders",
        lambda **_kwargs: tiny_split_loaders(torch_module),
    )
    monkeypatch.setattr(
        full_training_module,
        "build_fixed_palette_resnet18",
        lambda **_kwargs: tiny_classifier(nn_module),
    )

    args = full_training_module.parse_args(
        [
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--scheduler",
            "none",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--log-path",
            str(tmp_path / "logs" / "training_metrics.jsonl"),
            "--report-path",
            str(tmp_path / "reports" / "full_training_report.md"),
            "--loss-plot-path",
            str(tmp_path / "reports" / "loss_curve.png"),
            "--ndcg-plot-path",
            str(tmp_path / "reports" / "ndcg5_curve.png"),
            "--color-plot-path",
            str(tmp_path / "reports" / "color_distribution.png"),
        ]
    )

    result = full_training_module.run(args)

    assert len(result.history) == 1
    assert result.best_epoch == 1
    assert result.checkpoint_paths["best"].exists()
    assert result.checkpoint_paths["latest"].exists()
    assert result.report_path.exists()
    for plot_path in result.plot_paths.values():
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    log_path = Path(args.log_path)
    log_records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(log_records) == 1
    assert log_records[0]["epoch"] == 1
