from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.title_color_experiment_helpers import (
    TinyTitleColorDataset,
    tiny_classifier,
)


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def nn_module() -> Any:
    return pytest.importorskip("torch.nn")


@pytest.fixture()
def sweep_module() -> Any:
    pytest.importorskip("torch")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_hyperparameter_sweep"
    )


def _sweep_spec_path(tmp_path: Path) -> Path:
    path = tmp_path / "sweep.json"
    path.write_text(
        json.dumps(
            {
                "base": {
                    "batch_size": 2,
                    "epochs": 1,
                    "learning_rate": 1e-3,
                    "weight_decay": 0.0,
                    "num_workers": 0,
                    "device": "cpu",
                    "scheduler": "none",
                    "best_metric": "val_loss",
                    "seed": 7,
                },
                "trials": [
                    {
                        "name": "lr_a",
                        "learning_rate": 1e-3,
                    },
                    {
                        "name": "lr_b",
                        "learning_rate": 5e-4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_sweep_trials_applies_overrides(
    sweep_module: Any,
    tmp_path: Path,
) -> None:
    args = sweep_module.parse_args(
        [
            "--sweep-config",
            str(_sweep_spec_path(tmp_path)),
            "--epochs",
            "3",
            "--max-trials",
            "1",
        ]
    )
    spec = sweep_module.load_sweep_spec(args.sweep_config)

    trials = sweep_module.build_sweep_trials(spec, args)

    assert len(trials) == 1
    assert trials[0].name == "lr_a"
    assert trials[0].config.epochs == 3
    assert trials[0].config.best_metric == "val_loss"


def test_pick_best_trial_prefers_lower_validation_loss(
    sweep_module: Any,
) -> None:
    config = sweep_module.TrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        scheduler="none",
    )
    worse = sweep_module.SweepTrialResult(
        name="worse",
        config=config,
        best_epoch=1,
        selection_metric="val_loss",
        selection_metric_value=0.5,
        best_val_loss=0.5,
        best_val_ndcg_at_5=0.8,
        best_top1_wcag_pass_rate=0.5,
        best_max_color_share=0.4,
        final_train_loss=0.6,
        final_val_loss=0.5,
        checkpoint_dir=Path("a"),
        log_path=Path("a.jsonl"),
    )
    better = sweep_module.SweepTrialResult(
        name="better",
        config=config,
        best_epoch=1,
        selection_metric="val_loss",
        selection_metric_value=0.3,
        best_val_loss=0.3,
        best_val_ndcg_at_5=0.9,
        best_top1_wcag_pass_rate=0.6,
        best_max_color_share=0.5,
        final_train_loss=0.4,
        final_val_loss=0.3,
        checkpoint_dir=Path("b"),
        log_path=Path("b.jsonl"),
    )

    best = sweep_module.pick_best_trial([worse, better])

    assert best.name == "better"
    assert math.isclose(best.best_val_loss, 0.3, rel_tol=0.0, abs_tol=1e-12)


def test_run_sweep_writes_ranking_report_and_artifacts(
    sweep_module: Any,
    torch_module: Any,
    nn_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    monkeypatch.setattr(
        sweep_module,
        "create_sweep_datasets",
        lambda _args: {
            "train": TinyTitleColorDataset(torch_module, length=4),
            "val": TinyTitleColorDataset(torch_module, length=2),
        },
    )
    monkeypatch.setattr(
        sweep_module,
        "build_fixed_palette_resnet18",
        lambda **_kwargs: tiny_classifier(nn_module),
    )

    args = sweep_module.parse_args(
        [
            "--sweep-config",
            str(_sweep_spec_path(tmp_path)),
            "--output-dir",
            str(tmp_path / "sweep"),
            "--report-path",
            str(tmp_path / "reports" / "sweep.md"),
            "--results-csv-path",
            str(tmp_path / "reports" / "sweep.csv"),
            "--val-loss-plot-path",
            str(tmp_path / "reports" / "val_loss.png"),
            "--ndcg-plot-path",
            str(tmp_path / "reports" / "ndcg.png"),
            "--safety-plot-path",
            str(tmp_path / "reports" / "safety.png"),
        ]
    )

    result = sweep_module.run(args)

    assert len(result.trials) == 2
    assert result.report_path.exists()
    assert result.results_csv_path.exists()
    for plot_path in result.plot_paths.values():
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    report = result.report_path.read_text(encoding="utf-8")
    assert "test_split_used: `False`" in report
    assert "## Trial Ranking" in report

    with result.results_csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["selection_metric"] == "val_loss"
