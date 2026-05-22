from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def overfit_module() -> Any:
    pytest.importorskip("torch")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_overfit_test"
    )


def test_select_subset_rows_is_deterministic(overfit_module: Any) -> None:
    rows = [
        {"id": f"image_{index:03d}", "split": "train"}
        for index in range(10)
    ]

    first = overfit_module.select_subset_rows(rows, subset_size=4, seed=7)
    second = overfit_module.select_subset_rows(rows, subset_size=4, seed=7)

    assert first == second
    assert len(first) == 4
    assert [row["id"] for row in first] == sorted(row["id"] for row in first)


def test_write_subset_manifest(overfit_module: Any, tmp_path: Path) -> None:
    rows = [
        {"id": "image_a", "split": "train"},
        {"id": "image_b", "split": "train"},
    ]
    path = tmp_path / "subset.csv"

    overfit_module.write_subset_manifest(path, rows)

    with path.open("r", newline="", encoding="utf-8") as f:
        loaded_rows = list(csv.DictReader(f))
    assert loaded_rows == rows


def test_write_report_records_checks_and_metrics(
    overfit_module: Any,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "overfit_test_report.md"
    args = overfit_module.parse_args(
        [
            "--epochs",
            "1",
            "--subset-size",
            "2",
            "--batch-size",
            "1",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0.0",
        ]
    )
    history = [
        {
            "epoch": 0,
            "train_eval_loss": 1.0,
            "train_ndcg@5": 0.2,
            "top1_wcag_pass_rate": 0.5,
            "color_distribution": [0.5, 0.5] + [0.0] * 30,
        },
        {
            "epoch": 1,
            "train_loss": 0.8,
            "train_eval_loss": 0.7,
            "train_ndcg@5": 0.4,
            "top1_wcag_pass_rate": 0.5,
            "color_distribution": [0.5, 0.5] + [0.0] * 30,
        },
    ]
    checks = {
        "train_loss_decreased": True,
        "train_ndcg_increased": True,
        "not_collapsed": True,
    }
    plot_paths = {
        "loss": tmp_path / "overfit_loss_curve.png",
        "ndcg": tmp_path / "overfit_ndcg_curve.png",
        "color_distribution": tmp_path / "overfit_color_distribution.png",
    }

    overfit_module.write_report(
        report_path,
        args=args,
        history=history,
        checks=checks,
        checkpoint_path=Path("outputs/checkpoints/overfit_test.pt"),
        subset_manifest_path=Path("outputs/reports/overfit_train_subset_128.csv"),
        plot_paths=plot_paths,
    )

    report = report_path.read_text(encoding="utf-8")
    assert "status: `PASS`" in report
    assert "train_ndcg@5" in report
    assert "overfit_test.pt" in report
    assert "![Loss Curve](overfit_loss_curve.png)" in report
    assert "![NDCG Curve](overfit_ndcg_curve.png)" in report
    assert "![Color Distribution](overfit_color_distribution.png)" in report


def test_write_overfit_plots_creates_png_files(
    overfit_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        pytest.skip("matplotlib is not installed")

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    history = [
        {
            "epoch": 0,
            "train_eval_loss": 1.0,
            "train_ndcg@5": 0.2,
            "top1_wcag_pass_rate": 0.5,
            "color_distribution": [0.5, 0.5] + [0.0] * 30,
        },
        {
            "epoch": 1,
            "train_loss": 0.8,
            "train_eval_loss": 0.7,
            "train_ndcg@5": 0.4,
            "top1_wcag_pass_rate": 0.5,
            "color_distribution": [0.4, 0.6] + [0.0] * 30,
        },
    ]

    plot_paths = overfit_module.write_overfit_plots(
        history,
        loss_plot_path=tmp_path / "loss.png",
        ndcg_plot_path=tmp_path / "ndcg.png",
        color_plot_path=tmp_path / "colors.png",
    )

    assert set(plot_paths) == {"loss", "ndcg", "color_distribution"}
    for path in plot_paths.values():
        assert path.exists()
        assert path.stat().st_size > 0
