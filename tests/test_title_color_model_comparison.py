from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def comparison_module() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("matplotlib")
    return pytest.importorskip(
        "experiments.title_color_recommendation.run_model_comparison"
    )


def test_model_comparison_writes_summary_report_without_training(
    comparison_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    config_path = tmp_path / "model_comparison.yaml"
    config_path.write_text(
        "\n".join(
            [
                "models:",
                "  - simple_cnn",
                "  - simple_cnn_m",
                "training:",
                "  device: cpu",
                "  dropout: 0.3",
                "  weight_init: small_head",
                "  activation: hardswish",
                "latency:",
                "  batch_size: 1",
                "  warmup_steps: 0",
                "  benchmark_steps: 1",
            ]
        ),
        encoding="utf-8",
    )

    args = comparison_module.parse_args(
        [
            "--config",
            str(config_path),
            "--results-csv",
            str(tmp_path / "model_comparison_results.csv"),
            "--report-path",
            str(tmp_path / "model_comparison_report.md"),
            "--latency-plot-path",
            str(tmp_path / "model_comparison_latency.png"),
            "--device",
            "cpu",
        ]
    )

    rows = comparison_module.run(args)

    assert [row["model_name"] for row in rows] == ["simple_cnn", "simple_cnn_m"]
    assert all(row["trained"] is False for row in rows)
    assert (tmp_path / "model_comparison_report.md").exists()
    assert (tmp_path / "model_comparison_latency.png").exists()

    with (tmp_path / "model_comparison_results.csv").open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        records = list(csv.DictReader(file))
    assert len(records) == 2
    assert records[0]["model_name"] == "simple_cnn"
    assert records[0]["weight_init"] == "small_head"
    assert records[0]["activation"] == "hardswish"
