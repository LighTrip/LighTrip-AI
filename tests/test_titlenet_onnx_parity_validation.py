from __future__ import annotations

import numpy as np
import pytest

from experiments.title_color_recommendation import validate_titlenet_onnx


def test_selected_indices_are_deterministic_and_sorted() -> None:
    first = validate_titlenet_onnx.selected_indices(
        dataset_size=20,
        sample_count=5,
        seed=7,
    )
    second = validate_titlenet_onnx.selected_indices(
        dataset_size=20,
        sample_count=5,
        seed=7,
    )

    assert first == second
    assert first == sorted(first)
    assert len(first) == 5


def test_topk_indices_returns_descending_order() -> None:
    values = np.asarray([[0.1, 0.5, 0.3, 0.2]])

    assert validate_titlenet_onnx.topk_indices(values, k=3) == [1, 2, 3]


def test_summarize_results_marks_threshold_failures() -> None:
    results = [
        validate_titlenet_onnx.SampleParityResult(
            image_id="sample_ok",
            pytorch_top1=1,
            onnx_logits_top1=1,
            onnx_top1=1,
            top3_match=True,
            top5_match=True,
            max_abs_diff=1e-6,
            mean_abs_diff=1e-7,
        ),
        validate_titlenet_onnx.SampleParityResult(
            image_id="sample_bad",
            pytorch_top1=2,
            onnx_logits_top1=2,
            onnx_top1=2,
            top3_match=True,
            top5_match=True,
            max_abs_diff=2e-4,
            mean_abs_diff=1e-7,
        ),
    ]

    metrics = validate_titlenet_onnx.summarize_results(
        split="test",
        seed=42,
        results=results,
        max_abs_diff_threshold=1e-4,
        mean_abs_diff_threshold=1e-5,
    )

    assert metrics.top1_agreement == pytest.approx(1.0)
    assert metrics.failure_count == 1
    assert not metrics.passed
