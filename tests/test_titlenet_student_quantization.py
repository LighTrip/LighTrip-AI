from __future__ import annotations

import math
import json
from typing import Any

import numpy as np
import pytest

from experiments.title_color_recommendation import (
    quantize_titlenet_student_onnx as quantize,
)
from experiments.title_color_recommendation import (
    summarize_titlenet_quantization_results as summarize,
)
from experiments.title_color_recommendation import (
    run_titlenet_static_int8_sweep as static_sweep,
)
from experiments.title_color_recommendation import (
    package_titlenet_qat_fp16_deployment as deployment,
)
from tests.title_color_experiment_helpers import tiny_classifier


torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")


class FakeDataset:
    def __getitem__(self, index: int) -> dict[str, Any]:
        x = torch.full((4, 36, 136), float(index + 1) / 10.0)
        target = torch.zeros(32, dtype=torch.float32)
        target[31] = 1.0
        return {
            "x": x,
            "target_distribution": target,
            "image_id": f"sample_{index}",
        }


class FixedSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def run(self, _output_names: list[str], _feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [self.output.copy()]


def test_calibration_reader_rewinds_arrays() -> None:
    first = np.zeros((1, 4, 36, 136), dtype=np.float32)
    second = np.ones((1, 4, 36, 136), dtype=np.float32)
    reader = quantize.NumpyCalibrationDataReader([first, second])

    assert reader.get_next() is not None
    assert reader.get_next() is not None
    assert reader.get_next() is None

    reader.rewind()

    restarted = reader.get_next()
    assert restarted is not None
    assert math.isclose(float(restarted["input"].sum()), 0.0, abs_tol=1e-12)


def test_quantized_validation_accepts_matching_top1_outputs() -> None:
    fp32_logits = np.arange(32, dtype=np.float32).reshape(1, 32)
    quantized_logits = fp32_logits + np.float32(0.001)
    top1 = np.array([31], dtype=np.int64)

    metrics = quantize.validate_quantized_pair(
        fp32_logits_session=FixedSession(fp32_logits),
        fp32_top1_session=FixedSession(top1),
        quantized_logits_session=FixedSession(quantized_logits),
        quantized_top1_session=FixedSession(top1),
        dataset=FakeDataset(),  # type: ignore[arg-type]
        indices=[0, 1],
        min_top1_agreement=1.0,
        max_ndcg5_drop=0.0,
    )

    assert metrics.passed
    assert math.isclose(metrics.top1_model_agreement or 0.0, 1.0, abs_tol=1e-12)
    assert math.isclose(metrics.logits_top1_agreement or 0.0, 1.0, abs_tol=1e-12)
    assert metrics.valid_top1_range


def test_int4_trial_is_reported_as_unsupported(tmp_path) -> None:  # type: ignore[no-untyped-def]
    trial = quantize.trial_from_name(quantize.TRIAL_INT4_WEIGHT_ONLY)

    logits_path, top1_path, reason = quantize.create_quantized_pair(
        trial=trial,
        logits_onnx=tmp_path / "logits.onnx",
        top1_onnx=tmp_path / "top1.onnx",
        output_dir=tmp_path,
        calibration_reader_factory=lambda: quantize.NumpyCalibrationDataReader([]),
    )

    assert logits_path is None
    assert top1_path is None
    assert reason is not None
    assert "INT4" in reason


def test_qat_wrapper_extracts_float_student_state() -> None:
    qat = pytest.importorskip(
        "experiments.title_color_recommendation.train_titlenet_student_qat"
    )
    student = tiny_classifier(nn)
    reference = tiny_classifier(nn)

    prepared = qat.prepare_qat_student(student, backend="qnnpack")
    state = qat.extract_float_student_state(
        prepared_student=prepared,
        reference_student=reference,
    )

    assert set(state) == set(reference.state_dict())
    assert all(not key.startswith("quant") for key in state)


def test_qat_defaults_use_best_student_checkpoint() -> None:
    qat = pytest.importorskip(
        "experiments.title_color_recommendation.train_titlenet_student_qat"
    )
    args = qat.parse_args(["--device", "cpu"])

    assert args.student_init_checkpoint == qat.DEFAULT_STUDENT_INIT_CHECKPOINT
    assert args.epochs == qat.DEFAULT_QAT_EPOCHS
    assert math.isclose(
        args.learning_rate,
        qat.DEFAULT_QAT_LEARNING_RATE,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def quantization_payload(*, top1: float, ndcg5_drop: float, latency: float) -> dict:
    return {
        "inputs": {
            "fp32_logits": "",
            "fp32_top1": "",
        },
        "results": [
            {
                "name": "fp16",
                "precision": "fp16",
                "method": "float16_conversion",
                "status": "passed",
                "logits_size_mb": 0.15,
                "top1_size_mb": 0.16,
                "latency": {
                    "batch1_logits": {"inference_time_ms": latency},
                },
                "reason": None,
                "validation": {
                    "fp32_ndcg_at_3": 0.981,
                    "fp32_ndcg_at_5": 0.991,
                    "top1_model_agreement": top1,
                    "quantized_ndcg_at_3": 0.98,
                    "quantized_ndcg_at_5": 0.99,
                    "ndcg_at_3_drop": 0.001,
                    "ndcg_at_5_drop": ndcg5_drop,
                    "sample_count": 10,
                },
            }
        ],
        "validation": {
            "sample_count": 10,
            "seed": 42,
            "split": "test",
        },
    }


def test_summary_builds_rows_with_fp32_and_quantized_results() -> None:
    baseline = {
        "reference_metrics": {
            "test_metrics": {
                "val_ndcg@3": 0.985,
                "val_ndcg@5": 0.988,
            }
        },
        "onnx": {
            "logits": {"size_mb": 0.29},
            "top1": {"size_mb": 0.30},
        },
        "latency": {
            "onnxruntime": {
                "batch1_logits": {"inference_time_ms": 0.7},
            }
        },
    }
    qat_training = {
        "best_epoch": 2,
        "test_metrics": {
            "val_ndcg@3": 0.984,
            "val_ndcg@5": 0.987,
        },
    }

    rows = summarize.build_rows(
        baseline_payload=baseline,
        ptq_payload=quantization_payload(top1=0.99, ndcg5_drop=0.001, latency=1.0),
        qat_training_payload=qat_training,
        qat_quantization_payload=quantization_payload(
            top1=1.0,
            ndcg5_drop=-0.001,
            latency=0.8,
        ),
    )

    assert [row.model for row in rows] == [
        "Student",
        "Student",
        "Student-QAT",
        "Student-QAT",
    ]
    assert rows[0].precision == "FP32"
    assert math.isclose(rows[0].ndcg_at_5 or 0.0, 0.991, abs_tol=1e-12)
    assert rows[1].method == "PTQ float16_conversion"
    assert rows[2].status == "reference"
    assert rows[3].method == "QAT+PTQ float16_conversion"


def test_summary_uses_mobile_proxy_latency_when_available() -> None:
    mobile_payload = {
        "results": [
            {
                "model_key": "student_kd_fp32",
                "thread_mode": "default",
                "latency": {"p50_ms": 0.7, "p95_ms": 0.9},
            },
            {
                "model_key": "student_kd_ptq_fp16",
                "thread_mode": "default",
                "latency": {"p50_ms": 0.6, "p95_ms": 0.8},
            },
        ]
    }
    rows = summarize.build_rows(
        baseline_payload={},
        ptq_payload=quantization_payload(top1=0.99, ndcg5_drop=0.001, latency=1.0),
        qat_training_payload={"best_epoch": 2},
        qat_quantization_payload=quantization_payload(
            top1=1.0,
            ndcg5_drop=-0.001,
            latency=0.8,
        ),
        mobile_payload=mobile_payload,
    )

    assert math.isclose(rows[0].latency_ms or 0.0, 0.7, abs_tol=1e-12)
    assert math.isclose(rows[0].latency_p95_ms or 0.0, 0.9, abs_tol=1e-12)
    assert math.isclose(rows[1].latency_ms or 0.0, 0.6, abs_tol=1e-12)
    assert math.isclose(rows[1].latency_p95_ms or 0.0, 0.8, abs_tol=1e-12)


def test_summary_candidates_prioritize_qat_fp16_when_it_is_best() -> None:
    rows = [
        summarize.ComparisonRow(
            model="Student",
            precision="FP16",
            method="PTQ float16_conversion",
            status="passed",
            ndcg_at_3=0.98,
            ndcg_at_5=0.99,
            ndcg_at_3_drop=0.001,
            ndcg_at_5_drop=0.001,
            top1_agreement=0.99,
            logits_size_mb=0.15,
            top1_size_mb=0.16,
            latency_ms=1.0,
            latency_p95_ms=1.2,
            eval_split="test",
            eval_sample_count=10,
            eval_seed=42,
            note="",
        ),
        summarize.ComparisonRow(
            model="Student-QAT",
            precision="FP16",
            method="QAT+PTQ float16_conversion",
            status="passed",
            ndcg_at_3=0.98,
            ndcg_at_5=0.99,
            ndcg_at_3_drop=0.0,
            ndcg_at_5_drop=-0.001,
            top1_agreement=1.0,
            logits_size_mb=0.15,
            top1_size_mb=0.16,
            latency_ms=0.8,
            latency_p95_ms=1.0,
            eval_split="test",
            eval_sample_count=10,
            eval_seed=42,
            note="",
        ),
    ]

    candidates = summarize.deployment_candidates(rows)

    assert candidates[0].model == "Student-QAT"
    assert candidates[0].precision == "FP16"


def test_static_sweep_selects_sensitive_node_presets() -> None:
    nodes = [
        ("/features/net/net.0/Conv", "Conv"),
        ("/features/net/net.5/attention/conv/Conv", "Conv"),
        ("/head/head.2/Gemm", "Gemm"),
        ("/head/head.5/Gemm", "Gemm"),
    ]

    assert static_sweep.select_excluded_node_names(
        nodes,
        preset="first_conv",
    ) == ["/features/net/net.0/Conv"]
    assert static_sweep.select_excluded_node_names(
        nodes,
        preset="head",
    ) == ["/head/head.2/Gemm", "/head/head.5/Gemm"]
    assert static_sweep.select_excluded_node_names(
        nodes,
        preset="final_gemm",
    ) == ["/head/head.5/Gemm"]
    assert static_sweep.select_excluded_node_names(
        nodes,
        preset="first_conv_head",
    ) == [
        "/features/net/net.0/Conv",
        "/head/head.2/Gemm",
        "/head/head.5/Gemm",
    ]


def test_static_sweep_trial_sets_include_focused_trials() -> None:
    base = static_sweep.trials_for_set("base")
    focused = static_sweep.trials_for_set("focused")
    all_trials = static_sweep.trials_for_set("all")

    assert base
    assert any(trial.exclude_preset == "final_gemm" for trial in focused)
    assert len(all_trials) == len(base) + len(focused)


def test_static_sweep_ranks_top1_before_latency() -> None:
    slow_but_accurate = static_sweep.StaticInt8Result(
        target="qat",
        trial="accurate",
        status="passed",
        calibration_split="val",
        calibration_sample_count=200,
        calibration_method="minmax",
        exclude_preset="none",
        excluded_nodes=[],
        per_channel=False,
        logits_path="accurate.onnx",
        top1_path="accurate_top1.onnx",
        logits_size_mb=0.2,
        top1_size_mb=0.2,
        latency_ms=2.0,
        top1_agreement=1.0,
        ndcg_at_3_drop=0.0,
        ndcg_at_5_drop=0.0,
        max_abs_diff=0.1,
    )
    fast_but_less_accurate = static_sweep.StaticInt8Result(
        target="qat",
        trial="fast",
        status="passed",
        calibration_split="val",
        calibration_sample_count=200,
        calibration_method="minmax",
        exclude_preset="none",
        excluded_nodes=[],
        per_channel=False,
        logits_path="fast.onnx",
        top1_path="fast_top1.onnx",
        logits_size_mb=0.1,
        top1_size_mb=0.1,
        latency_ms=0.5,
        top1_agreement=0.99,
        ndcg_at_3_drop=0.0,
        ndcg_at_5_drop=0.0,
        max_abs_diff=0.1,
    )

    ranked = static_sweep.best_results([fast_but_less_accurate, slow_but_accurate])

    assert ranked[0].trial == "accurate"


def test_deployment_palette_validation_requires_expected_ids(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(deployment, "PROJECT_ROOT", tmp_path)
    valid_palette = tmp_path / "palette.json"
    valid_palette.write_text(
        json.dumps([{"id": index} for index in range(32)]),
        encoding="utf-8",
    )
    invalid_palette = tmp_path / "invalid_palette.json"
    invalid_palette.write_text(
        json.dumps([{"id": index} for index in range(31)]),
        encoding="utf-8",
    )

    valid_ids, valid_count = deployment.validate_palette(valid_palette)
    invalid_ids, invalid_count = deployment.validate_palette(invalid_palette)

    assert valid_ids
    assert valid_count == 32
    assert not invalid_ids
    assert invalid_count == 31


def test_deployment_selects_fp16_quantization_result() -> None:
    payload = {
        "results": [
            {"name": "int8_dynamic", "status": "passed"},
            {"name": "fp16", "status": "passed", "validation": {"passed": True}},
        ]
    }

    result = deployment.fp16_quantization_result(payload)

    assert result["name"] == "fp16"


def test_deployment_format_metric_for_report() -> None:
    assert deployment.format_metric(None) == "-"
    assert deployment.format_metric("ok") == "ok"
    assert deployment.format_metric(0.1234567) == "0.123457"
