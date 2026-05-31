from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.title_color_experiment_helpers import tiny_classifier, tiny_split_loaders


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def nn_module() -> Any:
    return pytest.importorskip("torch.nn")


@pytest.fixture()
def distillation_module() -> Any:
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    return pytest.importorskip(
        "experiments.title_color_recommendation.train_titlenet_student_distillation"
    )


def test_student_distillation_config_loads_nested_yaml(
    distillation_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(distillation_module, "PROJECT_ROOT", tmp_path)
    data_root = tmp_path / "data" / "title_color_recommendation"
    checkpoint_path = tmp_path / "outputs" / "teacher.pt"
    student_init_path = tmp_path / "outputs" / "student_only.pt"
    data_root.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"placeholder")
    student_init_path.write_bytes(b"placeholder")
    config_path = tmp_path / "titlenet_student_distillation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "teacher:",
                "  model_name: titlenet",
                "  checkpoint_path: outputs/teacher.pt",
                "  activation: gelu",
                "student:",
                "  model_name: titlenet_student",
                "  activation: hardswish",
                "  init_checkpoint_path: outputs/student_only.pt",
                "training:",
                "  epochs: 3",
                "  batch_size: 8",
                "  device: cpu",
                "data:",
                "  data_root: data/title_color_recommendation",
                "distillation:",
                "  temperature: 3.0",
                "  base_loss_weight: 0.4",
                "  distillation_loss_weight: 0.6",
            ]
        ),
        encoding="utf-8",
    )

    args = distillation_module.parse_args(["--config", str(config_path)])
    config = distillation_module.load_student_distillation_config(args)

    assert config.teacher.model_name == "titlenet"
    assert config.student.model_name == "titlenet_student"
    assert config.student.activation == "hardswish"
    assert config.training.epochs == 3
    assert config.training.batch_size == 8
    assert config.training.device == "cpu"
    assert math.isclose(config.loss.temperature, 3.0, rel_tol=0.0, abs_tol=1e-12)
    assert config.student_init_checkpoint == student_init_path


def test_run_distillation_loop_writes_checkpoints_and_log(
    distillation_module: Any,
    torch_module: Any,
    nn_module: Any,
    tmp_path: Path,
) -> None:
    loaders = tiny_split_loaders(torch_module)
    teacher = tiny_classifier(nn_module)
    student = tiny_classifier(nn_module)
    training = distillation_module.TrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        device="cpu",
        scheduler="none",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "distillation.jsonl"),
        best_metric="val_ndcg@5",
        seed=123,
    )
    loss_config = distillation_module.DistillationLossConfig(
        temperature=2.0,
        base_loss_weight=0.5,
        distillation_loss_weight=0.5,
    )

    history, best_epoch, best_value, best_state = (
        distillation_module.run_distillation_loop(
            student=student,
            teacher=teacher,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            training=training,
            loss_config=loss_config,
        )
    )

    assert len(history) == 1
    assert best_epoch == 1
    assert math.isfinite(best_value)
    assert best_state
    assert (tmp_path / "checkpoints" / "checkpoint_best.pt").exists()
    assert (tmp_path / "checkpoints" / "checkpoint_latest.pt").exists()

    records = [
        json.loads(line)
        for line in (tmp_path / "distillation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1
    assert "train_base_loss" in records[0]
    assert "train_distillation_loss" in records[0]
    assert "teacher_top1_agreement" in records[0]


def test_teacher_agreement_returns_overlap_metrics(
    distillation_module: Any,
    torch_module: Any,
    nn_module: Any,
) -> None:
    loaders = tiny_split_loaders(torch_module)
    teacher = tiny_classifier(nn_module)
    student = tiny_classifier(nn_module)
    student.load_state_dict(teacher.state_dict())

    metrics = distillation_module.teacher_agreement(
        student,
        teacher,
        loaders["test"],
        device=torch_module.device("cpu"),
    )

    assert math.isclose(
        metrics["teacher_top1_agreement"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        metrics["teacher_top3_overlap"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert math.isclose(
        metrics["teacher_top5_overlap"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
