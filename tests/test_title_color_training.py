from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def nn_module() -> Any:
    return pytest.importorskip("torch.nn")


@pytest.fixture()
def training_modules() -> dict[str, Any]:
    return {
        "config": pytest.importorskip("src.title_color_recommendation.training.config"),
        "losses": pytest.importorskip("src.title_color_recommendation.training.losses"),
        "metrics": pytest.importorskip(
            "src.title_color_recommendation.training.metrics"
        ),
        "trainer": pytest.importorskip(
            "src.title_color_recommendation.training.trainer"
        ),
    }


class TinyTitleColorDataset:
    def __init__(self, torch_module: Any, *, length: int = 6) -> None:
        self.torch = torch_module
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self.torch.full((4, 4, 4), float(index + 1) / 10.0)
        target = self.torch.zeros(32, dtype=self.torch.float32)
        target[index % 32] = 0.7
        target[(index + 1) % 32] = 0.3
        pseudo_scores = self.torch.linspace(0.0, 1.0, steps=32)
        wcag_pass = self.torch.zeros(32, dtype=self.torch.float32)
        wcag_pass[::2] = 1.0
        return {
            "x": x,
            "target_distribution": target,
            "pseudo_scores": pseudo_scores,
            "wcag_pass": wcag_pass,
            "image_id": f"sample_{index}",
        }


def _tiny_classifier(nn_module: Any) -> Any:
    return nn_module.Sequential(
        nn_module.Flatten(),
        nn_module.Linear(4 * 4 * 4, 32),
    )


def _loader(torch_module: Any, dataset: TinyTitleColorDataset, batch_size: int) -> Any:
    data_module = pytest.importorskip("torch.utils.data")
    generator = torch_module.Generator()
    generator.manual_seed(7)
    return data_module.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=generator,
    )


def test_soft_label_kl_divergence_matches_expected_shape(
    torch_module: Any,
    training_modules: dict[str, Any],
) -> None:
    logits = torch_module.zeros((2, 32), dtype=torch_module.float32)
    target = torch_module.full((2, 32), 1.0 / 32.0, dtype=torch_module.float32)

    loss = training_modules["losses"].soft_label_kl_divergence(logits, target)

    assert tuple(loss.shape) == ()
    assert math.isclose(float(loss.item()), 0.0, rel_tol=0.0, abs_tol=1e-6)


def test_validation_metrics_compute_expected_shapes(
    torch_module: Any,
    training_modules: dict[str, Any],
) -> None:
    logits = torch_module.eye(32, dtype=torch_module.float32)[:2]
    target = torch_module.eye(32, dtype=torch_module.float32)[:2]
    wcag_pass = torch_module.zeros((2, 32), dtype=torch_module.float32)
    wcag_pass[:, 0] = 1.0

    ndcg = training_modules["metrics"].mean_ndcg_at_k(logits, target, k=5)
    pass_rate = training_modules["metrics"].top1_wcag_pass_rate(logits, wcag_pass)
    topk_logits = torch_module.tensor(
        [
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
            [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        ],
        dtype=torch_module.float32,
    )
    topk_wcag = torch_module.zeros((2, 6), dtype=torch_module.float32)
    topk_wcag[0, 4] = 1.0
    topk_wcag[1, 5] = 1.0
    top5_pass_rate = training_modules["metrics"].top5_any_wcag_pass_rate(
        topk_logits,
        topk_wcag,
    )
    distribution = training_modules["metrics"].color_distribution(
        logits,
        num_classes=32,
    )

    assert math.isclose(ndcg, 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(pass_rate, 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(top5_pass_rate, 0.5, rel_tol=0.0, abs_tol=1e-6)
    assert len(distribution) == 32
    assert math.isclose(sum(distribution), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_training_config_loads_from_json(
    tmp_path: Path,
    training_modules: dict[str, Any],
) -> None:
    config_path = tmp_path / "training_config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch_size": 8,
                "epochs": 3,
                "learning_rate": 0.001,
                "weight_decay": 0.01,
                "scheduler": "none",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    config = training_modules["config"].load_training_config(config_path)

    assert config.batch_size == 8
    assert config.epochs == 3
    assert math.isclose(config.learning_rate, 0.001, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(config.weight_decay, 0.01, rel_tol=0.0, abs_tol=1e-12)
    assert config.scheduler == "none"
    assert config.device == "cpu"


def test_fit_runs_one_epoch_and_writes_checkpoints(
    tmp_path: Path,
    torch_module: Any,
    nn_module: Any,
    training_modules: dict[str, Any],
) -> None:
    train_dataset = TinyTitleColorDataset(torch_module, length=6)
    val_dataset = TinyTitleColorDataset(torch_module, length=4)
    train_loader = _loader(torch_module, train_dataset, batch_size=2)
    val_loader = _loader(torch_module, val_dataset, batch_size=2)
    model = _tiny_classifier(nn_module)
    config = training_modules["config"].TrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        num_workers=0,
        device="cpu",
        scheduler="none",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_path=str(tmp_path / "train_log.jsonl"),
        seed=123,
    )

    history = training_modules["trainer"].fit(
        model,
        train_loader,
        val_loader,
        config,
    )

    latest_path = tmp_path / "checkpoints" / "checkpoint_latest.pt"
    best_path = tmp_path / "checkpoints" / "checkpoint_best.pt"
    assert len(history) == 1
    assert math.isfinite(history[0]["train_loss"])
    assert math.isfinite(history[0]["val_loss"])
    assert latest_path.exists()
    assert best_path.exists()

    log_lines = (tmp_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    log_record = json.loads(log_lines[0])
    assert log_record["epoch"] == 1
    assert "train_loss" in log_record
    assert "val_loss" in log_record
    assert "val_ndcg@3" in log_record
    assert "val_ndcg@5" in log_record
    assert "top1_wcag_pass_rate" in log_record
    assert "top5_any_wcag_pass_rate" in log_record
    assert len(log_record["color_distribution"]) == 32
