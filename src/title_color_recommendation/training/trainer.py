from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from src.title_color_recommendation.training.config import TrainingConfig
from src.title_color_recommendation.training.losses import soft_label_kl_divergence
from src.title_color_recommendation.training.metrics import (
    ValidationMetrics,
    ndcg_at_k,
)

LOGGER = logging.getLogger(__name__)


def set_training_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> LRScheduler | ReduceLROnPlateau | None:
    scheduler_name = config.scheduler.strip().lower()
    if scheduler_name in {"", "none"}:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(config.epochs, 1),
        )
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved


def _required_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a Tensor")
    return value


def _weighted_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Optimizer,
    *,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for batch in dataloader:
        moved_batch = _move_batch_to_device(batch, device)
        x = _required_tensor(moved_batch, "x")
        target_distribution = _required_tensor(moved_batch, "target_distribution")

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = soft_label_kl_divergence(logits, target_distribution)
        loss.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size

    return _weighted_average(total_loss, sample_count)


def validate(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    *,
    device: torch.device,
    num_classes: int,
) -> ValidationMetrics:
    model.eval()
    total_loss = 0.0
    total_ndcg = 0.0
    total_wcag = 0.0
    sample_count = 0
    color_counts = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for batch in dataloader:
            moved_batch = _move_batch_to_device(batch, device)
            x = _required_tensor(moved_batch, "x")
            target_distribution = _required_tensor(
                moved_batch,
                "target_distribution",
            )
            wcag_pass = _required_tensor(moved_batch, "wcag_pass")

            logits = model(x)
            loss = soft_label_kl_divergence(logits, target_distribution)
            batch_size = int(x.shape[0])

            ndcg = ndcg_at_k(logits, target_distribution, k=5)
            top1 = logits.argmax(dim=-1, keepdim=True)
            wcag_values = wcag_pass.gather(dim=-1, index=top1).float()

            total_loss += float(loss.item()) * batch_size
            total_ndcg += float(ndcg.sum().item())
            total_wcag += float(wcag_values.sum().item())
            sample_count += batch_size

            batch_counts = torch.bincount(
                top1.squeeze(dim=-1).detach().cpu(),
                minlength=num_classes,
            ).double()
            color_counts += batch_counts

    total_predictions = float(color_counts.sum().item())
    if total_predictions <= 0:
        distribution = [0.0 for _index in range(num_classes)]
    else:
        distribution = (color_counts / total_predictions).tolist()

    return ValidationMetrics(
        val_loss=_weighted_average(total_loss, sample_count),
        val_ndcg_at_5=_weighted_average(total_ndcg, sample_count),
        top1_wcag_pass_rate=_weighted_average(total_wcag, sample_count),
        color_distribution=distribution,
    )


def _scheduler_step(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    val_loss: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
        return
    scheduler.step()


def _scheduler_state_dict(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    return scheduler.state_dict()


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    epoch: int,
    config: TrainingConfig,
    metrics: Mapping[str, Any],
    best_metric_value: float,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": _scheduler_state_dict(scheduler),
            "config": config.as_dict(),
            "metrics": dict(metrics),
            "best_metric": config.best_metric,
            "best_metric_value": best_metric_value,
        },
        checkpoint_path,
    )


def append_jsonl_log(path: str | Path, record: Mapping[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        json.dump(dict(record), f, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def _metric_is_better(
    *,
    candidate: float,
    best: float | None,
    metric_name: str,
) -> bool:
    if best is None:
        return True
    if metric_name.endswith("loss"):
        return candidate < best
    return candidate > best


def fit(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Any]],
    val_loader: Iterable[Mapping[str, Any]],
    config: TrainingConfig,
    *,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | ReduceLROnPlateau | None = None,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    target_logger = LOGGER if logger is None else logger
    set_training_seed(config.seed)
    device = resolve_device(config.device)
    model.to(device)

    active_optimizer = optimizer
    if active_optimizer is None:
        active_optimizer = create_optimizer(model, config)

    active_scheduler = scheduler
    if active_scheduler is None:
        active_scheduler = create_scheduler(active_optimizer, config)

    checkpoint_dir = Path(config.checkpoint_dir)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"
    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            active_optimizer,
            device=device,
        )
        validation = validate(
            model,
            val_loader,
            device=device,
            num_classes=config.num_classes,
        )
        _scheduler_step(active_scheduler, validation.val_loss)

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **validation.as_dict(),
        }
        history.append(record)
        append_jsonl_log(config.log_path, record)

        if config.best_metric not in record:
            raise ValueError(f"best_metric not found in metrics: {config.best_metric}")
        metric_value = float(record[config.best_metric])
        is_best = _metric_is_better(
            candidate=metric_value,
            best=best_metric_value,
            metric_name=config.best_metric,
        )
        if is_best:
            best_metric_value = metric_value

        checkpoint_best_value = (
            metric_value if best_metric_value is None else best_metric_value
        )
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=active_optimizer,
            scheduler=active_scheduler,
            epoch=epoch,
            config=config,
            metrics=record,
            best_metric_value=checkpoint_best_value,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=active_optimizer,
                scheduler=active_scheduler,
                epoch=epoch,
                config=config,
                metrics=record,
                best_metric_value=checkpoint_best_value,
            )

        target_logger.info(
            "epoch=%s train_loss=%.6f val_loss=%.6f val_ndcg@5=%.6f "
            "top1_wcag_pass_rate=%.6f",
            epoch,
            train_loss,
            validation.val_loss,
            validation.val_ndcg_at_5,
            validation.top1_wcag_pass_rate,
        )

    return history
