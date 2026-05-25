from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ValidationMetrics:
    val_loss: float
    val_ndcg_at_3: float
    val_ndcg_at_5: float
    top1_wcag_pass_rate: float
    top5_any_wcag_pass_rate: float
    color_distribution: list[float]

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "val_loss": self.val_loss,
            "val_ndcg@3": self.val_ndcg_at_3,
            "val_ndcg@5": self.val_ndcg_at_5,
            "top1_wcag_pass_rate": self.top1_wcag_pass_rate,
            "top5_any_wcag_pass_rate": self.top5_any_wcag_pass_rate,
            "color_distribution": self.color_distribution,
        }


def ndcg_at_k(
    logits: Tensor,
    target_distribution: Tensor,
    *,
    k: int = 5,
) -> Tensor:
    if logits.shape != target_distribution.shape:
        raise ValueError(
            "logits and target_distribution must have the same shape: "
            f"logits={tuple(logits.shape)}, target={tuple(target_distribution.shape)}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive: {k}")

    num_classes = logits.shape[-1]
    top_k = min(k, num_classes)
    prediction_indices = logits.topk(top_k, dim=-1).indices
    prediction_gains = target_distribution.gather(dim=-1, index=prediction_indices)
    ideal_gains = target_distribution.topk(top_k, dim=-1).values

    discounts = torch.log2(
        torch.arange(
            2,
            top_k + 2,
            device=logits.device,
            dtype=target_distribution.dtype,
        )
    )
    dcg = (prediction_gains / discounts).sum(dim=-1)
    idcg = (ideal_gains / discounts).sum(dim=-1)
    return torch.where(idcg > 0, dcg / idcg, torch.zeros_like(dcg))


def mean_ndcg_at_k(
    logits: Tensor,
    target_distribution: Tensor,
    *,
    k: int = 5,
) -> float:
    ndcg = ndcg_at_k(logits, target_distribution, k=k)
    return float(ndcg.mean().item())


def top1_wcag_pass_rate(logits: Tensor, wcag_pass: Tensor) -> float:
    if logits.shape != wcag_pass.shape:
        raise ValueError(
            "logits and wcag_pass must have the same shape: "
            f"logits={tuple(logits.shape)}, wcag={tuple(wcag_pass.shape)}"
        )
    top1 = logits.argmax(dim=-1, keepdim=True)
    pass_values = wcag_pass.gather(dim=-1, index=top1)
    return float(pass_values.float().mean().item())


def topk_any_wcag_pass_rate(logits: Tensor, wcag_pass: Tensor, *, k: int) -> float:
    if logits.shape != wcag_pass.shape:
        raise ValueError(
            "logits and wcag_pass must have the same shape: "
            f"logits={tuple(logits.shape)}, wcag={tuple(wcag_pass.shape)}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive: {k}")

    top_k = min(k, logits.shape[-1])
    indices = logits.topk(top_k, dim=-1).indices
    pass_values = wcag_pass.gather(dim=-1, index=indices)
    return float(pass_values.bool().any(dim=-1).float().mean().item())


def top5_any_wcag_pass_rate(logits: Tensor, wcag_pass: Tensor) -> float:
    return topk_any_wcag_pass_rate(logits, wcag_pass, k=5)


def color_distribution(
    logits: Tensor,
    *,
    num_classes: int,
) -> list[float]:
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive: {num_classes}")
    top1 = logits.argmax(dim=-1)
    counts = torch.bincount(top1, minlength=num_classes).float()
    total = float(counts.sum().item())
    if math.isclose(total, 0.0, rel_tol=0.0, abs_tol=0.0):
        return [0.0 for _index in range(num_classes)]
    return (counts / total).cpu().tolist()
