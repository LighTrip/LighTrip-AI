from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def soft_label_kl_divergence(
    logits: Tensor,
    target_distribution: Tensor,
) -> Tensor:
    if logits.shape != target_distribution.shape:
        raise ValueError(
            "logits and target_distribution must have the same shape: "
            f"logits={tuple(logits.shape)}, target={tuple(target_distribution.shape)}"
        )
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(
        log_probs,
        target_distribution,
        reduction="batchmean",
    )
