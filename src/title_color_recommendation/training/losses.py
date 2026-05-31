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


def distillation_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float,
) -> Tensor:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape: "
            f"student={tuple(student_logits.shape)}, "
            f"teacher={tuple(teacher_logits.shape)}"
        )
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive: {temperature}")

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature * temperature)


def combined_soft_label_distillation_loss(
    student_logits: Tensor,
    target_distribution: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float,
    base_loss_weight: float,
    distillation_loss_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if base_loss_weight < 0.0:
        raise ValueError(f"base_loss_weight must be non-negative: {base_loss_weight}")
    if distillation_loss_weight < 0.0:
        raise ValueError(
            "distillation_loss_weight must be non-negative: "
            f"{distillation_loss_weight}"
        )
    if base_loss_weight <= 0.0 and distillation_loss_weight <= 0.0:
        raise ValueError("at least one loss weight must be positive")

    base_loss = soft_label_kl_divergence(student_logits, target_distribution)
    distillation_loss = distillation_kl_divergence(
        student_logits,
        teacher_logits,
        temperature=temperature,
    )
    total_loss = (
        base_loss_weight * base_loss
        + distillation_loss_weight * distillation_loss
    )
    return total_loss, base_loss, distillation_loss
