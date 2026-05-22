"""Training utilities for title color recommendation."""

from src.title_color_recommendation.training.config import (
    TrainingConfig,
    load_training_config,
    training_config_from_mapping,
)
from src.title_color_recommendation.training.losses import soft_label_kl_divergence
from src.title_color_recommendation.training.metrics import (
    ValidationMetrics,
    color_distribution,
    mean_ndcg_at_k,
    ndcg_at_k,
    top1_wcag_pass_rate,
)
from src.title_color_recommendation.training.trainer import (
    append_jsonl_log,
    create_optimizer,
    create_scheduler,
    fit,
    save_checkpoint,
    train_one_epoch,
    validate,
)

__all__ = [
    "TrainingConfig",
    "ValidationMetrics",
    "append_jsonl_log",
    "color_distribution",
    "create_optimizer",
    "create_scheduler",
    "fit",
    "load_training_config",
    "mean_ndcg_at_k",
    "ndcg_at_k",
    "save_checkpoint",
    "soft_label_kl_divergence",
    "top1_wcag_pass_rate",
    "train_one_epoch",
    "training_config_from_mapping",
    "validate",
]
