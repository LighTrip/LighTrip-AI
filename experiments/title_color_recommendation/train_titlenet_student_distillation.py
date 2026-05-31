from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.evaluate_model_checkpoints import (
    load_checkpoint,
)
from experiments.title_color_recommendation.run_model_comparison import (
    measure_latency,
    model_size_mb,
)
from src.models.fixed_palette_classifier import (
    count_total_parameters,
    count_trainable_parameters,
)
from src.models.title_color_model_registry import build_title_color_model
from src.title_color_recommendation.data.dataloader import (
    create_title_color_dataloaders,
)
from src.title_color_recommendation.training.config import (
    TrainingConfig,
    training_config_from_mapping,
)
from src.title_color_recommendation.training.losses import (
    combined_soft_label_distillation_loss,
)
from src.title_color_recommendation.training.metrics import ValidationMetrics
from src.title_color_recommendation.training.trainer import (
    append_jsonl_log,
    create_optimizer,
    create_scheduler,
    resolve_device,
    save_checkpoint,
    set_training_seed,
    validate,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(
    "configs/title_color_recommendation/titlenet_student_distillation.yaml"
)
DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_TEACHER_CHECKPOINT = Path(
    "outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt"
)
DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints/titlenet_student_distillation")
DEFAULT_LOG_PATH = Path("outputs/logs/titlenet_student_distillation.jsonl")
DEFAULT_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_distillation_report.md"
)
DEFAULT_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_distillation_metrics.json"
)
TRAIN_LOSS_KEY = "train_loss"
TRAIN_BASE_LOSS_KEY = "train_base_loss"
TRAIN_DISTILLATION_LOSS_KEY = "train_distillation_loss"
VAL_LOSS_KEY = "val_loss"
VAL_NDCG_AT_3_KEY = "val_ndcg@3"
VAL_NDCG_AT_5_KEY = "val_ndcg@5"
TEACHER_TOP1_AGREEMENT_KEY = "teacher_top1_agreement"
TEACHER_TOP3_OVERLAP_KEY = "teacher_top3_overlap"
TEACHER_TOP5_OVERLAP_KEY = "teacher_top5_overlap"


@dataclass(frozen=True)
class ModelBuildConfig:
    model_name: str
    dropout: float
    weight_init: str
    activation: str


@dataclass(frozen=True)
class DistillationLossConfig:
    temperature: float
    base_loss_weight: float
    distillation_loss_weight: float


@dataclass(frozen=True)
class LatencyConfig:
    warmup_steps: int
    benchmark_steps: int


@dataclass(frozen=True)
class StudentDistillationConfig:
    teacher: ModelBuildConfig
    student: ModelBuildConfig
    training: TrainingConfig
    loss: DistillationLossConfig
    latency: LatencyConfig
    data_root: Path
    labels_matrix: Path | None
    labels_soft: Path | None
    teacher_checkpoint: Path
    student_init_checkpoint: Path | None
    report_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class StudentDistillationResult:
    history: list[dict[str, Any]]
    test_metrics: ValidationMetrics
    test_agreement: dict[str, float]
    best_epoch: int
    best_metric_value: float
    dataset_sizes: dict[str, int]
    checkpoint_paths: dict[str, Path]
    report_path: Path
    metrics_path: Path
    metrics_payload: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an ablation-guided TitLeNet student with distillation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--student-init-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--best-metric", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--base-loss-weight", type=float, default=None)
    parser.add_argument("--distillation-loss-weight", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--metrics-path", type=Path, default=None)
    return parser.parse_args(argv)


def _require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    return value


def _load_yaml_config(path: Path) -> Mapping[str, Any]:
    config_path = resolve_inside_project(
        PROJECT_ROOT,
        path,
        must_exist=True,
        description="student distillation config",
    )
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return _require_mapping(payload, description="student distillation config")


def _model_config(section: Mapping[str, Any], *, default_name: str) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_name=str(section.get("model_name", default_name)),
        dropout=float(section.get("dropout", 0.2)),
        weight_init=str(section.get("weight_init", "small_head")),
        activation=str(section.get("activation", "gelu")),
    )


def _path_from_section(
    section: Mapping[str, Any],
    key: str,
    default: Path,
    *,
    override: Path | None = None,
    must_exist: bool = False,
) -> Path:
    value = override if override is not None else section.get(key, default)
    return resolve_inside_project(
        PROJECT_ROOT,
        Path(value),
        must_exist=must_exist,
        description=key,
    )


def _optional_path(
    value: str | Path | None,
    *,
    override: Path | None,
    description: str,
) -> Path | None:
    selected = override if override is not None else value
    if selected in {None, ""}:
        return None
    return resolve_inside_project(
        PROJECT_ROOT,
        Path(selected),
        must_exist=True,
        description=description,
    )


def _training_overrides(args: argparse.Namespace) -> dict[str, Any]:
    pairs = (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("weight_decay", "weight_decay"),
        ("num_workers", "num_workers"),
        ("device", "device"),
        ("scheduler", "scheduler"),
        ("best_metric", "best_metric"),
        ("seed", "seed"),
    )
    return {
        config_name: value
        for arg_name, config_name in pairs
        for value in (getattr(args, arg_name),)
        if value is not None
    }


def load_student_distillation_config(
    args: argparse.Namespace,
) -> StudentDistillationConfig:
    payload = _load_yaml_config(args.config)
    teacher_section = _require_mapping(payload.get("teacher"), description="teacher")
    student_section = _require_mapping(payload.get("student"), description="student")
    training_section = dict(
        _require_mapping(payload.get("training"), description="training")
    )
    data_section = _require_mapping(payload.get("data"), description="data")
    output_section = _require_mapping(payload.get("outputs"), description="outputs")
    distillation_section = _require_mapping(
        payload.get("distillation"),
        description="distillation",
    )
    latency_section = _require_mapping(payload.get("latency"), description="latency")

    checkpoint_dir = _path_from_section(
        output_section,
        "checkpoint_dir",
        DEFAULT_CHECKPOINT_DIR,
        override=args.checkpoint_dir,
    )
    log_path = _path_from_section(
        output_section,
        "log_path",
        DEFAULT_LOG_PATH,
        override=args.log_path,
    )
    training_values = {
        "batch_size": 64,
        "epochs": 20,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "num_classes": 32,
        "num_workers": 4,
        "device": "cuda",
        "scheduler": "cosine",
        "best_metric": VAL_NDCG_AT_5_KEY,
        "seed": 42,
        "model_name": str(student_section.get("model_name", "titlenet_student")),
        "dropout": float(student_section.get("dropout", 0.2)),
        "weight_init": str(student_section.get("weight_init", "small_head")),
        "activation": str(student_section.get("activation", "hardswish")),
        "checkpoint_dir": str(checkpoint_dir),
        "log_path": str(log_path),
        **training_section,
        **_training_overrides(args),
    }
    training = training_config_from_mapping(training_values)

    temperature = float(distillation_section.get("temperature", 2.0))
    base_loss_weight = float(
        args.base_loss_weight
        if args.base_loss_weight is not None
        else distillation_section.get("base_loss_weight", 0.5)
    )
    distillation_loss_weight = float(
        args.distillation_loss_weight
        if args.distillation_loss_weight is not None
        else distillation_section.get("distillation_loss_weight", 0.5)
    )
    if args.temperature is not None:
        temperature = float(args.temperature)

    data_root = _path_from_section(
        data_section,
        "data_root",
        DEFAULT_DATA_ROOT,
        override=args.data_root,
        must_exist=True,
    )
    labels_matrix = _optional_path(
        data_section.get("labels_matrix"),
        override=args.labels_matrix,
        description="labels_matrix",
    )
    labels_soft = _optional_path(
        data_section.get("labels_soft"),
        override=args.labels_soft,
        description="labels_soft",
    )

    return StudentDistillationConfig(
        teacher=_model_config(teacher_section, default_name="titlenet"),
        student=_model_config(student_section, default_name="titlenet_student"),
        training=training,
        loss=DistillationLossConfig(
            temperature=temperature,
            base_loss_weight=base_loss_weight,
            distillation_loss_weight=distillation_loss_weight,
        ),
        latency=LatencyConfig(
            warmup_steps=int(latency_section.get("warmup_steps", 10)),
            benchmark_steps=int(latency_section.get("benchmark_steps", 50)),
        ),
        data_root=data_root,
        labels_matrix=labels_matrix,
        labels_soft=labels_soft,
        teacher_checkpoint=_path_from_section(
            teacher_section,
            "checkpoint_path",
            DEFAULT_TEACHER_CHECKPOINT,
            override=args.teacher_checkpoint,
            must_exist=True,
        ),
        student_init_checkpoint=_optional_path(
            student_section.get("init_checkpoint_path"),
            override=args.student_init_checkpoint,
            description="student_init_checkpoint",
        ),
        report_path=_path_from_section(
            output_section,
            "report_path",
            DEFAULT_REPORT_PATH,
            override=args.report_path,
        ),
        metrics_path=_path_from_section(
            output_section,
            "metrics_path",
            DEFAULT_METRICS_PATH,
            override=args.metrics_path,
        ),
    )


def validate_distillation_config(config: StudentDistillationConfig) -> None:
    if config.loss.temperature <= 0.0:
        raise ValueError(f"temperature must be positive: {config.loss.temperature}")
    if config.loss.base_loss_weight < 0.0:
        raise ValueError(
            f"base_loss_weight must be non-negative: {config.loss.base_loss_weight}"
        )
    if config.loss.distillation_loss_weight < 0.0:
        raise ValueError(
            "distillation_loss_weight must be non-negative: "
            f"{config.loss.distillation_loss_weight}"
        )
    if (
        config.loss.base_loss_weight <= 0.0
        and config.loss.distillation_loss_weight <= 0.0
    ):
        raise ValueError("at least one loss weight must be positive")
    if config.latency.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if config.latency.benchmark_steps <= 0:
        raise ValueError("benchmark_steps must be positive")


def build_model(
    model_config: ModelBuildConfig,
    *,
    num_classes: int,
    checkpoint_path: Path | None = None,
) -> nn.Module:
    model = build_title_color_model(
        model_config.model_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=model_config.dropout,
        weight_init=model_config.weight_init,
        activation=model_config.activation,
    )
    if checkpoint_path is not None:
        checkpoint = load_checkpoint(checkpoint_path)
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError("checkpoint model_state_dict must be a mapping")
        model.load_state_dict(state_dict)
    return model


def _move_batch_to_device(
    batch: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _required_tensor(batch: Mapping[str, Any], key: str) -> Tensor:
    value = batch.get(key)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] must be a Tensor")
    return value


def _weighted_average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def train_distillation_epoch(
    *,
    student: nn.Module,
    teacher: nn.Module,
    dataloader: Any,
    optimizer: Optimizer,
    device: torch.device,
    loss_config: DistillationLossConfig,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_base_loss = 0.0
    total_distillation_loss = 0.0
    sample_count = 0

    for batch in dataloader:
        moved_batch = _move_batch_to_device(batch, device)
        x = _required_tensor(moved_batch, "x")
        target_distribution = _required_tensor(moved_batch, "target_distribution")

        with torch.no_grad():
            teacher_logits = teacher(x)

        optimizer.zero_grad(set_to_none=True)
        student_logits = student(x)
        loss, base_loss, distillation_loss = combined_soft_label_distillation_loss(
            student_logits,
            target_distribution,
            teacher_logits,
            temperature=loss_config.temperature,
            base_loss_weight=loss_config.base_loss_weight,
            distillation_loss_weight=loss_config.distillation_loss_weight,
        )
        loss.backward()
        optimizer.step()

        batch_size = int(x.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_base_loss += float(base_loss.detach().item()) * batch_size
        total_distillation_loss += float(distillation_loss.detach().item()) * batch_size
        sample_count += batch_size

    return {
        TRAIN_LOSS_KEY: _weighted_average(total_loss, sample_count),
        TRAIN_BASE_LOSS_KEY: _weighted_average(total_base_loss, sample_count),
        TRAIN_DISTILLATION_LOSS_KEY: _weighted_average(
            total_distillation_loss,
            sample_count,
        ),
    }


def teacher_agreement(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    student.eval()
    teacher.eval()
    sample_count = 0
    top1_matches = 0.0
    top3_overlaps = 0.0
    top5_overlaps = 0.0

    with torch.no_grad():
        for batch in dataloader:
            moved_batch = _move_batch_to_device(batch, device)
            x = _required_tensor(moved_batch, "x")
            student_logits = student(x)
            teacher_logits = teacher(x)
            batch_size = int(x.shape[0])

            student_top1 = student_logits.argmax(dim=-1)
            teacher_top1 = teacher_logits.argmax(dim=-1)
            top1_matches += float((student_top1 == teacher_top1).float().sum().item())

            student_top3 = student_logits.topk(min(3, student_logits.shape[-1]), dim=-1).indices
            teacher_top3 = teacher_logits.topk(min(3, teacher_logits.shape[-1]), dim=-1).indices
            student_top5 = student_logits.topk(min(5, student_logits.shape[-1]), dim=-1).indices
            teacher_top5 = teacher_logits.topk(min(5, teacher_logits.shape[-1]), dim=-1).indices
            top3_overlaps += float(
                (student_top3.unsqueeze(-1) == teacher_top3.unsqueeze(1))
                .any(dim=(-1, -2))
                .float()
                .sum()
                .item()
            )
            top5_overlaps += float(
                (student_top5.unsqueeze(-1) == teacher_top5.unsqueeze(1))
                .any(dim=(-1, -2))
                .float()
                .sum()
                .item()
            )
            sample_count += batch_size

    return {
        TEACHER_TOP1_AGREEMENT_KEY: _weighted_average(top1_matches, sample_count),
        TEACHER_TOP3_OVERLAP_KEY: _weighted_average(top3_overlaps, sample_count),
        TEACHER_TOP5_OVERLAP_KEY: _weighted_average(top5_overlaps, sample_count),
    }


def _scheduler_step(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    *,
    val_loss: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
        return
    scheduler.step()


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


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def run_distillation_loop(
    *,
    student: nn.Module,
    teacher: nn.Module,
    train_loader: Any,
    val_loader: Any,
    training: TrainingConfig,
    loss_config: DistillationLossConfig,
) -> tuple[list[dict[str, Any]], int, float, dict[str, Tensor]]:
    device = resolve_device(training.device)
    student.to(device)
    teacher.to(device)
    optimizer = create_optimizer(student, training)
    scheduler = create_scheduler(optimizer, training)
    checkpoint_dir = Path(training.checkpoint_dir)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path(training.log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(training.log_path).write_text("", encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_epoch = 0
    best_model_state: dict[str, Tensor] | None = None

    for epoch in range(1, training.epochs + 1):
        train_record = train_distillation_epoch(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_config=loss_config,
        )
        validation = validate(
            student,
            val_loader,
            device=device,
            num_classes=training.num_classes,
        )
        agreement = teacher_agreement(
            student,
            teacher,
            val_loader,
            device=device,
        )
        _scheduler_step(scheduler, val_loss=validation.val_loss)

        record = {
            "epoch": epoch,
            **train_record,
            **validation.as_dict(),
            **agreement,
        }
        history.append(record)
        append_jsonl_log(training.log_path, record)

        if training.best_metric not in record:
            raise ValueError(f"best_metric not found in metrics: {training.best_metric}")
        metric_value = float(record[training.best_metric])
        is_best = _metric_is_better(
            candidate=metric_value,
            best=best_metric_value,
            metric_name=training.best_metric,
        )
        if is_best:
            best_metric_value = metric_value
            best_epoch = epoch
            best_model_state = _clone_model_state(student)

        checkpoint_best_value = (
            metric_value if best_metric_value is None else best_metric_value
        )
        save_checkpoint(
            latest_path,
            model=student,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=training,
            metrics=record,
            best_metric_value=checkpoint_best_value,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config=training,
                metrics=record,
                best_metric_value=checkpoint_best_value,
            )

        LOGGER.info(
            "epoch=%s train_loss=%.6f base_loss=%.6f distill_loss=%.6f "
            "val_ndcg@5=%.6f teacher_top1_agreement=%.6f",
            epoch,
            train_record[TRAIN_LOSS_KEY],
            train_record[TRAIN_BASE_LOSS_KEY],
            train_record[TRAIN_DISTILLATION_LOSS_KEY],
            validation.val_ndcg_at_5,
            agreement[TEACHER_TOP1_AGREEMENT_KEY],
        )

    if best_metric_value is None or best_model_state is None:
        raise RuntimeError("distillation training did not produce a best checkpoint")
    return history, best_epoch, best_metric_value, best_model_state


def _dataset_kwargs(config: StudentDistillationConfig) -> dict[str, Path]:
    dataset_kwargs: dict[str, Path] = {}
    if config.labels_matrix is not None:
        dataset_kwargs["labels_matrix_path"] = config.labels_matrix
    if config.labels_soft is not None:
        dataset_kwargs["labels_soft_path"] = config.labels_soft
    return dataset_kwargs


def model_profile(
    model: nn.Module,
    *,
    device: torch.device,
    latency: LatencyConfig,
) -> dict[str, Any]:
    return {
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "model_size_mb": model_size_mb(model),
        "batch1_latency": measure_latency(
            model,
            device=device,
            batch_size=1,
            warmup_steps=latency.warmup_steps,
            benchmark_steps=latency.benchmark_steps,
        ),
        "batch64_latency": measure_latency(
            model,
            device=device,
            batch_size=64,
            warmup_steps=latency.warmup_steps,
            benchmark_steps=latency.benchmark_steps,
        ),
    }


def write_metrics_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report(
    path: Path,
    *,
    config: StudentDistillationConfig,
    result_payload: Mapping[str, Any],
) -> None:
    teacher_profile = result_payload["teacher_profile"]
    student_profile = result_payload["student_profile"]
    test_metrics = result_payload["test_metrics"]
    test_agreement = result_payload["test_teacher_agreement"]

    lines = [
        "# TitLeNet Student Distillation Report",
        "",
        "## Design Basis",
        "",
        "- Student architecture is newly registered as `titlenet_student`.",
        "- It is not a direct reuse of an existing ablation variant.",
        "- The design combines ablation-guided simplification with a mobile-oriented narrow channel plan, ECA attention, and a smaller classifier head.",
        "",
        "## Models",
        "",
        "| role | model | activation | checkpoint |",
        "| --- | --- | --- | --- |",
        (
            f"| Teacher | `{config.teacher.model_name}` | "
            f"`{config.teacher.activation}` | `{config.teacher_checkpoint}` |"
        ),
        (
            f"| Student | `{config.student.model_name}` | "
            f"`{config.student.activation}` | "
            f"`{Path(config.training.checkpoint_dir) / 'checkpoint_best.pt'}` |"
        ),
        "",
        "## Initialization",
        "",
        (
            f"- student_init_checkpoint: "
            f"`{config.student_init_checkpoint or 'random initialization'}`"
        ),
        "",
        "## Distillation",
        "",
        f"- temperature: `{config.loss.temperature}`",
        f"- base_loss_weight: `{config.loss.base_loss_weight}`",
        f"- distillation_loss_weight: `{config.loss.distillation_loss_weight}`",
        "",
        "## Profile",
        "",
        "| model | params | size_mb | batch1_ms | batch64_ms |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Teacher | {teacher_profile['total_parameters']} | "
            f"{teacher_profile['model_size_mb']:.6f} | "
            f"{teacher_profile['batch1_latency']['inference_time_ms']:.6f} | "
            f"{teacher_profile['batch64_latency']['inference_time_ms']:.6f} |"
        ),
        (
            f"| Student-distilled | {student_profile['total_parameters']} | "
            f"{student_profile['model_size_mb']:.6f} | "
            f"{student_profile['batch1_latency']['inference_time_ms']:.6f} | "
            f"{student_profile['batch64_latency']['inference_time_ms']:.6f} |"
        ),
        "",
        "## Test Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| val_loss | {float(test_metrics[VAL_LOSS_KEY]):.6f} |",
        f"| val_ndcg@3 | {float(test_metrics[VAL_NDCG_AT_3_KEY]):.6f} |",
        f"| val_ndcg@5 | {float(test_metrics[VAL_NDCG_AT_5_KEY]):.6f} |",
        (
            f"| teacher_top1_agreement | "
            f"{float(test_agreement[TEACHER_TOP1_AGREEMENT_KEY]):.6f} |"
        ),
        (
            f"| teacher_top3_overlap | "
            f"{float(test_agreement[TEACHER_TOP3_OVERLAP_KEY]):.6f} |"
        ),
        (
            f"| teacher_top5_overlap | "
            f"{float(test_agreement[TEACHER_TOP5_OVERLAP_KEY]):.6f} |"
        ),
        "",
        "## Artifacts",
        "",
        f"- best_checkpoint: `{Path(config.training.checkpoint_dir) / 'checkpoint_best.pt'}`",
        f"- latest_checkpoint: `{Path(config.training.checkpoint_dir) / 'checkpoint_latest.pt'}`",
        f"- log_path: `{config.training.log_path}`",
        f"- metrics_path: `{config.metrics_path}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> StudentDistillationResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = load_student_distillation_config(args)
    validate_distillation_config(config)
    set_training_seed(config.training.seed)
    device = resolve_device(config.training.device)

    loaders = create_title_color_dataloaders(
        batch_size=config.training.batch_size,
        splits=("train", "val", "test"),
        data_root=config.data_root,
        project_root=PROJECT_ROOT,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        seed=config.training.seed,
        dataset_kwargs=_dataset_kwargs(config),
    )
    dataset_sizes = {
        split: len(loaders[split].dataset)
        for split in ("train", "val", "test")
    }

    teacher = build_model(
        config.teacher,
        num_classes=config.training.num_classes,
        checkpoint_path=config.teacher_checkpoint,
    )
    student = build_model(
        config.student,
        num_classes=config.training.num_classes,
        checkpoint_path=config.student_init_checkpoint,
    )
    history, best_epoch, best_metric_value, best_state = run_distillation_loop(
        student=student,
        teacher=teacher,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        training=config.training,
        loss_config=config.loss,
    )

    student.load_state_dict(best_state)
    student.to(device)
    teacher.to(device)
    test_metrics = validate(
        student,
        loaders["test"],
        device=device,
        num_classes=config.training.num_classes,
    )
    test_agreement = teacher_agreement(
        student,
        teacher,
        loaders["test"],
        device=device,
    )
    teacher_profile = model_profile(teacher, device=device, latency=config.latency)
    student_profile = model_profile(student, device=device, latency=config.latency)
    checkpoint_paths = {
        "best": Path(config.training.checkpoint_dir) / "checkpoint_best.pt",
        "latest": Path(config.training.checkpoint_dir) / "checkpoint_latest.pt",
    }
    metrics_payload = {
        "teacher": asdict(config.teacher),
        "student": asdict(config.student),
        "student_init_checkpoint": (
            str(config.student_init_checkpoint)
            if config.student_init_checkpoint is not None
            else None
        ),
        "training": config.training.as_dict(),
        "distillation": asdict(config.loss),
        "dataset_sizes": dataset_sizes,
        "best_epoch": best_epoch,
        "best_metric": config.training.best_metric,
        "best_metric_value": best_metric_value,
        "test_metrics": test_metrics.as_dict(),
        "test_teacher_agreement": test_agreement,
        "teacher_profile": teacher_profile,
        "student_profile": student_profile,
        "checkpoint_paths": {
            name: str(path)
            for name, path in checkpoint_paths.items()
        },
    }
    write_metrics_json(config.metrics_path, metrics_payload)
    write_report(
        config.report_path,
        config=config,
        result_payload=metrics_payload,
    )
    return StudentDistillationResult(
        history=history,
        test_metrics=test_metrics,
        test_agreement=test_agreement,
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        dataset_sizes=dataset_sizes,
        checkpoint_paths=checkpoint_paths,
        report_path=config.report_path,
        metrics_path=config.metrics_path,
        metrics_payload=metrics_payload,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
