from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation import (
    train_titlenet_student_distillation as distillation,
)
from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from src.title_color_recommendation.data.dataloader import (
    create_title_color_dataloaders,
)
from src.title_color_recommendation.training.losses import (
    combined_soft_label_distillation_loss,
)
from src.title_color_recommendation.training.trainer import append_jsonl_log
from src.title_color_recommendation.training.trainer import create_optimizer
from src.title_color_recommendation.training.trainer import create_scheduler
from src.title_color_recommendation.training.trainer import resolve_device
from src.title_color_recommendation.training.trainer import set_training_seed
from src.title_color_recommendation.training.trainer import validate


LOGGER = logging.getLogger(__name__)
DEFAULT_STUDENT_INIT_CHECKPOINT = Path(
    "outputs/checkpoints/titlenet_student_kd_weight_sweep/"
    "warm_start/kd_90_10/checkpoint_best.pt"
)
DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints/titlenet_student_qat/kd_90_10")
DEFAULT_LOG_PATH = Path("outputs/logs/titlenet_student_qat_kd_90_10.jsonl")
DEFAULT_REPORT_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_qat_kd_90_10_report.md"
)
DEFAULT_METRICS_PATH = Path(
    "outputs/reports/model_evaluation/titlenet_student_qat_kd_90_10_metrics.json"
)
DEFAULT_QAT_EPOCHS = 5
DEFAULT_QAT_LEARNING_RATE = 5e-5
DEFAULT_BASE_LOSS_WEIGHT = 0.9
DEFAULT_DISTILLATION_LOSS_WEIGHT = 0.1
DEFAULT_QAT_BACKEND = "qnnpack"
DEFAULT_DISABLE_OBSERVER_EPOCH = 4
DEFAULT_FREEZE_BN_EPOCH = 4


@dataclass(frozen=True)
class QATRuntimeConfig:
    backend: str
    disable_observer_epoch: int
    freeze_bn_epoch: int


@dataclass(frozen=True)
class QATResult:
    history: list[dict[str, Any]]
    test_metrics: Mapping[str, Any]
    test_agreement: Mapping[str, float]
    best_epoch: int
    best_metric_value: float
    dataset_sizes: dict[str, int]
    checkpoint_paths: dict[str, Path]
    report_path: Path
    metrics_path: Path
    metrics_payload: dict[str, Any]


class QuantizedIOWrapper(nn.Module):
    def __init__(self, wrapped_model: nn.Module) -> None:
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.wrapped_model = wrapped_model
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        return self.dequant(self.wrapped_model(self.quant(x)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the best TitLeNet Student with fake-quant QAT before "
            "ONNX/PTQ export."
        )
    )
    distillation.add_distillation_training_args(parser)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--qat-backend",
        choices=("qnnpack", "fbgemm", "x86"),
        default=DEFAULT_QAT_BACKEND,
    )
    parser.add_argument(
        "--disable-observer-epoch",
        type=int,
        default=DEFAULT_DISABLE_OBSERVER_EPOCH,
    )
    parser.add_argument(
        "--freeze-bn-epoch",
        type=int,
        default=DEFAULT_FREEZE_BN_EPOCH,
    )
    args = parser.parse_args(argv)
    apply_qat_defaults(args)
    return args


def apply_qat_defaults(args: argparse.Namespace) -> None:
    if args.student_init_checkpoint is None:
        args.student_init_checkpoint = DEFAULT_STUDENT_INIT_CHECKPOINT
    if args.epochs is None:
        args.epochs = DEFAULT_QAT_EPOCHS
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_QAT_LEARNING_RATE
    if args.base_loss_weight is None:
        args.base_loss_weight = DEFAULT_BASE_LOSS_WEIGHT
    if args.distillation_loss_weight is None:
        args.distillation_loss_weight = DEFAULT_DISTILLATION_LOSS_WEIGHT


def validate_qat_runtime_config(config: QATRuntimeConfig) -> None:
    if config.disable_observer_epoch <= 0:
        raise ValueError("disable_observer_epoch must be positive")
    if config.freeze_bn_epoch <= 0:
        raise ValueError("freeze_bn_epoch must be positive")
    if config.backend not in torch.backends.quantized.supported_engines:
        raise ValueError(
            f"QAT backend is not supported by this PyTorch build: {config.backend}"
        )


def prepare_qat_student(
    student: nn.Module,
    *,
    backend: str,
) -> QuantizedIOWrapper:
    torch.backends.quantized.engine = backend
    prepared = QuantizedIOWrapper(student)
    prepared.train()
    prepared.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    torch.ao.quantization.prepare_qat(prepared, inplace=True)
    return prepared


def disable_observers(model: nn.Module) -> None:
    model.apply(torch.ao.quantization.disable_observer)


def freeze_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def apply_qat_epoch_schedule(
    model: nn.Module,
    *,
    epoch: int,
    qat_config: QATRuntimeConfig,
) -> None:
    if epoch >= qat_config.disable_observer_epoch:
        disable_observers(model)
    if epoch >= qat_config.freeze_bn_epoch:
        freeze_batch_norm(model)


def train_qat_epoch(
    *,
    student: nn.Module,
    teacher: nn.Module,
    dataloader: Any,
    optimizer: Optimizer,
    device: torch.device,
    loss_config: distillation.DistillationLossConfig,
    epoch: int,
    qat_config: QATRuntimeConfig,
) -> dict[str, float]:
    student.train()
    apply_qat_epoch_schedule(student, epoch=epoch, qat_config=qat_config)
    teacher.eval()
    total_loss = 0.0
    total_base_loss = 0.0
    total_distillation_loss = 0.0
    sample_count = 0

    for batch in dataloader:
        moved_batch = distillation._move_batch_to_device(batch, device)
        x = distillation._required_tensor(moved_batch, "x")
        target_distribution = distillation._required_tensor(
            moved_batch,
            "target_distribution",
        )

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
        distillation.TRAIN_LOSS_KEY: distillation._weighted_average(
            total_loss,
            sample_count,
        ),
        distillation.TRAIN_BASE_LOSS_KEY: distillation._weighted_average(
            total_base_loss,
            sample_count,
        ),
        distillation.TRAIN_DISTILLATION_LOSS_KEY: distillation._weighted_average(
            total_distillation_loss,
            sample_count,
        ),
    }


def scheduler_state_dict(
    scheduler: LRScheduler | ReduceLROnPlateau | None,
) -> dict[str, Any] | None:
    if scheduler is None:
        return None
    return scheduler.state_dict()


def extract_float_student_state(
    *,
    prepared_student: QuantizedIOWrapper,
    reference_student: nn.Module,
) -> dict[str, Tensor]:
    prepared_state = prepared_student.wrapped_model.state_dict()
    reference_keys = set(reference_student.state_dict())
    missing_keys = sorted(reference_keys.difference(prepared_state))
    if missing_keys:
        raise KeyError(f"QAT prepared model is missing base keys: {missing_keys[:5]}")
    return {
        key: prepared_state[key].detach().cpu().clone()
        for key in reference_student.state_dict()
    }


def save_qat_float_checkpoint(
    path: Path,
    *,
    model_state_dict: Mapping[str, Tensor],
    optimizer: Optimizer,
    scheduler: LRScheduler | ReduceLROnPlateau | None,
    epoch: int,
    training: Any,
    qat_config: QATRuntimeConfig,
    metrics: Mapping[str, Any],
    best_metric_value: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **training.as_dict(),
        "qat": asdict(qat_config),
        "checkpoint_note": (
            "Float student weights trained under fake-quant QAT. Export this "
            "checkpoint to ONNX, then run PTQ/static quantization for deployment."
        ),
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": dict(model_state_dict),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state_dict(scheduler),
            "config": config_payload,
            "metrics": dict(metrics),
            "best_metric": training.best_metric,
            "best_metric_value": best_metric_value,
        },
        path,
    )


def run_qat_loop(
    *,
    prepared_student: QuantizedIOWrapper,
    reference_student: nn.Module,
    teacher: nn.Module,
    train_loader: Any,
    val_loader: Any,
    training: Any,
    loss_config: distillation.DistillationLossConfig,
    qat_config: QATRuntimeConfig,
) -> tuple[list[dict[str, Any]], int, float, dict[str, Tensor], dict[str, Tensor]]:
    device = resolve_device(training.device)
    prepared_student.to(device)
    teacher.to(device)
    optimizer = create_optimizer(prepared_student, training)
    scheduler = create_scheduler(optimizer, training)
    checkpoint_dir = resolve_inside_project(
        PROJECT_ROOT,
        training.checkpoint_dir,
        description="checkpoint_dir",
    )
    log_path = resolve_inside_project(
        PROJECT_ROOT,
        training.log_path,
        description="log_path",
    )
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    best_path = checkpoint_dir / "checkpoint_best.pt"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    history: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_epoch = 0
    best_float_state: dict[str, Tensor] | None = None
    best_qat_state: dict[str, Tensor] | None = None

    for epoch in range(1, training.epochs + 1):
        train_record = train_qat_epoch(
            student=prepared_student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_config=loss_config,
            epoch=epoch,
            qat_config=qat_config,
        )
        validation = validate(
            prepared_student,
            val_loader,
            device=device,
            num_classes=training.num_classes,
        )
        agreement = distillation.teacher_agreement(
            prepared_student,
            teacher,
            val_loader,
            device=device,
        )
        distillation._scheduler_step(scheduler, val_loss=validation.val_loss)

        record = {
            "epoch": epoch,
            **train_record,
            **validation.as_dict(),
            **agreement,
            "qat_observer_enabled": epoch < qat_config.disable_observer_epoch,
            "qat_batch_norm_frozen": epoch >= qat_config.freeze_bn_epoch,
        }
        history.append(record)
        append_jsonl_log(log_path, record)

        if training.best_metric not in record:
            raise ValueError(f"best_metric not found in metrics: {training.best_metric}")
        metric_value = float(record[training.best_metric])
        is_best = distillation._metric_is_better(
            candidate=metric_value,
            best=best_metric_value,
            metric_name=training.best_metric,
        )
        if is_best:
            best_metric_value = metric_value
            best_epoch = epoch
            best_float_state = extract_float_student_state(
                prepared_student=prepared_student,
                reference_student=reference_student,
            )
            best_qat_state = distillation._clone_model_state(prepared_student)

        checkpoint_best_value = (
            metric_value if best_metric_value is None else best_metric_value
        )
        latest_float_state = extract_float_student_state(
            prepared_student=prepared_student,
            reference_student=reference_student,
        )
        save_qat_float_checkpoint(
            latest_path,
            model_state_dict=latest_float_state,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            training=training,
            qat_config=qat_config,
            metrics=record,
            best_metric_value=checkpoint_best_value,
        )
        if is_best and best_float_state is not None:
            save_qat_float_checkpoint(
                best_path,
                model_state_dict=best_float_state,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                training=training,
                qat_config=qat_config,
                metrics=record,
                best_metric_value=checkpoint_best_value,
            )

        LOGGER.info(
            "epoch=%s train_loss=%.6f val_ndcg@5=%.6f teacher_top1=%.6f",
            epoch,
            train_record[distillation.TRAIN_LOSS_KEY],
            validation.val_ndcg_at_5,
            agreement[distillation.TEACHER_TOP1_AGREEMENT_KEY],
        )

    if (
        best_metric_value is None
        or best_float_state is None
        or best_qat_state is None
    ):
        raise RuntimeError("QAT training did not produce a best checkpoint")
    return history, best_epoch, best_metric_value, best_float_state, best_qat_state


def write_report(
    path: Path,
    *,
    config: distillation.StudentDistillationConfig,
    qat_config: QATRuntimeConfig,
    payload: Mapping[str, Any],
) -> None:
    test_metrics = payload["test_metrics"]
    test_agreement = payload["test_teacher_agreement"]
    checkpoint_paths = payload["checkpoint_paths"]
    lines = [
        "# TitLeNet Student QAT Report",
        "",
        "## Summary",
        "",
        f"- student: `{config.student.model_name}`",
        f"- init_checkpoint: `{config.student_init_checkpoint}`",
        f"- teacher_checkpoint: `{config.teacher_checkpoint}`",
        f"- qat_backend: `{qat_config.backend}`",
        f"- disable_observer_epoch: `{qat_config.disable_observer_epoch}`",
        f"- freeze_bn_epoch: `{qat_config.freeze_bn_epoch}`",
        "",
        "## Loss",
        "",
        f"- temperature: `{config.loss.temperature}`",
        f"- base_loss_weight: `{config.loss.base_loss_weight}`",
        f"- distillation_loss_weight: `{config.loss.distillation_loss_weight}`",
        "",
        "## Test Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| val_loss | {float(test_metrics[distillation.VAL_LOSS_KEY]):.6f} |",
        f"| val_ndcg@3 | {float(test_metrics[distillation.VAL_NDCG_AT_3_KEY]):.6f} |",
        f"| val_ndcg@5 | {float(test_metrics[distillation.VAL_NDCG_AT_5_KEY]):.6f} |",
        (
            f"| teacher_top1_agreement | "
            f"{float(test_agreement[distillation.TEACHER_TOP1_AGREEMENT_KEY]):.6f} |"
        ),
        "",
        "## Artifacts",
        "",
        f"- best_checkpoint: `{checkpoint_paths['best']}`",
        f"- latest_checkpoint: `{checkpoint_paths['latest']}`",
        f"- log_path: `{config.training.log_path}`",
        f"- metrics_path: `{config.metrics_path}`",
        "",
        "## Next Step",
        "",
        (
            "- Export `checkpoint_best.pt` to ONNX, then run "
            "`quantize_titlenet_student_onnx.py` to compare QAT+PTQ against "
            "the FP32 Student baseline."
        ),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> QATResult:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = distillation.load_student_distillation_config(args)
    distillation.validate_distillation_config(config)
    qat_config = QATRuntimeConfig(
        backend=args.qat_backend,
        disable_observer_epoch=args.disable_observer_epoch,
        freeze_bn_epoch=args.freeze_bn_epoch,
    )
    validate_qat_runtime_config(qat_config)
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
        dataset_kwargs=distillation._dataset_kwargs(config),
    )
    dataset_sizes = {
        split: len(loaders[split].dataset)
        for split in ("train", "val", "test")
    }
    teacher = distillation.build_model(
        config.teacher,
        num_classes=config.training.num_classes,
        checkpoint_path=config.teacher_checkpoint,
    )
    base_student = distillation.build_model(
        config.student,
        num_classes=config.training.num_classes,
        checkpoint_path=config.student_init_checkpoint,
    )
    reference_student = distillation.build_model(
        config.student,
        num_classes=config.training.num_classes,
        checkpoint_path=None,
    )
    prepared_student = prepare_qat_student(base_student, backend=qat_config.backend)
    (
        history,
        best_epoch,
        best_metric_value,
        best_float_state,
        best_qat_state,
    ) = run_qat_loop(
        prepared_student=prepared_student,
        reference_student=reference_student,
        teacher=teacher,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        training=config.training,
        loss_config=config.loss,
        qat_config=qat_config,
    )

    prepared_student.load_state_dict(best_qat_state)
    prepared_student.to(device)
    teacher.to(device)
    test_metrics = validate(
        prepared_student,
        loaders["test"],
        device=device,
        num_classes=config.training.num_classes,
    )
    test_agreement = distillation.teacher_agreement(
        prepared_student,
        teacher,
        loaders["test"],
        device=device,
    )
    reference_student.load_state_dict(best_float_state)
    checkpoint_paths = {
        "best": Path(config.training.checkpoint_dir) / "checkpoint_best.pt",
        "latest": Path(config.training.checkpoint_dir) / "checkpoint_latest.pt",
    }
    metrics_payload = {
        "teacher": asdict(config.teacher),
        "student": asdict(config.student),
        "student_init_checkpoint": str(config.student_init_checkpoint),
        "training": config.training.as_dict(),
        "distillation": asdict(config.loss),
        "qat": asdict(qat_config),
        "dataset_sizes": dataset_sizes,
        "best_epoch": best_epoch,
        "best_metric": config.training.best_metric,
        "best_metric_value": best_metric_value,
        "test_metrics": test_metrics.as_dict(),
        "test_teacher_agreement": test_agreement,
        "checkpoint_paths": {
            name: str(path)
            for name, path in checkpoint_paths.items()
        },
    }
    distillation.write_metrics_json(config.metrics_path, metrics_payload)
    write_report(
        config.report_path,
        config=config,
        qat_config=qat_config,
        payload=metrics_payload,
    )
    return QATResult(
        history=history,
        test_metrics=test_metrics.as_dict(),
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
