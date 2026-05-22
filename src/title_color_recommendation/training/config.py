from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_classes: int = 32
    num_workers: int = 4
    device: str = "cuda"
    scheduler: str = "cosine"
    checkpoint_dir: str = "outputs/title_color_recommendation/checkpoints"
    log_path: str = "outputs/title_color_recommendation/train_log.jsonl"
    best_metric: str = "val_loss"
    seed: int = 42

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def training_config_from_mapping(
    values: Mapping[str, Any],
    *,
    defaults: TrainingConfig | None = None,
) -> TrainingConfig:
    base = defaults.as_dict() if defaults is not None else TrainingConfig().as_dict()
    valid_names = {field.name for field in fields(TrainingConfig)}
    unknown_names = sorted(set(values).difference(valid_names))
    if unknown_names:
        raise ValueError(f"Unknown training config keys: {unknown_names}")
    base.update(values)
    return TrainingConfig(**base)


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return stripped.strip("\"'")


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            content = line.split("#", 1)[0].strip()
            if not content:
                continue
            if ":" not in content:
                raise ValueError(
                    f"Invalid config line {line_number} in {path}: {line.rstrip()}"
                )
            key, value = content.split(":", 1)
            values[key.strip()] = _parse_scalar(value)
    return values


def load_training_config(path: str | Path) -> TrainingConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"training config not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            values = json.load(f)
    elif suffix in {".yaml", ".yml"}:
        values = _load_simple_yaml(config_path)
    else:
        raise ValueError(f"Unsupported training config format: {config_path}")

    if not isinstance(values, Mapping):
        raise ValueError(f"training config must be a mapping: {config_path}")
    return training_config_from_mapping(values)
