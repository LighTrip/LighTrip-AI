from __future__ import annotations

from torch import nn


DEFAULT_WEIGHT_INIT = "pytorch_default"
WEIGHT_INIT_ALIASES = {
    "default": DEFAULT_WEIGHT_INIT,
    "none": DEFAULT_WEIGHT_INIT,
    "pytorch-default": DEFAULT_WEIGHT_INIT,
    "kaiming": "kaiming_normal",
    "he_normal": "kaiming_normal",
    "he_uniform": "kaiming_uniform",
    "xavier": "xavier_uniform",
    "glorot_normal": "xavier_normal",
    "glorot_uniform": "xavier_uniform",
    "small-head": "small_head",
}
WEIGHT_INIT_STRATEGIES = (
    DEFAULT_WEIGHT_INIT,
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "small_head",
)


def available_weight_initializers() -> list[str]:
    return list(WEIGHT_INIT_STRATEGIES)


def normalize_weight_init_name(name: str | None) -> str:
    if name is None:
        return DEFAULT_WEIGHT_INIT
    normalized = name.strip().lower().replace("-", "_")
    normalized = WEIGHT_INIT_ALIASES.get(normalized, normalized)
    if normalized not in WEIGHT_INIT_STRATEGIES:
        available = ", ".join(WEIGHT_INIT_STRATEGIES)
        raise ValueError(f"unknown weight_init={name!r}; available={available}")
    return normalized


def _zero_bias(module: nn.Module) -> None:
    bias = getattr(module, "bias", None)
    if bias is not None:
        nn.init.zeros_(bias)


def _init_weight(module: nn.Module, strategy: str) -> None:
    weight = getattr(module, "weight", None)
    if weight is None:
        return
    if strategy == "kaiming_normal":
        nn.init.kaiming_normal_(weight, mode="fan_out", nonlinearity="relu")
    elif strategy == "kaiming_uniform":
        nn.init.kaiming_uniform_(weight, mode="fan_in", nonlinearity="relu")
    elif strategy == "xavier_normal":
        nn.init.xavier_normal_(weight)
    elif strategy == "xavier_uniform":
        nn.init.xavier_uniform_(weight)
    else:
        raise ValueError(f"unsupported initializer strategy: {strategy}")


def _linear_modules(model: nn.Module) -> list[nn.Linear]:
    return [module for module in model.modules() if isinstance(module, nn.Linear)]


def _init_normalization(module: nn.Module) -> None:
    weight = getattr(module, "weight", None)
    if weight is not None:
        nn.init.ones_(weight)
    _zero_bias(module)


def apply_weight_initialization(
    model: nn.Module,
    strategy: str | None = DEFAULT_WEIGHT_INIT,
) -> nn.Module:
    normalized = normalize_weight_init_name(strategy)
    if normalized == DEFAULT_WEIGHT_INIT:
        return model

    linears = _linear_modules(model)
    final_linear_ids = {id(linears[-1])} if linears else set()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            _init_normalization(module)
        elif isinstance(module, (nn.Conv2d, nn.Linear)):
            if normalized == "small_head" and id(module) in final_linear_ids:
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif normalized == "small_head":
                hidden_strategy = (
                    "kaiming_normal"
                    if isinstance(module, nn.Conv2d)
                    else "xavier_uniform"
                )
                _init_weight(module, hidden_strategy)
            else:
                _init_weight(module, normalized)
            _zero_bias(module)
    return model
