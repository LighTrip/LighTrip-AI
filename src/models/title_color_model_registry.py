from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from torch import nn

from src.models.fixed_palette_classifier import (
    DEFAULT_NUM_CLASSES,
    build_fixed_palette_resnet18,
)
from src.models.title_color_initialization import (
    DEFAULT_WEIGHT_INIT,
    apply_weight_initialization,
    normalize_weight_init_name,
)
from src.models.title_color_models import (
    DEFAULT_ACTIVATION,
    build_convnext_tiny_classifier,
    build_efficientnet_b0_classifier,
    build_flatten_mlp,
    build_mask_aware_cnn,
    build_mask_aware_cnn_medium,
    build_mask_aware_palette_net,
    build_mask_aware_tiny_hybrid_ranker,
    build_resnet34_classifier,
    build_simple_cnn,
    build_simple_cnn_large,
    build_simple_cnn_medium,
    build_simple_cnn_medium_mask_pool,
    build_simple_cnn_medium_residual_deeper,
    build_simple_cnn_medium_residual_mask_pool,
    build_simple_cnn_medium_residual,
    build_simple_cnn_medium_residual_se,
    build_swin_tiny_classifier,
    build_title_hybrid_fast,
    build_title_hybrid_tiny,
    build_titlenet_fast_a,
    build_titlenet_fast_b,
    build_titlenet_fast_c,
    build_vit_tiny,
    normalize_activation_name,
)


ModelBuilder = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: ModelBuilder
    supports_pretrained: bool = False
    supports_activation: bool = False


def _custom_spec(name: str, builder: ModelBuilder) -> ModelSpec:
    return ModelSpec(name=name, builder=builder, supports_activation=True)


MODEL_SPECS: dict[str, ModelSpec] = {
    "resnet18": ModelSpec(
        name="resnet18",
        builder=build_fixed_palette_resnet18,
        supports_pretrained=True,
    ),
    "resnet34": ModelSpec(
        name="resnet34",
        builder=build_resnet34_classifier,
        supports_pretrained=True,
    ),
    "efficientnet_b0": ModelSpec(
        name="efficientnet_b0",
        builder=build_efficientnet_b0_classifier,
        supports_pretrained=True,
    ),
    "convnext_tiny": ModelSpec(
        name="convnext_tiny",
        builder=build_convnext_tiny_classifier,
        supports_pretrained=True,
    ),
    "vit_tiny": ModelSpec(name="vit_tiny", builder=build_vit_tiny),
    "title_hybrid_tiny": _custom_spec(
        "title_hybrid_tiny",
        build_title_hybrid_tiny,
    ),
    "title_hybrid_fast": _custom_spec(
        "title_hybrid_fast",
        build_title_hybrid_fast,
    ),
    "swin_tiny": ModelSpec(
        name="swin_tiny",
        builder=build_swin_tiny_classifier,
        supports_pretrained=True,
    ),
    "flatten_mlp": ModelSpec(name="flatten_mlp", builder=build_flatten_mlp),
    "simple_cnn": _custom_spec("simple_cnn", build_simple_cnn),
    "simple_cnn_m": _custom_spec("simple_cnn_m", build_simple_cnn_medium),
    "simple_cnn_m_res": _custom_spec(
        "simple_cnn_m_res",
        build_simple_cnn_medium_residual,
    ),
    "simple_cnn_m_res_mask_pool": _custom_spec(
        "simple_cnn_m_res_mask_pool",
        build_simple_cnn_medium_residual_mask_pool,
    ),
    "simple_cnn_m_res_se": _custom_spec(
        "simple_cnn_m_res_se",
        build_simple_cnn_medium_residual_se,
    ),
    "titlenet": _custom_spec(
        "titlenet",
        build_simple_cnn_medium_residual_se,
    ),
    "titlenet_fast_a": _custom_spec(
        "titlenet_fast_a",
        build_titlenet_fast_a,
    ),
    "titlenet_fast_b": _custom_spec(
        "titlenet_fast_b",
        build_titlenet_fast_b,
    ),
    "titlenet_fast_c": _custom_spec(
        "titlenet_fast_c",
        build_titlenet_fast_c,
    ),
    "simple_cnn_m_res_deeper": _custom_spec(
        "simple_cnn_m_res_deeper",
        build_simple_cnn_medium_residual_deeper,
    ),
    "simple_cnn_m_mask_pool": _custom_spec(
        "simple_cnn_m_mask_pool",
        build_simple_cnn_medium_mask_pool,
    ),
    "simple_cnn_l": _custom_spec("simple_cnn_l", build_simple_cnn_large),
    "mask_aware_cnn": _custom_spec("mask_aware_cnn", build_mask_aware_cnn),
    "mask_aware_cnn_m": _custom_spec(
        "mask_aware_cnn_m",
        build_mask_aware_cnn_medium,
    ),
    "mask_aware_palette_net": ModelSpec(
        name="mask_aware_palette_net",
        builder=build_mask_aware_palette_net,
    ),
    "mask_aware_tiny_hybrid_ranker": _custom_spec(
        "mask_aware_tiny_hybrid_ranker",
        build_mask_aware_tiny_hybrid_ranker,
    ),
}
MODEL_ALIASES = {
    "efficientnet-b0": "efficientnet_b0",
    "convnext-tiny": "convnext_tiny",
    "vit-tiny": "vit_tiny",
    "titlehybridtiny": "title_hybrid_tiny",
    "title-hybrid-tiny": "title_hybrid_tiny",
    "titleformer-lite": "title_hybrid_tiny",
    "titleformer_lite": "title_hybrid_tiny",
    "title_former_lite": "title_hybrid_tiny",
    "titlehybridfast": "title_hybrid_fast",
    "title-hybrid-fast": "title_hybrid_fast",
    "swin-tiny": "swin_tiny",
    "mlp": "flatten_mlp",
    "flattenmlp": "flatten_mlp",
    "simplecnn": "simple_cnn",
    "simple-cnn": "simple_cnn",
    "simplecnn-m": "simple_cnn_m",
    "simplecnn_m": "simple_cnn_m",
    "simple-cnn-m": "simple_cnn_m",
    "simplecnn-m-res": "simple_cnn_m_res",
    "simplecnn_m_res": "simple_cnn_m_res",
    "simple-cnn-m-res": "simple_cnn_m_res",
    "simplecnn-m-residual": "simple_cnn_m_res",
    "simple-cnn-m-residual": "simple_cnn_m_res",
    "simplecnn-m-res-mask-pool": "simple_cnn_m_res_mask_pool",
    "simplecnn_m_res_mask_pool": "simple_cnn_m_res_mask_pool",
    "simple-cnn-m-res-mask-pool": "simple_cnn_m_res_mask_pool",
    "simplecnn-m-res-se": "simple_cnn_m_res_se",
    "simplecnn_m_res_se": "simple_cnn_m_res_se",
    "simple-cnn-m-res-se": "simple_cnn_m_res_se",
    "title_net": "titlenet",
    "title-net": "titlenet",
    "titlenet-fast-a": "titlenet_fast_a",
    "title-net-fast-a": "titlenet_fast_a",
    "titlenet-fast-b": "titlenet_fast_b",
    "title-net-fast-b": "titlenet_fast_b",
    "titlenet-fast-c": "titlenet_fast_c",
    "title-net-fast-c": "titlenet_fast_c",
    "simplecnn-m-res-deeper": "simple_cnn_m_res_deeper",
    "simplecnn_m_res_deeper": "simple_cnn_m_res_deeper",
    "simple-cnn-m-res-deeper": "simple_cnn_m_res_deeper",
    "simplecnn-m-mask-pool": "simple_cnn_m_mask_pool",
    "simplecnn_m_mask_pool": "simple_cnn_m_mask_pool",
    "simple-cnn-m-mask-pool": "simple_cnn_m_mask_pool",
    "simplecnn-l": "simple_cnn_l",
    "simplecnn_l": "simple_cnn_l",
    "simple-cnn-l": "simple_cnn_l",
    "maskawarecnn": "mask_aware_cnn",
    "maskawarecnn-m": "mask_aware_cnn_m",
    "maskawarecnn_m": "mask_aware_cnn_m",
    "mask-aware-cnn-m": "mask_aware_cnn_m",
    "maskawarepalettenet": "mask_aware_palette_net",
    "maskawaretinyhybridranker": "mask_aware_tiny_hybrid_ranker",
    "mask-aware-tiny-hybrid-ranker": "mask_aware_tiny_hybrid_ranker",
    "mask_aware_tiny_hybrid_color_ranker": "mask_aware_tiny_hybrid_ranker",
    "mask-aware-tiny-hybrid-color-ranker": "mask_aware_tiny_hybrid_ranker",
    "tiny-hybrid-color-ranker": "mask_aware_tiny_hybrid_ranker",
    "tiny_hybrid_color_ranker": "mask_aware_tiny_hybrid_ranker",
}


def normalize_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    return MODEL_ALIASES.get(normalized, normalized)


def available_model_names() -> list[str]:
    return sorted(MODEL_SPECS)


def get_model_spec(name: str) -> ModelSpec:
    normalized = normalize_model_name(name)
    try:
        return MODEL_SPECS[normalized]
    except KeyError as exc:
        available = ", ".join(available_model_names())
        raise ValueError(f"unknown model_name={name!r}; available={available}") from exc


def build_title_color_model(
    model_name: str,
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    pretrained: bool = False,
    dropout: float = 0.2,
    weight_init: str | None = DEFAULT_WEIGHT_INIT,
    activation: str | None = DEFAULT_ACTIVATION,
) -> nn.Module:
    spec = get_model_spec(model_name)
    normalized_weight_init = normalize_weight_init_name(weight_init)
    normalized_activation = normalize_activation_name(activation)
    if pretrained and not spec.supports_pretrained:
        raise ValueError(f"model does not support pretrained weights: {spec.name}")
    if pretrained and normalized_weight_init != DEFAULT_WEIGHT_INIT:
        raise ValueError("weight_init must be pytorch_default when pretrained=True")
    if normalized_activation != DEFAULT_ACTIVATION and not spec.supports_activation:
        raise ValueError(f"model does not support custom activation: {spec.name}")
    builder_kwargs = {
        "num_classes": num_classes,
        "pretrained": pretrained,
        "dropout": dropout,
    }
    if spec.supports_activation:
        builder_kwargs["activation"] = normalized_activation
    model = spec.builder(**builder_kwargs)
    return apply_weight_initialization(model, normalized_weight_init)
