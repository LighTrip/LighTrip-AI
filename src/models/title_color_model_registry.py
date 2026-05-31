from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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
    build_titlenet_ablation_variant,
    build_titlenet_fast_a,
    build_titlenet_fast_b,
    build_titlenet_fast_c,
    build_titlenet_student,
    build_titlenet_stage_ablation_variant,
    build_vit_tiny,
    normalize_activation_name,
)


ModelBuilder = Callable[..., nn.Module]
MODEL_RESNET18 = "resnet18"
MODEL_RESNET34 = "resnet34"
MODEL_EFFICIENTNET_B0 = "efficientnet_b0"
MODEL_CONVNEXT_TINY = "convnext_tiny"
MODEL_VIT_TINY = "vit_tiny"
MODEL_TITLE_HYBRID_TINY = "title_hybrid_tiny"
MODEL_TITLE_HYBRID_FAST = "title_hybrid_fast"
MODEL_SWIN_TINY = "swin_tiny"
MODEL_FLATTEN_MLP = "flatten_mlp"
MODEL_SIMPLE_CNN = "simple_cnn"
MODEL_SIMPLE_CNN_M = "simple_cnn_m"
MODEL_SIMPLE_CNN_M_RES = "simple_cnn_m_res"
MODEL_SIMPLE_CNN_M_RES_MASK_POOL = "simple_cnn_m_res_mask_pool"
MODEL_SIMPLE_CNN_M_RES_SE = "simple_cnn_m_res_se"
MODEL_SIMPLE_CNN_M_RES_DEEPER = "simple_cnn_m_res_deeper"
MODEL_SIMPLE_CNN_M_MASK_POOL = "simple_cnn_m_mask_pool"
MODEL_SIMPLE_CNN_L = "simple_cnn_l"
MODEL_TITLENET = "titlenet"
MODEL_TITLENET_FAST_A = "titlenet_fast_a"
MODEL_TITLENET_FAST_B = "titlenet_fast_b"
MODEL_TITLENET_FAST_C = "titlenet_fast_c"
MODEL_TITLENET_STUDENT = "titlenet_student"
MODEL_MASK_AWARE_CNN = "mask_aware_cnn"
MODEL_MASK_AWARE_CNN_M = "mask_aware_cnn_m"
MODEL_MASK_AWARE_PALETTE_NET = "mask_aware_palette_net"
MODEL_MASK_AWARE_TINY_HYBRID_RANKER = "mask_aware_tiny_hybrid_ranker"
TITLENET_NO_SE = "titlenet_no_se"
TITLENET_NO_RESIDUAL = "titlenet_no_residual"
TITLENET_NO_FIRST_RESIDUAL = "titlenet_no_first_residual"
TITLENET_NO_MIDDLE_RESIDUAL = "titlenet_no_middle_residual"
TITLENET_NO_LAST_RESIDUAL = "titlenet_no_last_residual"
TITLENET_NO_LAST_EXTRA_RESIDUAL = "titlenet_no_last_extra_residual"
TITLENET_NO_STEM = "titlenet_no_stem"
TITLENET_NO_STAGE1 = "titlenet_no_stage1"
TITLENET_NO_STAGE2 = "titlenet_no_stage2"
TITLENET_NO_STAGE3 = "titlenet_no_stage3"
TITLENET_ECA = "titlenet_eca"
TITLENET_NARROW = "titlenet_narrow"
TITLENET_WIDE = "titlenet_wide"
TITLENET_SHALLOW = "titlenet_shallow"
TITLENET_DEEPER = "titlenet_deeper"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: ModelBuilder
    supports_pretrained: bool = False
    supports_activation: bool = False


def _custom_spec(name: str, builder: ModelBuilder) -> ModelSpec:
    return ModelSpec(name=name, builder=builder, supports_activation=True)


def _titlenet_ablation_spec(name: str, variant: str) -> ModelSpec:
    return _custom_spec(
        name,
        partial(build_titlenet_ablation_variant, variant=variant),
    )


def _titlenet_stage_ablation_spec(name: str, variant: str) -> ModelSpec:
    return _custom_spec(
        name,
        partial(build_titlenet_stage_ablation_variant, variant=variant),
    )


MODEL_SPECS: dict[str, ModelSpec] = {
    MODEL_RESNET18: ModelSpec(
        name=MODEL_RESNET18,
        builder=build_fixed_palette_resnet18,
        supports_pretrained=True,
    ),
    MODEL_RESNET34: ModelSpec(
        name=MODEL_RESNET34,
        builder=build_resnet34_classifier,
        supports_pretrained=True,
    ),
    MODEL_EFFICIENTNET_B0: ModelSpec(
        name=MODEL_EFFICIENTNET_B0,
        builder=build_efficientnet_b0_classifier,
        supports_pretrained=True,
    ),
    MODEL_CONVNEXT_TINY: ModelSpec(
        name=MODEL_CONVNEXT_TINY,
        builder=build_convnext_tiny_classifier,
        supports_pretrained=True,
    ),
    MODEL_VIT_TINY: ModelSpec(name=MODEL_VIT_TINY, builder=build_vit_tiny),
    MODEL_TITLE_HYBRID_TINY: _custom_spec(
        MODEL_TITLE_HYBRID_TINY,
        build_title_hybrid_tiny,
    ),
    MODEL_TITLE_HYBRID_FAST: _custom_spec(
        MODEL_TITLE_HYBRID_FAST,
        build_title_hybrid_fast,
    ),
    MODEL_SWIN_TINY: ModelSpec(
        name=MODEL_SWIN_TINY,
        builder=build_swin_tiny_classifier,
        supports_pretrained=True,
    ),
    MODEL_FLATTEN_MLP: ModelSpec(name=MODEL_FLATTEN_MLP, builder=build_flatten_mlp),
    MODEL_SIMPLE_CNN: _custom_spec(MODEL_SIMPLE_CNN, build_simple_cnn),
    MODEL_SIMPLE_CNN_M: _custom_spec(MODEL_SIMPLE_CNN_M, build_simple_cnn_medium),
    MODEL_SIMPLE_CNN_M_RES: _custom_spec(
        MODEL_SIMPLE_CNN_M_RES,
        build_simple_cnn_medium_residual,
    ),
    MODEL_SIMPLE_CNN_M_RES_MASK_POOL: _custom_spec(
        MODEL_SIMPLE_CNN_M_RES_MASK_POOL,
        build_simple_cnn_medium_residual_mask_pool,
    ),
    MODEL_SIMPLE_CNN_M_RES_SE: _custom_spec(
        MODEL_SIMPLE_CNN_M_RES_SE,
        build_simple_cnn_medium_residual_se,
    ),
    MODEL_TITLENET: _custom_spec(
        MODEL_TITLENET,
        build_simple_cnn_medium_residual_se,
    ),
    TITLENET_NO_SE: _titlenet_ablation_spec(TITLENET_NO_SE, "no_se"),
    TITLENET_NO_RESIDUAL: _titlenet_ablation_spec(
        TITLENET_NO_RESIDUAL,
        "no_residual",
    ),
    TITLENET_NO_FIRST_RESIDUAL: _titlenet_ablation_spec(
        TITLENET_NO_FIRST_RESIDUAL,
        "no_first_residual",
    ),
    TITLENET_NO_MIDDLE_RESIDUAL: _titlenet_ablation_spec(
        TITLENET_NO_MIDDLE_RESIDUAL,
        "no_middle_residual",
    ),
    TITLENET_NO_LAST_RESIDUAL: _titlenet_ablation_spec(
        TITLENET_NO_LAST_RESIDUAL,
        "no_last_residual",
    ),
    TITLENET_NO_LAST_EXTRA_RESIDUAL: _titlenet_ablation_spec(
        TITLENET_NO_LAST_EXTRA_RESIDUAL,
        "no_last_extra_residual",
    ),
    TITLENET_NO_STEM: _titlenet_stage_ablation_spec(
        TITLENET_NO_STEM,
        "no_stem",
    ),
    TITLENET_NO_STAGE1: _titlenet_stage_ablation_spec(
        TITLENET_NO_STAGE1,
        "no_stage1",
    ),
    TITLENET_NO_STAGE2: _titlenet_stage_ablation_spec(
        TITLENET_NO_STAGE2,
        "no_stage2",
    ),
    TITLENET_NO_STAGE3: _titlenet_stage_ablation_spec(
        TITLENET_NO_STAGE3,
        "no_stage3",
    ),
    TITLENET_ECA: _titlenet_ablation_spec(TITLENET_ECA, "eca"),
    TITLENET_NARROW: _titlenet_ablation_spec(TITLENET_NARROW, "narrow"),
    TITLENET_WIDE: _titlenet_ablation_spec(TITLENET_WIDE, "wide"),
    TITLENET_SHALLOW: _titlenet_ablation_spec(TITLENET_SHALLOW, "shallow"),
    TITLENET_DEEPER: _titlenet_ablation_spec(TITLENET_DEEPER, "deeper"),
    MODEL_TITLENET_FAST_A: _custom_spec(
        MODEL_TITLENET_FAST_A,
        build_titlenet_fast_a,
    ),
    MODEL_TITLENET_FAST_B: _custom_spec(
        MODEL_TITLENET_FAST_B,
        build_titlenet_fast_b,
    ),
    MODEL_TITLENET_FAST_C: _custom_spec(
        MODEL_TITLENET_FAST_C,
        build_titlenet_fast_c,
    ),
    MODEL_TITLENET_STUDENT: _custom_spec(
        MODEL_TITLENET_STUDENT,
        build_titlenet_student,
    ),
    MODEL_SIMPLE_CNN_M_RES_DEEPER: _custom_spec(
        MODEL_SIMPLE_CNN_M_RES_DEEPER,
        build_simple_cnn_medium_residual_deeper,
    ),
    MODEL_SIMPLE_CNN_M_MASK_POOL: _custom_spec(
        MODEL_SIMPLE_CNN_M_MASK_POOL,
        build_simple_cnn_medium_mask_pool,
    ),
    MODEL_SIMPLE_CNN_L: _custom_spec(MODEL_SIMPLE_CNN_L, build_simple_cnn_large),
    MODEL_MASK_AWARE_CNN: _custom_spec(MODEL_MASK_AWARE_CNN, build_mask_aware_cnn),
    MODEL_MASK_AWARE_CNN_M: _custom_spec(
        MODEL_MASK_AWARE_CNN_M,
        build_mask_aware_cnn_medium,
    ),
    MODEL_MASK_AWARE_PALETTE_NET: ModelSpec(
        name=MODEL_MASK_AWARE_PALETTE_NET,
        builder=build_mask_aware_palette_net,
    ),
    MODEL_MASK_AWARE_TINY_HYBRID_RANKER: _custom_spec(
        MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
        build_mask_aware_tiny_hybrid_ranker,
    ),
}
MODEL_ALIASES = {
    "efficientnet-b0": MODEL_EFFICIENTNET_B0,
    "convnext-tiny": MODEL_CONVNEXT_TINY,
    "vit-tiny": MODEL_VIT_TINY,
    "titlehybridtiny": MODEL_TITLE_HYBRID_TINY,
    "title-hybrid-tiny": MODEL_TITLE_HYBRID_TINY,
    "titleformer-lite": MODEL_TITLE_HYBRID_TINY,
    "titleformer_lite": MODEL_TITLE_HYBRID_TINY,
    "title_former_lite": MODEL_TITLE_HYBRID_TINY,
    "titlehybridfast": MODEL_TITLE_HYBRID_FAST,
    "title-hybrid-fast": MODEL_TITLE_HYBRID_FAST,
    "swin-tiny": MODEL_SWIN_TINY,
    "mlp": MODEL_FLATTEN_MLP,
    "flattenmlp": MODEL_FLATTEN_MLP,
    "simplecnn": MODEL_SIMPLE_CNN,
    "simple-cnn": MODEL_SIMPLE_CNN,
    "simplecnn-m": MODEL_SIMPLE_CNN_M,
    "simplecnn_m": MODEL_SIMPLE_CNN_M,
    "simple-cnn-m": MODEL_SIMPLE_CNN_M,
    "simplecnn-m-res": MODEL_SIMPLE_CNN_M_RES,
    "simplecnn_m_res": MODEL_SIMPLE_CNN_M_RES,
    "simple-cnn-m-res": MODEL_SIMPLE_CNN_M_RES,
    "simplecnn-m-residual": MODEL_SIMPLE_CNN_M_RES,
    "simple-cnn-m-residual": MODEL_SIMPLE_CNN_M_RES,
    "simplecnn-m-res-mask-pool": MODEL_SIMPLE_CNN_M_RES_MASK_POOL,
    "simplecnn_m_res_mask_pool": MODEL_SIMPLE_CNN_M_RES_MASK_POOL,
    "simple-cnn-m-res-mask-pool": MODEL_SIMPLE_CNN_M_RES_MASK_POOL,
    "simplecnn-m-res-se": MODEL_SIMPLE_CNN_M_RES_SE,
    "simplecnn_m_res_se": MODEL_SIMPLE_CNN_M_RES_SE,
    "simple-cnn-m-res-se": MODEL_SIMPLE_CNN_M_RES_SE,
    "title_net": MODEL_TITLENET,
    "title-net": MODEL_TITLENET,
    "titlenet-no-se": TITLENET_NO_SE,
    "title-net-no-se": TITLENET_NO_SE,
    "titlenet-without-se": TITLENET_NO_SE,
    "titlenet_without_se": TITLENET_NO_SE,
    "title-net-without-se": TITLENET_NO_SE,
    "title_net_without_se": TITLENET_NO_SE,
    "titlenet-no-residual": TITLENET_NO_RESIDUAL,
    "titlenet_no_residuals": TITLENET_NO_RESIDUAL,
    "titlenet-without-residual": TITLENET_NO_RESIDUAL,
    "titlenet_without_residual": TITLENET_NO_RESIDUAL,
    "title-net-no-residual": TITLENET_NO_RESIDUAL,
    "titlenet-no-first-residual": TITLENET_NO_FIRST_RESIDUAL,
    "titlenet_no_first_residual": TITLENET_NO_FIRST_RESIDUAL,
    "titlenet-no-stage1-residual": TITLENET_NO_FIRST_RESIDUAL,
    "titlenet_no_stage1_residual": TITLENET_NO_FIRST_RESIDUAL,
    "titlenet-no-middle-residual": TITLENET_NO_MIDDLE_RESIDUAL,
    "titlenet_no_middle_residual": TITLENET_NO_MIDDLE_RESIDUAL,
    "titlenet-no-stage2-residual": TITLENET_NO_MIDDLE_RESIDUAL,
    "titlenet_no_stage2_residual": TITLENET_NO_MIDDLE_RESIDUAL,
    "titlenet-no-last-residual": TITLENET_NO_LAST_RESIDUAL,
    "titlenet_no_last_residual": TITLENET_NO_LAST_RESIDUAL,
    "titlenet-no-stage3-residual": TITLENET_NO_LAST_RESIDUAL,
    "titlenet_no_stage3_residual": TITLENET_NO_LAST_RESIDUAL,
    "titlenet-no-last-extra-residual": TITLENET_NO_LAST_EXTRA_RESIDUAL,
    "titlenet_no_last_extra_residual": TITLENET_NO_LAST_EXTRA_RESIDUAL,
    "titlenet-no-extra-residual": TITLENET_NO_LAST_EXTRA_RESIDUAL,
    "titlenet_no_extra_residual": TITLENET_NO_LAST_EXTRA_RESIDUAL,
    "titlenet-no-stem": TITLENET_NO_STEM,
    "titlenet_without_stem": TITLENET_NO_STEM,
    "titlenet-without-stem": TITLENET_NO_STEM,
    "title-net-no-stem": TITLENET_NO_STEM,
    "titlenet-no-stage1": TITLENET_NO_STAGE1,
    "titlenet_no_stage1": TITLENET_NO_STAGE1,
    "titlenet-without-stage1": TITLENET_NO_STAGE1,
    "titlenet_without_stage1": TITLENET_NO_STAGE1,
    "titlenet-no-stage-1": TITLENET_NO_STAGE1,
    "titlenet-no-stage2": TITLENET_NO_STAGE2,
    "titlenet_no_stage2": TITLENET_NO_STAGE2,
    "titlenet-without-stage2": TITLENET_NO_STAGE2,
    "titlenet_without_stage2": TITLENET_NO_STAGE2,
    "titlenet-no-stage-2": TITLENET_NO_STAGE2,
    "titlenet-no-stage3": TITLENET_NO_STAGE3,
    "titlenet_no_stage3": TITLENET_NO_STAGE3,
    "titlenet-without-stage3": TITLENET_NO_STAGE3,
    "titlenet_without_stage3": TITLENET_NO_STAGE3,
    "titlenet-no-stage-3": TITLENET_NO_STAGE3,
    "titlenet-eca": TITLENET_ECA,
    "titlenet-with-eca": TITLENET_ECA,
    "titlenet_with_eca": TITLENET_ECA,
    "title-net-with-eca": TITLENET_ECA,
    "title_net_with_eca": TITLENET_ECA,
    "titlenet-narrow": TITLENET_NARROW,
    "title-net-narrow": TITLENET_NARROW,
    "titlenet-wide": TITLENET_WIDE,
    "title-net-wide": TITLENET_WIDE,
    "titlenet-shallow": TITLENET_SHALLOW,
    "title-net-shallow": TITLENET_SHALLOW,
    "titlenet-deeper": TITLENET_DEEPER,
    "title-net-deeper": TITLENET_DEEPER,
    "titlenet-fast-a": MODEL_TITLENET_FAST_A,
    "title-net-fast-a": MODEL_TITLENET_FAST_A,
    "titlenet-fast-b": MODEL_TITLENET_FAST_B,
    "title-net-fast-b": MODEL_TITLENET_FAST_B,
    "titlenet-fast-c": MODEL_TITLENET_FAST_C,
    "title-net-fast-c": MODEL_TITLENET_FAST_C,
    "titlenet-student": MODEL_TITLENET_STUDENT,
    "title-net-student": MODEL_TITLENET_STUDENT,
    "titlenet_student_v1": MODEL_TITLENET_STUDENT,
    "titlenet-student-v1": MODEL_TITLENET_STUDENT,
    "titlenet-ablation-guided-student": MODEL_TITLENET_STUDENT,
    "titlenet_ablation_guided_student": MODEL_TITLENET_STUDENT,
    "simplecnn-m-res-deeper": MODEL_SIMPLE_CNN_M_RES_DEEPER,
    "simplecnn_m_res_deeper": MODEL_SIMPLE_CNN_M_RES_DEEPER,
    "simple-cnn-m-res-deeper": MODEL_SIMPLE_CNN_M_RES_DEEPER,
    "simplecnn-m-mask-pool": MODEL_SIMPLE_CNN_M_MASK_POOL,
    "simplecnn_m_mask_pool": MODEL_SIMPLE_CNN_M_MASK_POOL,
    "simple-cnn-m-mask-pool": MODEL_SIMPLE_CNN_M_MASK_POOL,
    "simplecnn-l": MODEL_SIMPLE_CNN_L,
    "simplecnn_l": MODEL_SIMPLE_CNN_L,
    "simple-cnn-l": MODEL_SIMPLE_CNN_L,
    "maskawarecnn": MODEL_MASK_AWARE_CNN,
    "maskawarecnn-m": MODEL_MASK_AWARE_CNN_M,
    "maskawarecnn_m": MODEL_MASK_AWARE_CNN_M,
    "mask-aware-cnn-m": MODEL_MASK_AWARE_CNN_M,
    "maskawarepalettenet": MODEL_MASK_AWARE_PALETTE_NET,
    "maskawaretinyhybridranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
    "mask-aware-tiny-hybrid-ranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
    "mask_aware_tiny_hybrid_color_ranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
    "mask-aware-tiny-hybrid-color-ranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
    "tiny-hybrid-color-ranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
    "tiny_hybrid_color_ranker": MODEL_MASK_AWARE_TINY_HYBRID_RANKER,
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
