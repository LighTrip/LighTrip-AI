from __future__ import annotations

from typing import Any

import pytest


GELU_ACTIVATION = "gelu"
HARDSWISH_ACTIVATION = "hardswish"
RESNET18 = "resnet18"
TITLE_HYBRID_TINY = "title_hybrid_tiny"
TITLE_HYBRID_FAST = "title_hybrid_fast"
FLATTEN_MLP = "flatten_mlp"
SIMPLE_CNN = "simple_cnn"
SIMPLE_CNN_M = "simple_cnn_m"
SIMPLE_CNN_M_RES = "simple_cnn_m_res"
SIMPLE_CNN_M_RES_MASK_POOL = "simple_cnn_m_res_mask_pool"
SIMPLE_CNN_M_RES_SE = "simple_cnn_m_res_se"
SIMPLE_CNN_M_RES_DEEPER = "simple_cnn_m_res_deeper"
SIMPLE_CNN_M_MASK_POOL = "simple_cnn_m_mask_pool"
SIMPLE_CNN_L = "simple_cnn_l"
TITLENET = "titlenet"
TITLENET_FAST_A = "titlenet_fast_a"
TITLENET_FAST_B = "titlenet_fast_b"
TITLENET_FAST_C = "titlenet_fast_c"
TITLENET_STUDENT = "titlenet_student"
MASK_AWARE_PALETTE_NET = "mask_aware_palette_net"
MASK_AWARE_TINY_HYBRID_RANKER = "mask_aware_tiny_hybrid_ranker"
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
MODEL_NAMES = (
    RESNET18,
    "resnet34",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny",
    TITLE_HYBRID_TINY,
    TITLE_HYBRID_FAST,
    "swin_tiny",
    FLATTEN_MLP,
    SIMPLE_CNN,
    SIMPLE_CNN_M,
    SIMPLE_CNN_M_RES,
    SIMPLE_CNN_M_RES_MASK_POOL,
    SIMPLE_CNN_M_RES_SE,
    TITLENET,
    TITLENET_NO_SE,
    TITLENET_NO_RESIDUAL,
    TITLENET_NO_FIRST_RESIDUAL,
    TITLENET_NO_MIDDLE_RESIDUAL,
    TITLENET_NO_LAST_RESIDUAL,
    TITLENET_NO_LAST_EXTRA_RESIDUAL,
    TITLENET_NO_STEM,
    TITLENET_NO_STAGE1,
    TITLENET_NO_STAGE2,
    TITLENET_NO_STAGE3,
    TITLENET_ECA,
    TITLENET_NARROW,
    TITLENET_WIDE,
    TITLENET_SHALLOW,
    TITLENET_DEEPER,
    TITLENET_FAST_A,
    TITLENET_FAST_B,
    TITLENET_FAST_C,
    TITLENET_STUDENT,
    SIMPLE_CNN_M_RES_DEEPER,
    SIMPLE_CNN_M_MASK_POOL,
    SIMPLE_CNN_L,
    "mask_aware_cnn",
    "mask_aware_cnn_m",
    MASK_AWARE_PALETTE_NET,
    MASK_AWARE_TINY_HYBRID_RANKER,
)


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def registry_module() -> Any:
    pytest.importorskip("torchvision")
    return pytest.importorskip("src.models.title_color_model_registry")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_title_color_models_accept_four_channel_input(
    model_name: str,
    torch_module: Any,
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(model_name, pretrained=False)
    model.eval()

    with torch_module.no_grad():
        logits = model(torch_module.zeros((1, 4, 36, 136)))

    assert tuple(logits.shape) == (1, 32)


def test_title_color_model_registry_normalizes_aliases(registry_module: Any) -> None:
    assert registry_module.normalize_model_name("ConvNeXt-Tiny") == "convnext_tiny"
    assert registry_module.normalize_model_name("TitleHybridTiny") == TITLE_HYBRID_TINY
    assert registry_module.normalize_model_name("TitleFormer-Lite") == TITLE_HYBRID_TINY
    assert registry_module.normalize_model_name("TitleHybridFast") == TITLE_HYBRID_FAST
    assert registry_module.normalize_model_name("MLP") == FLATTEN_MLP
    assert registry_module.normalize_model_name("SimpleCNN-M") == SIMPLE_CNN_M
    assert registry_module.normalize_model_name("SimpleCNN-M-Res") == SIMPLE_CNN_M_RES
    assert (
        registry_module.normalize_model_name("SimpleCNN-M-Res-Mask-Pool")
        == SIMPLE_CNN_M_RES_MASK_POOL
    )
    assert registry_module.normalize_model_name("SimpleCNN-M-Res-SE") == SIMPLE_CNN_M_RES_SE
    assert registry_module.normalize_model_name("TitLeNet") == TITLENET
    assert registry_module.normalize_model_name("Title-Net") == TITLENET
    assert registry_module.normalize_model_name("TitLeNet-No-SE") == TITLENET_NO_SE
    assert (
        registry_module.normalize_model_name("TitLeNet-No-Residual")
        == TITLENET_NO_RESIDUAL
    )
    assert (
        registry_module.normalize_model_name("TitLeNet-No-Stage1-Residual")
        == TITLENET_NO_FIRST_RESIDUAL
    )
    assert (
        registry_module.normalize_model_name("TitLeNet-No-Stage2-Residual")
        == TITLENET_NO_MIDDLE_RESIDUAL
    )
    assert (
        registry_module.normalize_model_name("TitLeNet-No-Stage3-Residual")
        == TITLENET_NO_LAST_RESIDUAL
    )
    assert registry_module.normalize_model_name("TitLeNet-No-Stem") == TITLENET_NO_STEM
    assert (
        registry_module.normalize_model_name("TitLeNet-Without-Stage1")
        == TITLENET_NO_STAGE1
    )
    assert (
        registry_module.normalize_model_name("TitLeNet-Without-Stage2")
        == TITLENET_NO_STAGE2
    )
    assert (
        registry_module.normalize_model_name("TitLeNet-Without-Stage3")
        == TITLENET_NO_STAGE3
    )
    assert registry_module.normalize_model_name("TitLeNet-With-ECA") == TITLENET_ECA
    assert registry_module.normalize_model_name("TitLeNet-Narrow") == TITLENET_NARROW
    assert registry_module.normalize_model_name("TitLeNet-Wide") == TITLENET_WIDE
    assert registry_module.normalize_model_name("TitLeNet-Shallow") == TITLENET_SHALLOW
    assert registry_module.normalize_model_name("TitLeNet-Deeper") == TITLENET_DEEPER
    assert registry_module.normalize_model_name("TitLeNet-Fast-A") == TITLENET_FAST_A
    assert registry_module.normalize_model_name("TitLeNet-Student") == TITLENET_STUDENT
    assert (
        registry_module.normalize_model_name("TitLeNet-Ablation-Guided-Student")
        == TITLENET_STUDENT
    )
    assert (
        registry_module.normalize_model_name("Mask-Aware-Tiny-Hybrid-Ranker")
        == MASK_AWARE_TINY_HYBRID_RANKER
    )
    assert (
        registry_module.normalize_model_name("SimpleCNN-M-Mask-Pool")
        == SIMPLE_CNN_M_MASK_POOL
    )
    assert MASK_AWARE_PALETTE_NET in registry_module.available_model_names()


def test_simple_cnn_accepts_activation_override(registry_module: Any) -> None:
    model = registry_module.build_title_color_model(
        SIMPLE_CNN_M,
        pretrained=False,
        activation=HARDSWISH_ACTIVATION,
    )

    assert model.activation == HARDSWISH_ACTIVATION


def test_scaled_simple_cnn_variants_increase_capacity(registry_module: Any) -> None:
    small = registry_module.build_title_color_model(SIMPLE_CNN, pretrained=False)
    medium = registry_module.build_title_color_model(SIMPLE_CNN_M, pretrained=False)
    large = registry_module.build_title_color_model(SIMPLE_CNN_L, pretrained=False)

    small_count = sum(parameter.numel() for parameter in small.parameters())
    medium_count = sum(parameter.numel() for parameter in medium.parameters())
    large_count = sum(parameter.numel() for parameter in large.parameters())

    assert small_count < medium_count < large_count


def test_residual_simple_cnn_preserves_medium_feature_shape(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = (
        SIMPLE_CNN_M_RES,
        SIMPLE_CNN_M_RES_SE,
        SIMPLE_CNN_M_RES_DEEPER,
    )
    for model_name in model_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation=GELU_ACTIVATION,
        )

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))

        assert tuple(feature_map.shape) == (1, 160, 5, 17)


def test_title_hybrid_models_use_reduced_cnn_tokens(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = (TITLE_HYBRID_TINY, TITLE_HYBRID_FAST)
    for model_name in model_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation=GELU_ACTIVATION,
        )

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))
            logits = model(torch_module.zeros((1, 4, 36, 136)))

        assert tuple(feature_map.shape[-2:]) == (5, 17)
        expected_token_shape = (85, model.token_norm.normalized_shape[0])
        assert tuple(model.position_embedding.shape[1:]) == expected_token_shape
        assert tuple(logits.shape) == (1, 32)


def test_mask_aware_tiny_hybrid_ranker_scores_palette_candidates(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(
        MASK_AWARE_TINY_HYBRID_RANKER,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    example = torch_module.zeros((2, 4, 36, 136))
    example[:, :3, :, :] = 0.5
    example[:, 3:4, 8:24, 32:96] = 1.0

    with torch_module.no_grad():
        logits = model(example)
        stats = model.mask_statistics(example)
        contrast = model.candidate_contrast_features(example)

    assert tuple(logits.shape) == (2, 32)
    assert tuple(stats.shape) == (2, model.stat_feature_dim)
    assert tuple(contrast.shape) == (2, 32, model.candidate_contrast_dim)
    assert tuple(model.position_embedding.shape[1:]) == (28, 96)


def test_simple_cnn_mask_pool_uses_text_and_background_features(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(
        SIMPLE_CNN_M_MASK_POOL,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    example = torch_module.zeros((1, 4, 36, 136))
    example[:, 3:4, 8:24, 32:96] = 1.0

    with torch_module.no_grad():
        feature_map = model.features(example)
        pooled = model.pooled_features(feature_map, example[:, 3:4, :, :])

    assert tuple(feature_map.shape) == (1, 160, 5, 17)
    assert tuple(pooled.shape) == (1, 640)


def test_residual_simple_cnn_mask_pool_uses_text_and_background_features(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(
        SIMPLE_CNN_M_RES_MASK_POOL,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    example = torch_module.zeros((1, 4, 36, 136))
    example[:, 3:4, 8:24, 32:96] = 1.0

    with torch_module.no_grad():
        feature_map = model.features(example)
        pooled = model.pooled_features(feature_map, example[:, 3:4, :, :])

    assert tuple(feature_map.shape) == (1, 160, 5, 17)
    assert tuple(pooled.shape) == (1, 640)


def test_titlenet_uses_final_residual_se_architecture(registry_module: Any) -> None:
    titlenet = registry_module.build_title_color_model(TITLENET, pretrained=False)
    source_model = registry_module.build_title_color_model(
        SIMPLE_CNN_M_RES_SE,
        pretrained=False,
    )

    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())
    source_count = sum(parameter.numel() for parameter in source_model.parameters())

    assert type(titlenet) is type(source_model)
    assert titlenet_count == source_count


def test_titlenet_fast_variants_are_smaller_than_titlenet(
    torch_module: Any,
    registry_module: Any,
) -> None:
    titlenet = registry_module.build_title_color_model(
        TITLENET,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    fast_names = (TITLENET_FAST_A, TITLENET_FAST_B, TITLENET_FAST_C)
    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())

    for model_name in fast_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation=HARDSWISH_ACTIVATION,
        )

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count < titlenet_count
        assert tuple(feature_map.shape) == (1, 160, 5, 17)


def test_titlenet_student_is_ablation_guided_new_variant(
    torch_module: Any,
    registry_module: Any,
) -> None:
    titlenet = registry_module.build_title_color_model(
        TITLENET,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    student = registry_module.build_title_color_model(
        TITLENET_STUDENT,
        pretrained=False,
        activation=HARDSWISH_ACTIVATION,
    )

    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())
    student_count = sum(parameter.numel() for parameter in student.parameters())

    with torch_module.no_grad():
        feature_map = student.features(torch_module.zeros((1, 4, 36, 136)))
        logits = student(torch_module.zeros((1, 4, 36, 136)))

    assert student_count < titlenet_count
    assert tuple(feature_map.shape) == (1, 128, 5, 17)
    assert tuple(logits.shape) == (1, 32)


def test_titlenet_ablation_variants_have_expected_capacity(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = (
        TITLENET_NO_SE,
        TITLENET_NO_RESIDUAL,
        TITLENET_NO_FIRST_RESIDUAL,
        TITLENET_NO_MIDDLE_RESIDUAL,
        TITLENET_NO_LAST_RESIDUAL,
        TITLENET_NO_LAST_EXTRA_RESIDUAL,
        TITLENET_ECA,
        TITLENET_NARROW,
        TITLENET_WIDE,
        TITLENET_SHALLOW,
        TITLENET_DEEPER,
    )
    models = {
        model_name: registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation=GELU_ACTIVATION,
        )
        for model_name in model_names
    }
    counts = {
        model_name: sum(parameter.numel() for parameter in model.parameters())
        for model_name, model in models.items()
    }
    titlenet = registry_module.build_title_color_model(
        TITLENET,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())

    with torch_module.no_grad():
        narrow_features = models[TITLENET_NARROW].features(
            torch_module.zeros((1, 4, 36, 136))
        )
        wide_features = models[TITLENET_WIDE].features(
            torch_module.zeros((1, 4, 36, 136))
        )

    assert counts[TITLENET_NARROW] < titlenet_count < counts[TITLENET_WIDE]
    assert counts[TITLENET_SHALLOW] < titlenet_count < counts[TITLENET_DEEPER]
    assert counts[TITLENET_NO_SE] < titlenet_count
    assert counts[TITLENET_NO_RESIDUAL] < titlenet_count
    assert counts[TITLENET_NO_FIRST_RESIDUAL] < titlenet_count
    assert counts[TITLENET_NO_MIDDLE_RESIDUAL] < titlenet_count
    assert counts[TITLENET_NO_LAST_RESIDUAL] < titlenet_count
    assert counts[TITLENET_NO_LAST_EXTRA_RESIDUAL] < titlenet_count
    assert counts[TITLENET_ECA] < titlenet_count
    assert tuple(narrow_features.shape) == (1, 128, 5, 17)
    assert tuple(wide_features.shape) == (1, 192, 5, 17)


def test_titlenet_stage_ablation_variants_preserve_feature_shape(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = (
        TITLENET_NO_STEM,
        TITLENET_NO_STAGE1,
        TITLENET_NO_STAGE2,
        TITLENET_NO_STAGE3,
    )
    titlenet = registry_module.build_title_color_model(
        TITLENET,
        pretrained=False,
        activation=GELU_ACTIVATION,
    )
    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())

    for model_name in model_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation=GELU_ACTIVATION,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))
            logits = model(torch_module.zeros((1, 4, 36, 136)))

        assert parameter_count < titlenet_count
        assert tuple(feature_map.shape) == (1, 160, 5, 17)
        assert tuple(logits.shape) == (1, 32)


def test_custom_models_reject_pretrained_flag(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="pretrained"):
        registry_module.build_title_color_model(SIMPLE_CNN, pretrained=True)


def test_pretrained_backbone_rejects_custom_weight_init(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="weight_init"):
        registry_module.build_title_color_model(
            RESNET18,
            pretrained=True,
            weight_init="kaiming_normal",
        )


def test_custom_model_accepts_small_head_initialization(
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(
        SIMPLE_CNN,
        pretrained=False,
        weight_init="small_head",
    )

    final_weight_std = float(model.head[-1].weight.std().item())

    assert final_weight_std < 0.03


def test_unknown_weight_initialization_is_rejected(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="weight_init"):
        registry_module.build_title_color_model(SIMPLE_CNN, weight_init="mystery")


def test_backbone_rejects_custom_activation(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="activation"):
        registry_module.build_title_color_model(RESNET18, activation=HARDSWISH_ACTIVATION)


def test_unknown_activation_is_rejected(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="activation"):
        registry_module.build_title_color_model(SIMPLE_CNN_M, activation="mystery")
