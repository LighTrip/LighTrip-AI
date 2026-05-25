from __future__ import annotations

from typing import Any

import pytest


MODEL_NAMES = (
    "resnet18",
    "resnet34",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny",
    "title_hybrid_tiny",
    "title_hybrid_fast",
    "swin_tiny",
    "flatten_mlp",
    "simple_cnn",
    "simple_cnn_m",
    "simple_cnn_m_res",
    "simple_cnn_m_res_mask_pool",
    "simple_cnn_m_res_se",
    "titlenet",
    "titlenet_fast_a",
    "titlenet_fast_b",
    "titlenet_fast_c",
    "simple_cnn_m_res_deeper",
    "simple_cnn_m_mask_pool",
    "simple_cnn_l",
    "mask_aware_cnn",
    "mask_aware_cnn_m",
    "mask_aware_palette_net",
    "mask_aware_tiny_hybrid_ranker",
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
    assert registry_module.normalize_model_name("TitleHybridTiny") == "title_hybrid_tiny"
    assert registry_module.normalize_model_name("TitleFormer-Lite") == "title_hybrid_tiny"
    assert registry_module.normalize_model_name("TitleHybridFast") == "title_hybrid_fast"
    assert registry_module.normalize_model_name("MLP") == "flatten_mlp"
    assert registry_module.normalize_model_name("SimpleCNN-M") == "simple_cnn_m"
    assert registry_module.normalize_model_name("SimpleCNN-M-Res") == "simple_cnn_m_res"
    assert (
        registry_module.normalize_model_name("SimpleCNN-M-Res-Mask-Pool")
        == "simple_cnn_m_res_mask_pool"
    )
    assert registry_module.normalize_model_name("SimpleCNN-M-Res-SE") == "simple_cnn_m_res_se"
    assert registry_module.normalize_model_name("TitLeNet") == "titlenet"
    assert registry_module.normalize_model_name("Title-Net") == "titlenet"
    assert registry_module.normalize_model_name("TitLeNet-Fast-A") == "titlenet_fast_a"
    assert (
        registry_module.normalize_model_name("Mask-Aware-Tiny-Hybrid-Ranker")
        == "mask_aware_tiny_hybrid_ranker"
    )
    assert (
        registry_module.normalize_model_name("SimpleCNN-M-Mask-Pool")
        == "simple_cnn_m_mask_pool"
    )
    assert "mask_aware_palette_net" in registry_module.available_model_names()


def test_simple_cnn_accepts_activation_override(registry_module: Any) -> None:
    model = registry_module.build_title_color_model(
        "simple_cnn_m",
        pretrained=False,
        activation="hardswish",
    )

    assert model.activation == "hardswish"


def test_scaled_simple_cnn_variants_increase_capacity(registry_module: Any) -> None:
    small = registry_module.build_title_color_model("simple_cnn", pretrained=False)
    medium = registry_module.build_title_color_model("simple_cnn_m", pretrained=False)
    large = registry_module.build_title_color_model("simple_cnn_l", pretrained=False)

    small_count = sum(parameter.numel() for parameter in small.parameters())
    medium_count = sum(parameter.numel() for parameter in medium.parameters())
    large_count = sum(parameter.numel() for parameter in large.parameters())

    assert small_count < medium_count < large_count


def test_residual_simple_cnn_preserves_medium_feature_shape(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = (
        "simple_cnn_m_res",
        "simple_cnn_m_res_se",
        "simple_cnn_m_res_deeper",
    )
    for model_name in model_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation="gelu",
        )

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))

        assert tuple(feature_map.shape) == (1, 160, 5, 17)


def test_title_hybrid_models_use_reduced_cnn_tokens(
    torch_module: Any,
    registry_module: Any,
) -> None:
    model_names = ("title_hybrid_tiny", "title_hybrid_fast")
    for model_name in model_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation="gelu",
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
        "mask_aware_tiny_hybrid_ranker",
        pretrained=False,
        activation="gelu",
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
        "simple_cnn_m_mask_pool",
        pretrained=False,
        activation="gelu",
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
        "simple_cnn_m_res_mask_pool",
        pretrained=False,
        activation="gelu",
    )
    example = torch_module.zeros((1, 4, 36, 136))
    example[:, 3:4, 8:24, 32:96] = 1.0

    with torch_module.no_grad():
        feature_map = model.features(example)
        pooled = model.pooled_features(feature_map, example[:, 3:4, :, :])

    assert tuple(feature_map.shape) == (1, 160, 5, 17)
    assert tuple(pooled.shape) == (1, 640)


def test_titlenet_uses_final_residual_se_architecture(registry_module: Any) -> None:
    titlenet = registry_module.build_title_color_model("titlenet", pretrained=False)
    source_model = registry_module.build_title_color_model(
        "simple_cnn_m_res_se",
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
        "titlenet",
        pretrained=False,
        activation="gelu",
    )
    fast_names = ("titlenet_fast_a", "titlenet_fast_b", "titlenet_fast_c")
    titlenet_count = sum(parameter.numel() for parameter in titlenet.parameters())

    for model_name in fast_names:
        model = registry_module.build_title_color_model(
            model_name,
            pretrained=False,
            activation="hardswish",
        )

        with torch_module.no_grad():
            feature_map = model.features(torch_module.zeros((1, 4, 36, 136)))

        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        assert parameter_count < titlenet_count
        assert tuple(feature_map.shape) == (1, 160, 5, 17)


def test_custom_models_reject_pretrained_flag(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="pretrained"):
        registry_module.build_title_color_model("simple_cnn", pretrained=True)


def test_pretrained_backbone_rejects_custom_weight_init(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="weight_init"):
        registry_module.build_title_color_model(
            "resnet18",
            pretrained=True,
            weight_init="kaiming_normal",
        )


def test_custom_model_accepts_small_head_initialization(
    registry_module: Any,
) -> None:
    model = registry_module.build_title_color_model(
        "simple_cnn",
        pretrained=False,
        weight_init="small_head",
    )

    final_weight_std = float(model.head[-1].weight.std().item())

    assert final_weight_std < 0.03


def test_unknown_weight_initialization_is_rejected(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="weight_init"):
        registry_module.build_title_color_model("simple_cnn", weight_init="mystery")


def test_backbone_rejects_custom_activation(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="activation"):
        registry_module.build_title_color_model("resnet18", activation="hardswish")


def test_unknown_activation_is_rejected(registry_module: Any) -> None:
    with pytest.raises(ValueError, match="activation"):
        registry_module.build_title_color_model("simple_cnn_m", activation="mystery")
