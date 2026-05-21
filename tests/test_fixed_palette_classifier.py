from __future__ import annotations

import logging
from typing import Any

import pytest


@pytest.fixture()
def torch_module() -> Any:
    return pytest.importorskip("torch")


@pytest.fixture()
def nn_module() -> Any:
    return pytest.importorskip("torch.nn")


@pytest.fixture()
def resnet18_builder() -> Any:
    torchvision_models = pytest.importorskip("torchvision.models")
    return torchvision_models.resnet18


@pytest.fixture()
def classifier_module() -> Any:
    return pytest.importorskip("src.models.fixed_palette_classifier")


def _deterministic_resnet18_backbone(
    torch_module: Any,
    resnet18_builder: Any,
) -> Any:
    backbone = resnet18_builder(weights=None)
    original_weight = torch_module.linspace(
        -1.0,
        1.0,
        steps=backbone.conv1.weight.numel(),
        dtype=backbone.conv1.weight.dtype,
    ).view_as(backbone.conv1.weight)
    with torch_module.no_grad():
        backbone.conv1.weight.copy_(original_weight)
    return backbone


def test_conv1_is_expanded_to_four_channels_and_copies_rgb_weights(
    torch_module: Any,
    resnet18_builder: Any,
    classifier_module: Any,
) -> None:
    backbone = _deterministic_resnet18_backbone(
        torch_module,
        resnet18_builder,
    )
    original_weight = backbone.conv1.weight.detach().clone()

    model = classifier_module.FixedPaletteResNet18Classifier(
        pretrained=False,
        backbone=backbone,
    )

    conv1_weight = model.backbone.conv1.weight.detach()
    assert tuple(conv1_weight.shape) == (64, 4, 7, 7)
    assert torch_module.allclose(conv1_weight[:, :3, :, :], original_weight)
    assert torch_module.allclose(
        conv1_weight[:, 3:4, :, :],
        original_weight.mean(dim=1, keepdim=True),
    )


def test_classifier_head_outputs_32_logits(
    torch_module: Any,
    nn_module: Any,
    classifier_module: Any,
) -> None:
    model = classifier_module.build_fixed_palette_resnet18(pretrained=False)
    model.eval()

    with torch_module.no_grad():
        logits = model(
            torch_module.zeros((2, *classifier_module.DEFAULT_INPUT_SHAPE))
        )

    assert tuple(logits.shape) == (2, 32)
    classifier_head = model.backbone.fc
    assert isinstance(classifier_head, nn_module.Sequential)
    assert isinstance(classifier_head[0], nn_module.Linear)
    assert classifier_head[0].in_features == 512
    assert classifier_head[0].out_features == 256
    assert isinstance(classifier_head[1], nn_module.ReLU)
    assert isinstance(classifier_head[2], nn_module.Dropout)
    assert classifier_head[2].p == 0.2
    assert isinstance(classifier_head[3], nn_module.Linear)
    assert classifier_head[3].in_features == 256
    assert classifier_head[3].out_features == 32


def test_log_model_summary_reports_parameter_count_and_output_shape(
    caplog: pytest.LogCaptureFixture,
    classifier_module: Any,
) -> None:
    model = classifier_module.build_fixed_palette_resnet18(pretrained=False)

    with caplog.at_level(logging.INFO):
        summary = classifier_module.log_model_summary(model, batch_size=2)

    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["input_shape"] == (2, 4, 36, 136)
    assert summary["output_shape"] == (2, 32)
    assert "parameters" in caplog.text
    assert "output" in caplog.text
