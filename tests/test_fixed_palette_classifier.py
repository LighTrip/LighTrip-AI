from __future__ import annotations

import importlib.util
import logging

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None

if TORCH_AVAILABLE and TORCHVISION_AVAILABLE:
    import torch
    from torch import nn
    from torchvision.models import resnet18

    from src.models.fixed_palette_classifier import (
        DEFAULT_INPUT_SHAPE,
        FixedPaletteResNet18Classifier,
        build_fixed_palette_resnet18,
        log_model_summary,
    )
else:
    torch = None
    nn = None
    resnet18 = None
    DEFAULT_INPUT_SHAPE = None
    FixedPaletteResNet18Classifier = None
    build_fixed_palette_resnet18 = None
    log_model_summary = None

pytestmark = pytest.mark.skipif(
    not (TORCH_AVAILABLE and TORCHVISION_AVAILABLE),
    reason="PyTorch and torchvision are required for classifier tests.",
)


def _deterministic_resnet18_backbone() -> nn.Module:
    backbone = resnet18(weights=None)
    original_weight = torch.linspace(
        -1.0,
        1.0,
        steps=backbone.conv1.weight.numel(),
        dtype=backbone.conv1.weight.dtype,
    ).view_as(backbone.conv1.weight)
    with torch.no_grad():
        backbone.conv1.weight.copy_(original_weight)
    return backbone


def test_conv1_is_expanded_to_four_channels_and_copies_rgb_weights() -> None:
    backbone = _deterministic_resnet18_backbone()
    original_weight = backbone.conv1.weight.detach().clone()

    model = FixedPaletteResNet18Classifier(
        pretrained=False,
        backbone=backbone,
    )

    conv1_weight = model.backbone.conv1.weight.detach()
    assert tuple(conv1_weight.shape) == (64, 4, 7, 7)
    assert torch.allclose(conv1_weight[:, :3, :, :], original_weight)
    assert torch.allclose(
        conv1_weight[:, 3:4, :, :],
        original_weight.mean(dim=1, keepdim=True),
    )


def test_classifier_head_outputs_32_logits() -> None:
    model = build_fixed_palette_resnet18(pretrained=False)
    model.eval()

    with torch.no_grad():
        logits = model(torch.zeros((2, *DEFAULT_INPUT_SHAPE)))

    assert tuple(logits.shape) == (2, 32)
    assert isinstance(model.backbone.fc, nn.Sequential)
    assert isinstance(model.backbone.fc[0], nn.Linear)
    assert model.backbone.fc[0].in_features == 512
    assert model.backbone.fc[0].out_features == 256
    assert isinstance(model.backbone.fc[1], nn.ReLU)
    assert isinstance(model.backbone.fc[2], nn.Dropout)
    assert model.backbone.fc[2].p == 0.2
    assert isinstance(model.backbone.fc[3], nn.Linear)
    assert model.backbone.fc[3].in_features == 256
    assert model.backbone.fc[3].out_features == 32


def test_log_model_summary_reports_parameter_count_and_output_shape(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = build_fixed_palette_resnet18(pretrained=False)

    with caplog.at_level(logging.INFO):
        summary = log_model_summary(model, batch_size=2)

    assert summary["total_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["input_shape"] == (2, 4, 36, 136)
    assert summary["output_shape"] == (2, 32)
    assert "parameters" in caplog.text
    assert "output" in caplog.text
