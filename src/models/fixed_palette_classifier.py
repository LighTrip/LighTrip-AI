from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

LOGGER = logging.getLogger(__name__)
DEFAULT_NUM_CLASSES = 32
DEFAULT_INPUT_SHAPE = (4, 36, 136)


def _resolve_resnet18_weights(
    *,
    pretrained: bool,
    weights: ResNet18_Weights | str | None,
) -> ResNet18_Weights | None:
    if weights is not None:
        if isinstance(weights, str):
            return ResNet18_Weights[weights]
        return weights
    if pretrained:
        return ResNet18_Weights.DEFAULT
    return None


def _build_resnet18_backbone(
    *,
    pretrained: bool,
    weights: ResNet18_Weights | str | None,
) -> nn.Module:
    resolved_weights = _resolve_resnet18_weights(
        pretrained=pretrained,
        weights=weights,
    )
    return resnet18(weights=resolved_weights)


def _make_four_channel_conv1(original_conv: nn.Conv2d) -> nn.Conv2d:
    if original_conv.in_channels != 3:
        raise ValueError(
            "Expected ResNet18 conv1 to have 3 input channels: "
            f"actual={original_conv.in_channels}"
        )

    conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        dilation=original_conv.dilation,
        groups=original_conv.groups,
        bias=original_conv.bias is not None,
        padding_mode=original_conv.padding_mode,
    )
    with torch.no_grad():
        conv1.weight[:, :3, :, :].copy_(original_conv.weight)
        conv1.weight[:, 3:4, :, :].copy_(
            original_conv.weight.mean(dim=1, keepdim=True)
        )
        if original_conv.bias is not None and conv1.bias is not None:
            conv1.bias.copy_(original_conv.bias)
    return conv1


def _make_classifier_head(
    *,
    in_features: int,
    hidden_dim: int,
    dropout: float,
    num_classes: int,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, num_classes),
    )


class FixedPaletteResNet18Classifier(nn.Module):
    """ResNet18 classifier for ROI RGB plus text mask inputs."""

    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
        weights: ResNet18_Weights | str | None = None,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.backbone = backbone or _build_resnet18_backbone(
            pretrained=pretrained,
            weights=weights,
        )
        self.backbone.conv1 = _make_four_channel_conv1(self.backbone.conv1)

        in_features = int(self.backbone.fc.in_features)
        self.backbone.fc = _make_classifier_head(
            in_features=in_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)


def build_fixed_palette_resnet18(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = True,
    weights: ResNet18_Weights | str | None = None,
) -> FixedPaletteResNet18Classifier:
    return FixedPaletteResNet18Classifier(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pretrained=pretrained,
        weights=weights,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def model_summary(
    model: nn.Module,
    *,
    batch_size: int = 2,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    model_device = torch.device(device)
    was_training = model.training
    model = model.to(model_device)
    model.eval()
    with torch.no_grad():
        example = torch.zeros((batch_size, *input_shape), device=model_device)
        output = model(example)
    if was_training:
        model.train()
    return {
        "total_parameters": count_total_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        "input_shape": tuple(example.shape),
        "output_shape": tuple(output.shape),
    }


def log_model_summary(
    model: nn.Module,
    *,
    batch_size: int = 2,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    logger: logging.Logger | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    summary = model_summary(
        model,
        batch_size=batch_size,
        input_shape=input_shape,
        device=device,
    )
    target_logger = logger or LOGGER
    target_logger.info(
        "FixedPaletteResNet18Classifier parameters: total=%s trainable=%s",
        summary["total_parameters"],
        summary["trainable_parameters"],
    )
    target_logger.info(
        "FixedPaletteResNet18Classifier shapes: input=%s output=%s",
        summary["input_shape"],
        summary["output_shape"],
    )
    return summary
