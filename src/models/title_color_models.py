from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    ResNet34_Weights,
    Swin_T_Weights,
    convnext_tiny,
    efficientnet_b0,
    resnet34,
    swin_t,
)
from torchvision.models.resnet import ResNet

from src.models.fixed_palette_classifier import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    make_four_channel_conv1,
)


DEFAULT_PALETTE_PATH = Path("data/title_color_recommendation/processed/palette.json")
PALETTE_GROUPS = (
    "neutral_light",
    "neutral_dark",
    "cream",
    "pastel",
    "accent",
    "deep",
    "muted",
    "other",
)
PALETTE_FEATURE_DIM = 16
ACTIVATION_SILU = "silu"
ACTIVATION_RELU = "relu"
ACTIVATION_GELU = "gelu"
ACTIVATION_HARDSWISH = "hardswish"
ACTIVATION_LEAKY_RELU = "leaky_relu"
DEFAULT_ACTIVATION = ACTIVATION_SILU
ODD_KERNEL_ERROR = "kernel_size must be odd: {kernel_size}"
CHANNELS_MIN_ERROR = "channels must contain at least two entries"
EMPTY_ABLATED_PARTS: frozenset[str] = frozenset()
ACTIVATION_ALIASES = {
    "swish": ACTIVATION_SILU,
    "hard_swish": ACTIVATION_HARDSWISH,
    "hard-swish": ACTIVATION_HARDSWISH,
    "leaky-relu": ACTIVATION_LEAKY_RELU,
    "leakyrelu": ACTIVATION_LEAKY_RELU,
}
ACTIVATION_NAMES = (
    ACTIVATION_SILU,
    ACTIVATION_RELU,
    ACTIVATION_GELU,
    ACTIVATION_HARDSWISH,
    ACTIVATION_LEAKY_RELU,
)


def normalize_activation_name(name: str | None) -> str:
    if name is None:
        return DEFAULT_ACTIVATION
    normalized = name.strip().lower().replace("-", "_")
    normalized = ACTIVATION_ALIASES.get(normalized, normalized)
    if normalized not in ACTIVATION_NAMES:
        available = ", ".join(ACTIVATION_NAMES)
        raise ValueError(f"unknown activation={name!r}; available={available}")
    return normalized


def make_activation(name: str | None = DEFAULT_ACTIVATION) -> nn.Module:
    normalized = normalize_activation_name(name)
    if normalized == ACTIVATION_RELU:
        return nn.ReLU(inplace=True)
    if normalized == ACTIVATION_GELU:
        return nn.GELU()
    if normalized == ACTIVATION_HARDSWISH:
        return nn.Hardswish(inplace=True)
    if normalized == ACTIVATION_LEAKY_RELU:
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return nn.SiLU(inplace=True)


def resolve_torchvision_weights(
    weights_enum: Any,
    *,
    pretrained: bool,
    weights: Any,
) -> Any:
    if weights is not None:
        return weights_enum.verify(weights)
    if pretrained:
        return weights_enum.DEFAULT
    return None


def make_classifier_head(
    *,
    in_features: int,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _linear_in_features(layer: nn.Module, *, description: str) -> int:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"{description} must be nn.Linear: {type(layer).__name__}")
    return int(layer.in_features)


def build_resnet34_classifier(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = False,
    weights: ResNet34_Weights | str | None = None,
) -> ResNet:
    model = resnet34(
        weights=resolve_torchvision_weights(
            ResNet34_Weights,
            pretrained=pretrained,
            weights=weights,
        )
    )
    model.conv1 = make_four_channel_conv1(model.conv1)
    model.fc = make_classifier_head(
        in_features=_linear_in_features(model.fc, description="resnet34.fc"),
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model


def build_efficientnet_b0_classifier(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = False,
    weights: EfficientNet_B0_Weights | str | None = None,
) -> nn.Module:
    model = efficientnet_b0(
        weights=resolve_torchvision_weights(
            EfficientNet_B0_Weights,
            pretrained=pretrained,
            weights=weights,
        )
    )
    model.features[0][0] = make_four_channel_conv1(model.features[0][0])
    in_features = _linear_in_features(
        model.classifier[1],
        description="efficientnet_b0.classifier[1]",
    )
    model.classifier = make_classifier_head(
        in_features=in_features,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model


def build_convnext_tiny_classifier(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = False,
    weights: ConvNeXt_Tiny_Weights | str | None = None,
) -> nn.Module:
    model = convnext_tiny(
        weights=resolve_torchvision_weights(
            ConvNeXt_Tiny_Weights,
            pretrained=pretrained,
            weights=weights,
        )
    )
    model.features[0][0] = make_four_channel_conv1(model.features[0][0])
    in_features = _linear_in_features(
        model.classifier[2],
        description="convnext_tiny.classifier[2]",
    )
    model.classifier[2] = make_classifier_head(
        in_features=in_features,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model


def build_swin_tiny_classifier(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    hidden_dim: int = 256,
    dropout: float = 0.2,
    pretrained: bool = False,
    weights: Swin_T_Weights | str | None = None,
) -> nn.Module:
    model = swin_t(
        weights=resolve_torchvision_weights(
            Swin_T_Weights,
            pretrained=pretrained,
            weights=weights,
        )
    )
    model.features[0][0] = make_four_channel_conv1(model.features[0][0])
    model.head = make_classifier_head(
        in_features=_linear_in_features(model.head, description="swin_t.head"),
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
    )
    return model


class FlattenMLP(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_dim = int(input_shape[0] * input_shape[1] * input_shape[2])
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dims[1], num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        kernel_size: int = 3,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(ODD_KERNEL_ERROR.format(kernel_size=kernel_size))
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            make_activation(activation),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class PointwiseProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, *, reduction: int = 8) -> None:
        super().__init__()
        hidden_channels = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gate(x)


class EfficientChannelAttention(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(ODD_KERNEL_ERROR.format(kernel_size=kernel_size))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.pool(x).flatten(2).transpose(1, 2)
        weights = self.gate(self.conv(pooled)).transpose(1, 2).view(
            x.shape[0],
            x.shape[1],
            1,
            1,
        )
        return x * weights


class ResidualDepthwiseSeparableBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        use_se: bool = False,
        use_eca: bool = False,
        se_reduction: int = 8,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if use_se and use_eca:
            raise ValueError("use_se and use_eca cannot both be enabled")
        if kernel_size % 2 != 1:
            raise ValueError(ODD_KERNEL_ERROR.format(kernel_size=kernel_size))
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            make_activation(activation),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        if use_se:
            self.attention: nn.Module = SqueezeExcitation(
                channels,
                reduction=se_reduction,
            )
        elif use_eca:
            self.attention = EfficientChannelAttention(channels)
        else:
            self.attention = nn.Identity()
        self.output_activation = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_activation(x + self.attention(self.block(x)))


class LightweightFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        channels: tuple[int, ...] = (32, 64, 96, 128),
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(CHANNELS_MIN_ERROR)
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            make_activation(activation),
        ]
        for input_channels, output_channels in zip(channels, channels[1:]):
            layers.append(
                DepthwiseSeparableConv(
                    input_channels,
                    output_channels,
                    stride=2,
                    activation=activation,
                )
            )
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualLightweightFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        channels: tuple[int, ...] = (48, 96, 128, 160),
        residual_blocks: tuple[int, ...] = (1, 1, 2),
        use_se: bool = False,
        use_eca: bool = False,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(CHANNELS_MIN_ERROR)
        if len(residual_blocks) != len(channels) - 1:
            raise ValueError("residual_blocks must match downsample stages")

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            make_activation(activation),
        ]
        for input_channels, output_channels, block_count in zip(
            channels,
            channels[1:],
            residual_blocks,
        ):
            layers.append(
                DepthwiseSeparableConv(
                    input_channels,
                    output_channels,
                    stride=2,
                    activation=activation,
                )
            )
            for _ in range(block_count):
                layers.append(
                    ResidualDepthwiseSeparableBlock(
                        output_channels,
                        use_se=use_se,
                        use_eca=use_eca,
                        activation=activation,
                    )
                )
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StageAblatedResidualFeatureExtractor(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 4,
        channels: tuple[int, ...] = (48, 96, 128, 160),
        residual_blocks: tuple[int, ...] = (1, 1, 2),
        ablated_parts: frozenset[str] = EMPTY_ABLATED_PARTS,
        use_se: bool = True,
        use_eca: bool = False,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError(CHANNELS_MIN_ERROR)
        if len(residual_blocks) != len(channels) - 1:
            raise ValueError("residual_blocks must match downsample stages")

        layers: list[nn.Module] = [
            self._stem_layer(
                in_channels,
                channels[0],
                ablated_parts=ablated_parts,
                activation=activation,
            )
        ]
        for stage_index, (input_channels, output_channels, block_count) in enumerate(
            zip(channels, channels[1:], residual_blocks),
            start=1,
        ):
            stage_name = f"stage{stage_index}"
            if stage_name in ablated_parts:
                layers.append(
                    PointwiseProjection(
                        input_channels,
                        output_channels,
                        stride=2,
                        activation=activation,
                    )
                )
                continue
            layers.append(
                DepthwiseSeparableConv(
                    input_channels,
                    output_channels,
                    stride=2,
                    activation=activation,
                )
            )
            layers.extend(
                ResidualDepthwiseSeparableBlock(
                    output_channels,
                    use_se=use_se,
                    use_eca=use_eca,
                    activation=activation,
                )
                for _index in range(block_count)
            )
        self.net = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    @staticmethod
    def _stem_layer(
        in_channels: int,
        out_channels: int,
        *,
        ablated_parts: frozenset[str],
        activation: str,
    ) -> nn.Module:
        if "stem" in ablated_parts:
            return PointwiseProjection(
                in_channels,
                out_channels,
                activation=activation,
            )
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            make_activation(activation),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def masked_average_pool(features: Tensor, mask: Tensor, *, eps: float = 1e-6) -> Tensor:
    resized_mask = F.interpolate(
        mask,
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
    numerator = (features * resized_mask).sum(dim=(-2, -1))
    denominator = resized_mask.sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


def average_pool_with_weights(
    features: Tensor,
    weights: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    numerator = (features * weights).sum(dim=(-2, -1))
    denominator = weights.sum(dim=(-2, -1)).clamp_min(eps)
    return numerator / denominator


def resize_mask_like(mask: Tensor, features: Tensor) -> Tensor:
    return F.interpolate(
        mask,
        size=features.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def ring_from_mask(mask: Tensor, *, kernel_size: int = 3) -> Tensor:
    padding = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
    return (dilated - mask).clamp(0.0, 1.0)


def text_background_difference_pool(features: Tensor, mask: Tensor) -> Tensor:
    resized_mask = resize_mask_like(mask, features)
    background_mask = 1.0 - resized_mask
    global_features = features.mean(dim=(-2, -1))
    text_features = average_pool_with_weights(features, resized_mask)
    background_features = average_pool_with_weights(features, background_mask)
    difference_features = text_features - background_features
    return torch.cat(
        (
            global_features,
            text_features,
            background_features,
            difference_features,
        ),
        dim=1,
    )


class SimpleCNN(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 128,
        feature_channels: tuple[int, ...] = (32, 64, 96, 128),
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.features = LightweightFeatureExtractor(
            channels=feature_channels,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class ResidualSimpleCNN(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 192,
        feature_channels: tuple[int, ...] = (48, 96, 128, 160),
        residual_blocks: tuple[int, ...] = (1, 1, 2),
        use_se: bool = False,
        use_eca: bool = False,
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.features = ResidualLightweightFeatureExtractor(
            channels=feature_channels,
            residual_blocks=residual_blocks,
            use_se=use_se,
            use_eca=use_eca,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class StageAblatedTitLeNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 192,
        feature_channels: tuple[int, ...] = (48, 96, 128, 160),
        residual_blocks: tuple[int, ...] = (1, 1, 2),
        ablated_parts: frozenset[str] = EMPTY_ABLATED_PARTS,
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.ablated_parts = ablated_parts
        self.features = StageAblatedResidualFeatureExtractor(
            channels=feature_channels,
            residual_blocks=residual_blocks,
            ablated_parts=ablated_parts,
            use_se=True,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


class MaskAwareCNN(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 192,
        feature_channels: tuple[int, ...] = (32, 64, 96, 128),
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.features = LightweightFeatureExtractor(
            channels=feature_channels,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        mask = x[:, 3:4, :, :]
        global_features = features.mean(dim=(-2, -1))
        mask_features = masked_average_pool(features, mask)
        return self.head(torch.cat((global_features, mask_features), dim=1))


class SimpleCNNMaskPool(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 256,
        feature_channels: tuple[int, ...] = (48, 96, 128, 160),
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.features = LightweightFeatureExtractor(
            channels=feature_channels,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def pooled_features(self, features: Tensor, mask: Tensor) -> Tensor:
        return text_background_difference_pool(features, mask)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        pooled = self.pooled_features(features, x[:, 3:4, :, :])
        return self.head(pooled)


class ResidualSimpleCNNMaskPool(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        hidden_dim: int = 256,
        feature_channels: tuple[int, ...] = (48, 96, 128, 160),
        residual_blocks: tuple[int, ...] = (1, 1, 2),
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        self.activation = normalize_activation_name(activation)
        self.features = ResidualLightweightFeatureExtractor(
            channels=feature_channels,
            residual_blocks=residual_blocks,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def pooled_features(self, features: Tensor, mask: Tensor) -> Tensor:
        return text_background_difference_pool(features, mask)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        pooled = self.pooled_features(features, x[:, 3:4, :, :])
        return self.head(pooled)


def srgb_to_linear_tensor(values: Tensor) -> Tensor:
    return torch.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055).pow(2.4),
    )


def relative_luminance_tensor(rgb: Tensor) -> Tensor:
    linear = srgb_to_linear_tensor(rgb)
    return (
        (0.2126 * linear[:, 0, :, :])
        + (0.7152 * linear[:, 1, :, :])
        + (0.0722 * linear[:, 2, :, :])
    )


def rgb_to_lab_tensor(rgb: Tensor) -> Tensor:
    linear = srgb_to_linear_tensor(rgb.clamp(0.0, 1.0))
    red = linear[:, 0, :, :]
    green = linear[:, 1, :, :]
    blue = linear[:, 2, :, :]
    x_value = (0.4124564 * red) + (0.3575761 * green) + (0.1804375 * blue)
    y_value = (0.2126729 * red) + (0.7151522 * green) + (0.0721750 * blue)
    z_value = (0.0193339 * red) + (0.1191920 * green) + (0.9503041 * blue)
    x_scaled = x_value / 0.95047
    z_scaled = z_value / 1.08883
    fx_value = lab_pivot_tensor(x_scaled)
    fy_value = lab_pivot_tensor(y_value)
    fz_value = lab_pivot_tensor(z_scaled)
    lightness = ((116.0 * fy_value) - 16.0) / 100.0
    a_channel = (500.0 * (fx_value - fy_value)) / 128.0
    b_channel = (200.0 * (fy_value - fz_value)) / 128.0
    return torch.stack((lightness, a_channel, b_channel), dim=1)


def lab_pivot_tensor(values: Tensor) -> Tensor:
    return torch.where(
        values > 0.008856,
        values.clamp_min(0.0).pow(1.0 / 3.0),
        (7.787 * values) + (16.0 / 116.0),
    )


def contrast_ratio(candidate_luminance: Tensor, background_luminance: Tensor) -> Tensor:
    lighter = torch.maximum(candidate_luminance, background_luminance)
    darker = torch.minimum(candidate_luminance, background_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def weighted_channel_mean_std(
    values: Tensor,
    weights: Tensor,
    *,
    eps: float = 1e-6,
) -> Tensor:
    denominator = weights.sum(dim=(-2, -1)).clamp_min(eps)
    mean = (values * weights).sum(dim=(-2, -1)) / denominator
    centered = values - mean[:, :, None, None]
    variance = (centered.square() * weights).sum(dim=(-2, -1)) / denominator
    return torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=1)


def luminance_texture_tensor(luminance: Tensor) -> Tensor:
    horizontal = F.pad(
        (luminance[:, :, :, 1:] - luminance[:, :, :, :-1]).abs(),
        (0, 1, 0, 0),
    )
    vertical = F.pad(
        (luminance[:, :, 1:, :] - luminance[:, :, :-1, :]).abs(),
        (0, 0, 0, 1),
    )
    return 0.5 * (horizontal + vertical)


def empty_palette_features(num_classes: int) -> Tensor:
    return torch.zeros(num_classes, PALETTE_FEATURE_DIM, dtype=torch.float32)


def load_palette_features(
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Tensor:
    path = Path(palette_path)
    if not path.exists():
        return empty_palette_features(num_classes)

    payload = json.loads(path.read_text(encoding="utf-8"))
    features = empty_palette_features(num_classes)
    for item in payload:
        palette_id = int(item["id"])
        if not 0 <= palette_id < num_classes:
            continue
        rgb = [float(channel) / 255.0 for channel in item["rgb"]]
        lab = [
            float(item["lab"][0]) / 100.0,
            float(item["lab"][1]) / 128.0,
            float(item["lab"][2]) / 128.0,
        ]
        group = str(item.get("group", "other"))
        group_index = PALETTE_GROUPS.index(group) if group in PALETTE_GROUPS else 7
        row = torch.zeros(PALETTE_FEATURE_DIM, dtype=torch.float32)
        row[0:3] = torch.tensor(rgb, dtype=torch.float32)
        row[3:6] = torch.tensor(lab, dtype=torch.float32)
        row[6] = float(item["relative_luminance"])
        row[7] = float(item["aesthetic_prior"])
        row[8 + group_index] = 1.0
        features[palette_id] = row
    return features


class MaskAwarePaletteNet(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        palette_features: Tensor | None = None,
        image_dim: int = 192,
        palette_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        features = (
            empty_palette_features(num_classes)
            if palette_features is None
            else palette_features.float()
        )
        if tuple(features.shape) != (num_classes, PALETTE_FEATURE_DIM):
            raise ValueError(
                "palette_features must have shape "
                f"({num_classes}, {PALETTE_FEATURE_DIM}): {tuple(features.shape)}"
            )
        self.num_classes = num_classes
        self.features = LightweightFeatureExtractor(channels=(32, 64, 96, 128))
        self.image_projection = nn.Sequential(
            nn.Linear(256, image_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.palette_encoder = nn.Sequential(
            nn.Linear(PALETTE_FEATURE_DIM, palette_dim),
            nn.SiLU(inplace=True),
            nn.Linear(palette_dim, palette_dim),
            nn.SiLU(inplace=True),
        )
        self.scorer = nn.Sequential(
            nn.Linear(image_dim + palette_dim + 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("palette_features", features)
        self.register_buffer("palette_luminance", features[:, 6].clone())

    def contrast_features(self, x: Tensor) -> Tensor:
        rgb = x[:, :3, :, :].clamp(0.0, 1.0)
        mask = x[:, 3:4, :, :].clamp(0.0, 1.0)
        luminance = relative_luminance_tensor(rgb)
        mask_values = mask[:, 0, :, :]
        masked_mean = (
            (luminance * mask_values).sum(dim=(-2, -1))
            / mask_values.sum(dim=(-2, -1)).clamp_min(1e-6)
        )
        full_mean = luminance.mean(dim=(-2, -1))
        candidate = self.palette_luminance.view(1, self.num_classes)
        masked = masked_mean.view(-1, 1)
        full = full_mean.view(-1, 1)
        masked_contrast = contrast_ratio(candidate, masked)
        full_contrast = contrast_ratio(candidate, full)
        luminance_delta = (candidate - masked).abs()
        wcag_proxy = (masked_contrast >= 4.5).float()
        return torch.stack(
            (
                masked_contrast / 21.0,
                full_contrast / 21.0,
                luminance_delta,
                wcag_proxy,
            ),
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        mask = x[:, 3:4, :, :]
        global_features = features.mean(dim=(-2, -1))
        mask_features = masked_average_pool(features, mask)
        image_embedding = self.image_projection(
            torch.cat((global_features, mask_features), dim=1)
        )
        palette_embedding = self.palette_encoder(self.palette_features)
        batch_size = int(x.shape[0])
        image = image_embedding[:, None, :].expand(-1, self.num_classes, -1)
        palette = palette_embedding[None, :, :].expand(batch_size, -1, -1)
        contrast = self.contrast_features(x)
        return self.scorer(torch.cat((image, palette, contrast), dim=-1)).squeeze(-1)


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        patch_size: tuple[int, int] = (4, 8),
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 384,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        channels, height, width = input_shape
        patch_height, patch_width = patch_size
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                f"input_shape {input_shape} must be divisible by patch_size {patch_size}"
            )
        token_count = (height // patch_height) * (width // patch_width)
        self.patch_embed = nn.Conv2d(
            channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=ACTIVATION_GELU,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


def downsampled_feature_size(size: int, *, stride2_stages: int) -> int:
    output_size = size
    for _index in range(stride2_stages):
        output_size = (output_size + 1) // 2
    return output_size


class TitleHybridTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        feature_channels: tuple[int, ...] = (32, 64, 96, 128),
        residual_blocks: tuple[int, ...] = (0, 1, 1),
        embed_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads: {embed_dim}, {num_heads}"
            )
        self.activation = normalize_activation_name(activation)
        self.features = ResidualLightweightFeatureExtractor(
            channels=feature_channels,
            residual_blocks=residual_blocks,
            activation=self.activation,
        )
        feature_dim = self.features.out_channels
        stride2_stages = len(feature_channels) - 1
        _channels, height, width = input_shape
        token_height = downsampled_feature_size(height, stride2_stages=stride2_stages)
        token_width = downsampled_feature_size(width, stride2_stages=stride2_stages)
        token_count = token_height * token_width

        self.token_projection = nn.Sequential(
            nn.Conv2d(feature_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            make_activation(self.activation),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=ACTIVATION_GELU,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.mask_projection = nn.Sequential(
            nn.Linear(feature_dim * 4, embed_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        projected = self.token_projection(features)
        tokens = projected.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embedding
        encoded = self.token_norm(self.encoder(tokens))
        transformer_features = encoded.mean(dim=1)
        mask_features = self.mask_projection(
            text_background_difference_pool(features, x[:, 3:4, :, :])
        )
        return self.head(torch.cat((transformer_features, mask_features), dim=1))


@dataclass(frozen=True)
class MaskAwareTinyHybridRankerConfig:
    stem_dim: int = 32
    embed_dim: int = 96
    depth: int = 4
    num_heads: int = 4
    mlp_dim: int = 192
    image_dim: int = 96
    stat_dim: int = 48
    candidate_dim: int = 48
    hidden_dim: int = 96
    patch_size: tuple[int, int] = (12, 16)


class MaskAwareTinyHybridColorRanker(nn.Module):
    stat_feature_dim = 39
    candidate_contrast_dim = 6

    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
        palette_features: Tensor | None = None,
        architecture: MaskAwareTinyHybridRankerConfig | None = None,
        dropout: float = 0.2,
        activation: str = DEFAULT_ACTIVATION,
    ) -> None:
        super().__init__()
        config = architecture or MaskAwareTinyHybridRankerConfig()
        stem_dim = config.stem_dim
        embed_dim = config.embed_dim
        depth = config.depth
        num_heads = config.num_heads
        mlp_dim = config.mlp_dim
        image_dim = config.image_dim
        stat_dim = config.stat_dim
        candidate_dim = config.candidate_dim
        hidden_dim = config.hidden_dim
        patch_size = config.patch_size
        features = (
            empty_palette_features(num_classes)
            if palette_features is None
            else palette_features.float()
        )
        if tuple(features.shape) != (num_classes, PALETTE_FEATURE_DIM):
            raise ValueError(
                "palette_features must have shape "
                f"({num_classes}, {PALETTE_FEATURE_DIM}): {tuple(features.shape)}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads: {embed_dim}, {num_heads}"
            )

        self.num_classes = num_classes
        self.activation = normalize_activation_name(activation)
        self.patch_size = patch_size
        _channels, height, width = input_shape
        patch_height, patch_width = patch_size
        token_height = (height + patch_height - 1) // patch_height
        token_width = (width + patch_width - 1) // patch_width
        token_count = token_height * token_width

        self.stem = nn.Sequential(
            nn.Conv2d(4, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            make_activation(self.activation),
            DepthwiseSeparableConv(
                stem_dim,
                stem_dim,
                stride=1,
                activation=self.activation,
            ),
        )
        self.patch_embed = nn.Conv2d(
            stem_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, token_count + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation=ACTIVATION_GELU,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.visual_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, image_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
        )
        self.stat_encoder = nn.Sequential(
            nn.Linear(self.stat_feature_dim, stat_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(stat_dim, stat_dim),
            make_activation(self.activation),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(image_dim + stat_dim, image_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(PALETTE_FEATURE_DIM, candidate_dim),
            make_activation(self.activation),
            nn.Linear(candidate_dim, candidate_dim),
            make_activation(self.activation),
        )
        self.scorer = nn.Sequential(
            nn.Linear(image_dim + candidate_dim + self.candidate_contrast_dim, hidden_dim),
            make_activation(self.activation),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("palette_features", features)
        self.register_buffer("palette_luminance", features[:, 6].clone())
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def padded_stem_features(self, x: Tensor) -> Tensor:
        features = self.stem(x)
        patch_height, patch_width = self.patch_size
        height_padding = (-features.shape[-2]) % patch_height
        width_padding = (-features.shape[-1]) % patch_width
        if height_padding == 0 and width_padding == 0:
            return features
        return F.pad(features, (0, width_padding, 0, height_padding))

    def visual_features(self, x: Tensor) -> Tensor:
        patches = self.patch_embed(self.padded_stem_features(x))
        tokens = patches.flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        encoded = self.token_norm(self.encoder(tokens + self.position_embedding))
        pooled = torch.cat((encoded[:, 0], encoded[:, 1:].mean(dim=1)), dim=1)
        return self.visual_projection(pooled)

    def mask_statistics(self, x: Tensor) -> Tensor:
        rgb = x[:, :3, :, :].clamp(0.0, 1.0)
        mask = x[:, 3:4, :, :].clamp(0.0, 1.0)
        background_mask = 1.0 - mask
        ring_mask = ring_from_mask(mask)
        lab = rgb_to_lab_tensor(rgb)
        luminance = relative_luminance_tensor(rgb).unsqueeze(1)
        texture = luminance_texture_tensor(luminance)

        masked_luminance = weighted_channel_mean_std(luminance, mask)
        background_luminance = weighted_channel_mean_std(luminance, background_mask)
        text_mean = masked_luminance[:, 0]
        background_mean = background_luminance[:, 0]
        text_background_contrast = contrast_ratio(text_mean, background_mean).unsqueeze(1)
        luminance_delta = (text_mean - background_mean).abs().unsqueeze(1)
        mask_coverage = mask.mean(dim=(-2, -1))

        return torch.cat(
            (
                mask_coverage,
                weighted_channel_mean_std(rgb, mask),
                weighted_channel_mean_std(rgb, background_mask),
                weighted_channel_mean_std(lab, mask),
                weighted_channel_mean_std(lab, background_mask),
                masked_luminance,
                background_luminance,
                weighted_channel_mean_std(luminance, torch.ones_like(mask)),
                weighted_channel_mean_std(luminance, ring_mask),
                weighted_channel_mean_std(texture, mask),
                weighted_channel_mean_std(texture, torch.ones_like(mask)),
                text_background_contrast / 21.0,
                luminance_delta,
            ),
            dim=1,
        )

    def candidate_contrast_features(self, x: Tensor) -> Tensor:
        rgb = x[:, :3, :, :].clamp(0.0, 1.0)
        mask = x[:, 3:4, :, :].clamp(0.0, 1.0)
        background_mask = 1.0 - mask
        luminance = relative_luminance_tensor(rgb).unsqueeze(1)
        text_luminance = weighted_channel_mean_std(luminance, mask)[:, :1]
        background_luminance = weighted_channel_mean_std(luminance, background_mask)[:, :1]
        global_luminance = luminance.mean(dim=(-2, -1))
        candidate = self.palette_luminance.view(1, self.num_classes)
        text = text_luminance
        background = background_luminance
        global_value = global_luminance
        text_contrast = contrast_ratio(candidate, text)
        background_contrast = contrast_ratio(candidate, background)
        global_contrast = contrast_ratio(candidate, global_value)
        return torch.stack(
            (
                text_contrast / 21.0,
                background_contrast / 21.0,
                global_contrast / 21.0,
                (candidate - text).abs(),
                (candidate - background).abs(),
                (text_contrast >= 4.5).float(),
            ),
            dim=-1,
        )

    def forward(self, x: Tensor) -> Tensor:
        visual = self.visual_features(x)
        stats = self.stat_encoder(self.mask_statistics(x))
        context = self.context_projection(torch.cat((visual, stats), dim=1))
        candidate_embedding = self.candidate_encoder(self.palette_features)
        batch_size = int(x.shape[0])
        context_expanded = context[:, None, :].expand(-1, self.num_classes, -1)
        candidates = candidate_embedding[None, :, :].expand(batch_size, -1, -1)
        contrast = self.candidate_contrast_features(x)
        return self.scorer(
            torch.cat((context_expanded, candidates, contrast), dim=-1)
        ).squeeze(-1)


def build_flatten_mlp(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    **_kwargs: Any,
) -> FlattenMLP:
    return FlattenMLP(num_classes=num_classes, dropout=dropout)


def build_simple_cnn(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> SimpleCNN:
    return SimpleCNN(
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> SimpleCNN:
    return SimpleCNN(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        dropout=dropout,
        activation=activation,
    )


def _build_residual_simple_cnn_variant(
    *,
    num_classes: int,
    hidden_dim: int,
    feature_channels: tuple[int, ...],
    residual_blocks: tuple[int, ...],
    dropout: float,
    activation: str,
    use_se: bool = False,
    use_eca: bool = False,
) -> ResidualSimpleCNN:
    return ResidualSimpleCNN(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        feature_channels=feature_channels,
        residual_blocks=residual_blocks,
        use_se=use_se,
        use_eca=use_eca,
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium_residual(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 1, 2),
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium_residual_deeper(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 2, 3),
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium_residual_se(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 1, 2),
        use_se=True,
        dropout=dropout,
        activation=activation,
    )


def build_titlenet_fast_a(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 1, 1),
        dropout=dropout,
        activation=activation,
    )


def build_titlenet_fast_b(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 1, 1),
        use_eca=True,
        dropout=dropout,
        activation=activation,
    )


def build_titlenet_fast_c(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=192,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(0, 1, 1),
        use_eca=True,
        dropout=dropout,
        activation=activation,
    )


def build_titlenet_student(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        hidden_dim=128,
        feature_channels=(32, 64, 96, 128),
        residual_blocks=(0, 1, 1),
        use_eca=True,
        dropout=dropout,
        activation=activation,
    )


VARIANT_HIDDEN_DIM = "hidden_dim"
VARIANT_FEATURE_CHANNELS = "feature_channels"
VARIANT_RESIDUAL_BLOCKS = "residual_blocks"
VARIANT_USE_SE = "use_se"
VARIANT_USE_ECA = "use_eca"


TITLENET_ABLATION_VARIANTS: dict[str, dict[str, Any]] = {
    "no_se": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 2),
    },
    "no_residual": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (0, 0, 0),
    },
    "no_first_residual": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (0, 1, 2),
        VARIANT_USE_SE: True,
    },
    "no_middle_residual": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 0, 2),
        VARIANT_USE_SE: True,
    },
    "no_last_residual": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 0),
        VARIANT_USE_SE: True,
    },
    "no_last_extra_residual": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 1),
        VARIANT_USE_SE: True,
    },
    "eca": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 2),
        VARIANT_USE_ECA: True,
    },
    "narrow": {
        VARIANT_HIDDEN_DIM: 160,
        VARIANT_FEATURE_CHANNELS: (32, 64, 96, 128),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 2),
        VARIANT_USE_SE: True,
    },
    "wide": {
        VARIANT_HIDDEN_DIM: 256,
        VARIANT_FEATURE_CHANNELS: (64, 128, 160, 192),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 2),
        VARIANT_USE_SE: True,
    },
    "shallow": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 1, 1),
        VARIANT_USE_SE: True,
    },
    "deeper": {
        VARIANT_HIDDEN_DIM: 192,
        VARIANT_FEATURE_CHANNELS: (48, 96, 128, 160),
        VARIANT_RESIDUAL_BLOCKS: (1, 2, 3),
        VARIANT_USE_SE: True,
    },
}
TITLENET_STAGE_ABLATION_PARTS: dict[str, frozenset[str]] = {
    "no_stem": frozenset({"stem"}),
    "no_stage1": frozenset({"stage1"}),
    "no_stage2": frozenset({"stage2"}),
    "no_stage3": frozenset({"stage3"}),
}


def build_titlenet_ablation_variant(
    *,
    variant: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNN:
    try:
        variant_config = TITLENET_ABLATION_VARIANTS[variant]
    except KeyError as exc:
        available = ", ".join(sorted(TITLENET_ABLATION_VARIANTS))
        raise ValueError(f"unknown titlenet ablation variant={variant!r}: {available}") from exc
    return _build_residual_simple_cnn_variant(
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
        **variant_config,
    )


def build_titlenet_stage_ablation_variant(
    *,
    variant: str,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> StageAblatedTitLeNet:
    try:
        ablated_parts = TITLENET_STAGE_ABLATION_PARTS[variant]
    except KeyError as exc:
        available = ", ".join(sorted(TITLENET_STAGE_ABLATION_PARTS))
        message = f"unknown titlenet stage ablation variant={variant!r}: {available}"
        raise ValueError(message) from exc
    return StageAblatedTitLeNet(
        num_classes=num_classes,
        ablated_parts=ablated_parts,
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_large(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> SimpleCNN:
    return SimpleCNN(
        num_classes=num_classes,
        hidden_dim=256,
        feature_channels=(64, 128, 160, 192),
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium_mask_pool(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> SimpleCNNMaskPool:
    return SimpleCNNMaskPool(
        num_classes=num_classes,
        hidden_dim=256,
        feature_channels=(48, 96, 128, 160),
        dropout=dropout,
        activation=activation,
    )


def build_simple_cnn_medium_residual_mask_pool(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> ResidualSimpleCNNMaskPool:
    return ResidualSimpleCNNMaskPool(
        num_classes=num_classes,
        hidden_dim=256,
        feature_channels=(48, 96, 128, 160),
        residual_blocks=(1, 1, 2),
        dropout=dropout,
        activation=activation,
    )


def build_mask_aware_cnn(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> MaskAwareCNN:
    return MaskAwareCNN(
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
    )


def build_mask_aware_cnn_medium(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> MaskAwareCNN:
    return MaskAwareCNN(
        num_classes=num_classes,
        hidden_dim=256,
        feature_channels=(48, 96, 128, 160),
        dropout=dropout,
        activation=activation,
    )


def build_mask_aware_palette_net(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
    **_kwargs: Any,
) -> MaskAwarePaletteNet:
    return MaskAwarePaletteNet(
        num_classes=num_classes,
        dropout=dropout,
        palette_features=load_palette_features(palette_path, num_classes=num_classes),
    )


def build_mask_aware_tiny_hybrid_ranker(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    palette_path: str | Path = DEFAULT_PALETTE_PATH,
    **_kwargs: Any,
) -> MaskAwareTinyHybridColorRanker:
    return MaskAwareTinyHybridColorRanker(
        num_classes=num_classes,
        dropout=dropout,
        activation=activation,
        palette_features=load_palette_features(palette_path, num_classes=num_classes),
    )


def build_vit_tiny(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.1,
    **_kwargs: Any,
) -> TinyVisionTransformer:
    return TinyVisionTransformer(num_classes=num_classes, dropout=dropout)


def build_title_hybrid_tiny(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> TitleHybridTransformer:
    return TitleHybridTransformer(
        num_classes=num_classes,
        feature_channels=(32, 64, 96, 128),
        residual_blocks=(0, 1, 1),
        embed_dim=128,
        depth=2,
        num_heads=4,
        mlp_dim=256,
        hidden_dim=256,
        dropout=dropout,
        activation=activation,
    )


def build_title_hybrid_fast(
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout: float = 0.2,
    activation: str = DEFAULT_ACTIVATION,
    **_kwargs: Any,
) -> TitleHybridTransformer:
    return TitleHybridTransformer(
        num_classes=num_classes,
        feature_channels=(24, 48, 80, 96),
        residual_blocks=(0, 0, 1),
        embed_dim=96,
        depth=2,
        num_heads=4,
        mlp_dim=192,
        hidden_dim=192,
        dropout=dropout,
        activation=activation,
    )
