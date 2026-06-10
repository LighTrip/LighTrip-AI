from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.evaluate_model_checkpoints import (
    build_model_from_checkpoint,
    load_checkpoint,
    training_config_from_checkpoint,
)
from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    markdown_image_path,
)
from src.models.title_color_model_registry import normalize_model_name
from src.title_color_recommendation.data.dataset import TitleColorDataset
from src.title_color_recommendation.training.trainer import resolve_device


LOGGER = logging.getLogger(__name__)
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/titlenet_ndcg3_eval/checkpoint_best.pt")
DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_PALETTE_PATH = Path("data/title_color_recommendation/processed/palette.json")
DEFAULT_OUTPUT_DIR = Path("outputs/reports/titlenet_stage_visualization")
DEFAULT_PAPER_OVERVIEW_NAME = "titlenet_paper_all_in_one.png"
DEFAULT_COMPARISON_MODELS = (
    "titlenet_no_se",
    "titlenet_no_residual",
    "titlenet_no_stage3",
)
DEFAULT_COMPARISON_CHECKPOINT_ROOT = Path("outputs/checkpoints/titlenet_ablation")
INPUT_HEIGHT = 36
INPUT_WIDTH = 136
STEM_NAME = "stem"
STAGE1_NAME = "stage1"
STAGE2_NAME = "stage2"
STAGE3_NAME = "stage3"
STAGE_NAMES = (STEM_NAME, STAGE1_NAME, STAGE2_NAME, STAGE3_NAME)
MODEL_TITLENET = "titlenet"
FULL_TITLENET_LABEL = "Full TitLeNet"
HOOK_VALUE_KEY = "value"
ALIGN_CENTER = "center"
COLOR_WHITE = "white"
COMPARISON_LABELS = {
    MODEL_TITLENET: FULL_TITLENET_LABEL,
    "titlenet_no_se": "Without SE",
    "titlenet_no_residual": "Without residual",
    "titlenet_no_stage3": "Without Stage 3",
}


@dataclass(frozen=True)
class PaletteColor:
    palette_id: int
    name: str
    hex_code: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    label: str
    checkpoint_path: Path
    model_dtype: str
    model: nn.Module


@dataclass(frozen=True)
class SampleRecord:
    sample_number: int
    dataset_index: int
    image_id: str
    stage_figure_path: Path
    preview_figure_path: Path
    comparison_figure_path: Path | None
    top1_palette_id: int
    top1_probability: float


@dataclass(frozen=True)
class SampleFigurePaths:
    stage: Path
    preview: Path
    comparison: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize TitLeNet stage activations and Grad-CAM maps."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default=MODEL_TITLENET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--labels-matrix", type=Path, default=None)
    parser.add_argument("--labels-soft", type=Path, default=None)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--image-ids", default="")
    parser.add_argument("--target-class", type=int, default=None)
    parser.add_argument("--preview-top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="Model/input dtype for hook-based visualization.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-overview-path", type=Path, default=None)
    parser.add_argument(
        "--comparison-models",
        default=",".join(DEFAULT_COMPARISON_MODELS),
    )
    parser.add_argument(
        "--comparison-checkpoint-root",
        type=Path,
        default=DEFAULT_COMPARISON_CHECKPOINT_ROOT,
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args(argv)


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_inside_project(
        PROJECT_ROOT,
        path,
        must_exist=must_exist,
    )


def parse_csv_values(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def parse_sample_indices(raw_value: str) -> list[int]:
    return [int(value) for value in parse_csv_values(raw_value)]


def load_palette(path: Path) -> dict[int, PaletteColor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    palette: dict[int, PaletteColor] = {}
    for item in payload:
        palette_id = int(item["id"])
        rgb = tuple(float(channel) / 255.0 for channel in item["rgb"])
        if len(rgb) != 3:
            raise ValueError(f"palette rgb must contain three channels: {palette_id}")
        palette[palette_id] = PaletteColor(
            palette_id=palette_id,
            name=str(item["name"]),
            hex_code=str(item["hex"]),
            rgb=(rgb[0], rgb[1], rgb[2]),
        )
    return palette


def dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.labels_matrix is not None:
        kwargs["labels_matrix_path"] = resolve_path(args.labels_matrix, must_exist=True)
    if args.labels_soft is not None:
        kwargs["labels_soft_path"] = resolve_path(args.labels_soft, must_exist=True)
    return kwargs


def create_dataset(args: argparse.Namespace) -> TitleColorDataset:
    return TitleColorDataset(
        args.split,
        data_root=resolve_path(args.data_root, must_exist=True),
        project_root=PROJECT_ROOT,
        augment=False,
        **dataset_kwargs(args),
    )


def select_dataset_indices(
    dataset: TitleColorDataset,
    *,
    sample_count: int,
    raw_indices: str,
    raw_image_ids: str,
) -> list[int]:
    explicit_indices = parse_sample_indices(raw_indices)
    if explicit_indices:
        return explicit_indices

    image_ids = parse_csv_values(raw_image_ids)
    if image_ids:
        return indices_for_image_ids(dataset, image_ids)

    return list(range(min(sample_count, len(dataset))))


def indices_for_image_ids(dataset: TitleColorDataset, image_ids: list[str]) -> list[int]:
    index_by_id = {
        item.image_id: index
        for index, item in enumerate(dataset.items)
    }
    missing_ids = [image_id for image_id in image_ids if image_id not in index_by_id]
    if missing_ids:
        raise ValueError(f"image ids not found in split manifest: {missing_ids}")
    return [index_by_id[image_id] for image_id in image_ids]


def resolve_model_dtype(raw_dtype: str, device: torch.device) -> torch.dtype:
    if raw_dtype == "float32":
        return torch.float32
    if raw_dtype == "float16":
        return torch.float16
    raise ValueError(f"unsupported model dtype: {raw_dtype}")


def model_input_tensor(sample: Mapping[str, Any], bundle: ModelBundle, device: torch.device) -> Tensor:
    model_dtype = resolve_model_dtype(bundle.model_dtype, device)
    return sample["x"].unsqueeze(0).to(device=device, dtype=model_dtype)


def load_model_bundle(
    *,
    model_name: str,
    checkpoint_path: Path,
    label: str,
    device: torch.device,
    model_dtype: str,
) -> ModelBundle:
    torch_dtype = resolve_model_dtype(model_dtype, device)
    checkpoint = load_checkpoint(checkpoint_path)
    config = training_config_from_checkpoint(
        checkpoint,
        fallback_model_name=model_name,
        batch_size=1,
        device=device.type,
        num_workers=0,
    )
    model = build_model_from_checkpoint(checkpoint, config)
    model.to(device=device, dtype=torch_dtype)
    model.eval()
    return ModelBundle(
        model_name=normalize_model_name(model_name),
        label=label,
        checkpoint_path=checkpoint_path,
        model_dtype=model_dtype,
        model=model,
    )


def comparison_checkpoint_path(root: Path, model_name: str) -> Path:
    return root / normalize_model_name(model_name) / "checkpoint_best.pt"


def load_comparison_bundles(
    args: argparse.Namespace,
    *,
    full_bundle: ModelBundle,
    device: torch.device,
) -> list[ModelBundle]:
    bundles = [full_bundle]
    checkpoint_root = resolve_path(args.comparison_checkpoint_root)
    for model_name in parse_csv_values(args.comparison_models):
        normalized = normalize_model_name(model_name)
        checkpoint_path = comparison_checkpoint_path(checkpoint_root, normalized)
        if not checkpoint_path.exists():
            LOGGER.warning("skip missing comparison checkpoint: %s", checkpoint_path)
            continue
        bundles.append(
            load_model_bundle(
                model_name=normalized,
                checkpoint_path=checkpoint_path,
                label=display_model_name(normalized),
                device=device,
                model_dtype=full_bundle.model_dtype,
            )
        )
    return bundles


def display_model_name(model_name: str) -> str:
    if model_name in COMPARISON_LABELS:
        return COMPARISON_LABELS[model_name]
    return model_name.replace("titlenet_no_", "Without ").replace("_", " ")


def sequential_feature_modules(model: nn.Module) -> list[nn.Module]:
    features = getattr(model, "features", None)
    net = getattr(features, "net", None)
    if not isinstance(net, nn.Sequential):
        raise TypeError("TitLeNet visualization requires model.features.net Sequential")
    return list(net.children())


def is_pointwise_projection(module: nn.Module) -> bool:
    return module.__class__.__name__ == "PointwiseProjection"


def is_stage_start(module: nn.Module) -> bool:
    return module.__class__.__name__ in {
        "DepthwiseSeparableConv",
        "PointwiseProjection",
    }


def stem_end_index(modules: list[nn.Module]) -> int:
    if not modules:
        raise ValueError("feature extractor contains no modules")
    if is_pointwise_projection(modules[0]) or isinstance(modules[0], nn.Sequential):
        return 0
    if len(modules) < 3:
        raise ValueError("feature extractor stem must contain at least three modules")
    return 2


def titlenet_stage_modules(model: nn.Module) -> dict[str, nn.Module]:
    modules = sequential_feature_modules(model)
    stem_index = stem_end_index(modules)
    stages = {STEM_NAME: modules[stem_index]}
    stage_start_indices = [
        index
        for index in range(stem_index + 1, len(modules))
        if is_stage_start(modules[index])
    ]
    for stage_number, start_index in enumerate(stage_start_indices, start=1):
        end_index = (
            stage_start_indices[stage_number] - 1
            if stage_number < len(stage_start_indices)
            else len(modules) - 1
        )
        stages[f"stage{stage_number}"] = modules[end_index]
    missing_stages = [stage_name for stage_name in STAGE_NAMES if stage_name not in stages]
    if missing_stages:
        raise ValueError(f"missing TitLeNet stages: {missing_stages}")
    return stages


def capture_stage_activations(
    model: nn.Module,
    x: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    stage_modules = titlenet_stage_modules(model)
    activations: dict[str, Tensor] = {}
    hooks = [
        module.register_forward_hook(make_activation_hook(stage_name, activations))
        for stage_name, module in stage_modules.items()
    ]
    try:
        with torch.inference_mode():
            logits = model(x)
    finally:
        remove_hooks(hooks)
    return logits, activations


def make_activation_hook(
    stage_name: str,
    activations: dict[str, Tensor],
) -> Any:
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        activations[stage_name] = output.detach()

    return hook


def remove_hooks(hooks: list[Any]) -> None:
    for hook in hooks:
        hook.remove()


def gradcam_for_target(
    model: nn.Module,
    x: Tensor,
    *,
    target_class: int | None,
) -> tuple[Tensor, int, Tensor]:
    target_layer = titlenet_stage_modules(model)[STAGE3_NAME]
    activations: dict[str, Tensor] = {}
    gradients: dict[str, Tensor] = {}
    hooks = [
        target_layer.register_forward_hook(make_gradcam_activation_hook(activations)),
        target_layer.register_full_backward_hook(make_gradcam_gradient_hook(gradients)),
    ]
    model.zero_grad(set_to_none=True)
    grad_input = x.detach().clone().requires_grad_(True)
    try:
        logits = model(grad_input)
        class_index = int(logits.argmax(dim=-1).item()) if target_class is None else target_class
        logits[:, class_index].sum().backward()
        cam = gradcam_from_tensors(
            activations[HOOK_VALUE_KEY],
            gradients[HOOK_VALUE_KEY],
        )
    finally:
        remove_hooks(hooks)
    return cam, class_index, logits.detach()


def make_gradcam_activation_hook(activations: dict[str, Tensor]) -> Any:
    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Tensor) -> None:
        activations[HOOK_VALUE_KEY] = output

    return hook


def make_gradcam_gradient_hook(gradients: dict[str, Tensor]) -> Any:
    def hook(
        _module: nn.Module,
        _grad_input: tuple[Tensor, ...],
        grad_output: tuple[Tensor, ...],
    ) -> None:
        gradients[HOOK_VALUE_KEY] = grad_output[0]

    return hook


def gradcam_from_tensors(activations: Tensor, gradients: Tensor) -> Tensor:
    weights = gradients.mean(dim=(-2, -1), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    return resize_heatmap(cam, size=(INPUT_HEIGHT, INPUT_WIDTH))


def stage_heatmap(activation: Tensor) -> Tensor:
    heatmap = activation.detach().abs().mean(dim=1, keepdim=True)
    return resize_heatmap(heatmap, size=(INPUT_HEIGHT, INPUT_WIDTH))


def resize_heatmap(heatmap: Tensor, *, size: tuple[int, int]) -> Tensor:
    resized = F.interpolate(
        heatmap.float(),
        size=size,
        mode="bilinear",
        align_corners=False,
    )
    return normalize_tensor_map(resized)


def normalize_tensor_map(value: Tensor, *, eps: float = 1e-8) -> Tensor:
    flattened = value.flatten(start_dim=1)
    lower = flattened.min(dim=1).values.view(-1, 1, 1, 1)
    upper = flattened.max(dim=1).values.view(-1, 1, 1, 1)
    return (value - lower) / (upper - lower).clamp_min(eps)


def rgb_array(x: Tensor) -> np.ndarray:
    array = x[0, :3].detach().cpu().permute(1, 2, 0).numpy()
    return array.clip(0.0, 1.0).astype(np.float32, copy=False)


def mask_array(x: Tensor) -> np.ndarray:
    array = x[0, 3].detach().cpu().numpy()
    return array.clip(0.0, 1.0).astype(np.float32, copy=False)


def heatmap_array(heatmap: Tensor) -> np.ndarray:
    array = heatmap[0, 0].detach().cpu().numpy()
    return array.clip(0.0, 1.0).astype(np.float32, copy=False)


def top_probability(logits: Tensor, class_index: int) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    return float(probabilities[0, class_index].detach().cpu().item())


def topk_predictions(logits: Tensor, *, limit: int) -> list[tuple[int, float]]:
    probabilities = torch.softmax(logits.float(), dim=-1)
    topk = probabilities.topk(min(limit, int(probabilities.shape[-1])), dim=-1)
    return [
        (int(palette_id), float(probability))
        for palette_id, probability in zip(
            topk.indices[0].detach().cpu().tolist(),
            topk.values[0].detach().cpu().tolist(),
        )
    ]


def rendered_title_preview(
    x: Tensor,
    color: PaletteColor,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    image = rgb_array(x)
    text_alpha = mask_array(x)[:, :, None] * alpha
    color_array = np.asarray(color.rgb, dtype=np.float32).reshape(1, 1, 3)
    return (image * (1.0 - text_alpha) + color_array * text_alpha).clip(0.0, 1.0)


def add_image_axis(axis: Any, image: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def add_mask_axis(axis: Any, mask: np.ndarray) -> None:
    axis.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title("Text mask")
    axis.axis("off")


def add_overlay_axis(
    axis: Any,
    image: np.ndarray,
    heatmap: np.ndarray,
    title: str,
) -> None:
    axis.imshow(image)
    axis.imshow(heatmap, cmap="magma", alpha=0.55, vmin=0.0, vmax=1.0)
    axis.set_title(title)
    axis.axis("off")


def add_palette_axis(
    axis: Any,
    palette: Mapping[int, PaletteColor],
    *,
    palette_id: int,
    probability: float,
) -> None:
    color = palette.get(palette_id)
    rgb = color.rgb if color is not None else (0.0, 0.0, 0.0)
    name = color.name if color is not None else "unknown"
    hex_code = color.hex_code if color is not None else "-"
    swatch = np.ones((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.float32)
    swatch[:, :, :] = np.asarray(rgb, dtype=np.float32)
    axis.imshow(swatch)
    axis.set_title(f"Top-1: {palette_id}")
    axis.text(
        0.5,
        0.5,
        f"{name}\n{hex_code}\np={probability:.3f}",
        color=text_color_for_rgb(rgb),
        fontsize=9,
        ha=ALIGN_CENTER,
        va="center",
        transform=axis.transAxes,
        bbox={"facecolor": COLOR_WHITE, "alpha": 0.72, "edgecolor": "none"},
    )
    axis.axis("off")


def add_preview_axis(
    axis: Any,
    x: Tensor,
    palette: Mapping[int, PaletteColor],
    *,
    palette_id: int,
    probability: float,
    title: str,
    font_size: float = 8.5,
    show_label: bool = True,
) -> None:
    color = palette.get(palette_id)
    if color is None:
        add_image_axis(axis, rgb_array(x), title)
        return
    axis.imshow(rendered_title_preview(x, color))
    axis.set_title(title)
    if show_label:
        axis.text(
            0.5,
            0.06,
            f"{color.name} {color.hex_code} p={probability:.3f}",
            color=COLOR_WHITE,
            fontsize=font_size,
            ha=ALIGN_CENTER,
            va="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.48, "edgecolor": "none"},
        )
    axis.axis("off")


def add_row_label(axis: Any, label: str) -> None:
    axis.text(
        -0.08,
        0.5,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="right",
        va=ALIGN_CENTER,
    )


def text_color_for_rgb(rgb: tuple[float, float, float]) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.55 else COLOR_WHITE


def save_figure(figure: Any, path: Path, *, dpi: int, save_pdf: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def write_preview_figure(
    path: Path,
    *,
    image_id: str,
    x: Tensor,
    logits: Tensor,
    palette: Mapping[int, PaletteColor],
    top_k: int,
    dpi: int,
    save_pdf: bool,
) -> None:
    predictions = topk_predictions(logits, limit=top_k)
    plt = load_pyplot(PROJECT_ROOT)
    column_count = len(predictions) + 2
    figure, axes = plt.subplots(1, column_count, figsize=(3.4 * column_count, 2.1))
    figure.suptitle(
        f"Rendered title color preview: {image_id}",
        fontweight="bold",
        fontsize=12,
    )
    add_image_axis(axes[0], rgb_array(x), "ROI")
    add_mask_axis(axes[1], mask_array(x))
    for rank, (axis, prediction) in enumerate(zip(axes[2:], predictions), start=1):
        palette_id, probability = prediction
        add_preview_axis(
            axis,
            x,
            palette,
            palette_id=palette_id,
            probability=probability,
            title=f"Top-{rank}: {palette_id}",
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84), w_pad=0.35)
    save_figure(figure, path, dpi=dpi, save_pdf=save_pdf)
    plt.close(figure)


def write_stage_figure(
    path: Path,
    *,
    image_id: str,
    x: Tensor,
    activations: Mapping[str, Tensor],
    gradcam: Tensor,
    logits: Tensor,
    target_class: int,
    palette: Mapping[int, PaletteColor],
    dpi: int,
    save_pdf: bool,
) -> None:
    plt = load_pyplot(PROJECT_ROOT)
    image = rgb_array(x)
    probability = top_probability(logits, target_class)
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 4.2))
    figure.suptitle(
        f"TitLeNet stage visualization: {image_id}",
        fontweight="bold",
        fontsize=12,
    )
    add_image_axis(axes[0][0], image, "ROI")
    add_mask_axis(axes[0][1], mask_array(x))
    add_palette_axis(
        axes[0][2],
        palette,
        palette_id=target_class,
        probability=probability,
    )
    add_overlay_axis(
        axes[0][3],
        image,
        heatmap_array(stage_heatmap(activations[STEM_NAME])),
        "Stem",
    )
    for axis, stage_name in zip(axes[1], (STAGE1_NAME, STAGE2_NAME, STAGE3_NAME)):
        add_overlay_axis(
            axis,
            image,
            heatmap_array(stage_heatmap(activations[stage_name])),
            stage_name.title().replace("Stage", "Stage "),
        )
    add_overlay_axis(axes[1][3], image, heatmap_array(gradcam), "Top-1 Grad-CAM")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), h_pad=0.8, w_pad=0.5)
    save_figure(figure, path, dpi=dpi, save_pdf=save_pdf)
    plt.close(figure)


def write_comparison_figure(
    path: Path,
    *,
    image_id: str,
    x: Tensor,
    bundles: list[ModelBundle],
    target_class: int,
    dpi: int,
    save_pdf: bool,
) -> None:
    if len(bundles) <= 1:
        return
    plt = load_pyplot(PROJECT_ROOT)
    image = rgb_array(x)
    figure, axes = plt.subplots(
        1,
        len(bundles) + 1,
        figsize=(3.4 * (len(bundles) + 1), 2.15),
    )
    figure.suptitle(
        f"TitLeNet ablation Grad-CAM comparison: {image_id}",
        fontweight="bold",
        fontsize=12,
    )
    add_image_axis(axes[0], image, "ROI")
    for axis, bundle in zip(axes[1:], bundles):
        gradcam, _class_index, _logits = gradcam_for_target(
            bundle.model,
            x,
            target_class=target_class,
        )
        add_overlay_axis(axis, image, heatmap_array(gradcam), bundle.label)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.86), w_pad=0.35)
    save_figure(figure, path, dpi=dpi, save_pdf=save_pdf)
    plt.close(figure)


def paper_overview_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.paper_overview_path is not None:
        return resolve_path(args.paper_overview_path)
    return output_dir / DEFAULT_PAPER_OVERVIEW_NAME


def column_title(row_index: int, title: str) -> str:
    return title if row_index == 0 else ""


def add_paper_overview_row(
    axes: Any,
    *,
    row_index: int,
    dataset_index: int,
    dataset: TitleColorDataset,
    full_bundle: ModelBundle,
    palette: Mapping[int, PaletteColor],
    target_class: int | None,
    preview_top_k: int,
    device: torch.device,
) -> None:
    sample = dataset[dataset_index]
    x = model_input_tensor(sample, full_bundle, device)
    image = rgb_array(x)
    logits, activations = capture_stage_activations(full_bundle.model, x)
    gradcam, class_index, _gradcam_logits = gradcam_for_target(
        full_bundle.model,
        x,
        target_class=target_class,
    )
    predictions = topk_predictions(logits, limit=preview_top_k)

    add_image_axis(axes[0], image, column_title(row_index, "Input"))
    add_row_label(axes[0], f"({chr(ord('a') + row_index)})")
    for rank, (axis, prediction) in enumerate(zip(axes[1:4], predictions), start=1):
        palette_id, probability = prediction
        add_preview_axis(
            axis,
            x,
            palette,
            palette_id=palette_id,
            probability=probability,
            title=column_title(row_index, f"Top-{rank}"),
            font_size=6.2,
            show_label=False,
        )

    stage_columns = (
        (STAGE1_NAME, "Stage 1"),
        (STAGE2_NAME, "Stage 2"),
        (STAGE3_NAME, "Stage 3"),
    )
    for axis, (stage_name, title) in zip(axes[4:7], stage_columns):
        add_overlay_axis(
            axis,
            image,
            heatmap_array(stage_heatmap(activations[stage_name])),
            column_title(row_index, title),
        )
    add_overlay_axis(axes[7], image, heatmap_array(gradcam), column_title(row_index, "Grad-CAM"))


def write_paper_overview_figure(
    path: Path,
    *,
    indices: list[int],
    dataset: TitleColorDataset,
    full_bundle: ModelBundle,
    comparison_bundles: list[ModelBundle],
    palette: Mapping[int, PaletteColor],
    target_class: int | None,
    preview_top_k: int,
    device: torch.device,
    dpi: int,
    save_pdf: bool,
) -> None:
    column_count = 8
    row_count = len(indices)
    plt = load_pyplot(PROJECT_ROOT)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(2.1 * column_count, max(2.8, 0.76 * row_count)),
        squeeze=False,
    )
    for row_index, dataset_index in enumerate(indices):
        add_paper_overview_row(
            axes[row_index],
            row_index=row_index,
            dataset_index=dataset_index,
            dataset=dataset,
            full_bundle=full_bundle,
            palette=palette,
            target_class=target_class,
            preview_top_k=preview_top_k,
            device=device,
        )
    figure.tight_layout(w_pad=0.15, h_pad=0.0)
    save_figure(figure, path, dpi=dpi, save_pdf=save_pdf)
    plt.close(figure)


def safe_file_stem(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def sample_figure_paths(output_dir: Path, *, sample_number: int, image_id: str) -> SampleFigurePaths:
    sample_prefix = f"sample_{sample_number:03d}_{safe_file_stem(image_id)}"
    return SampleFigurePaths(
        stage=output_dir / f"{sample_prefix}_stage_heatmap.png",
        preview=output_dir / f"{sample_prefix}_rendered_preview.png",
        comparison=output_dir / f"{sample_prefix}_ablation_gradcam.png",
    )


def write_sample_figures(
    *,
    paths: SampleFigurePaths,
    image_id: str,
    x: Tensor,
    logits: Tensor,
    activations: Mapping[str, Tensor],
    gradcam: Tensor,
    gradcam_logits: Tensor,
    class_index: int,
    comparison_bundles: list[ModelBundle],
    palette: Mapping[int, PaletteColor],
    preview_top_k: int,
    dpi: int,
    save_pdf: bool,
) -> Path | None:
    write_preview_figure(
        paths.preview,
        image_id=image_id,
        x=x,
        logits=logits,
        palette=palette,
        top_k=preview_top_k,
        dpi=dpi,
        save_pdf=save_pdf,
    )
    write_stage_figure(
        paths.stage,
        image_id=image_id,
        x=x,
        activations=activations,
        gradcam=gradcam,
        logits=gradcam_logits,
        target_class=class_index,
        palette=palette,
        dpi=dpi,
        save_pdf=save_pdf,
    )
    if len(comparison_bundles) <= 1:
        return None
    write_comparison_figure(
        paths.comparison,
        image_id=image_id,
        x=x,
        bundles=comparison_bundles,
        target_class=class_index,
        dpi=dpi,
        save_pdf=save_pdf,
    )
    return paths.comparison


def visualize_sample(
    *,
    sample_number: int,
    dataset_index: int,
    dataset: TitleColorDataset,
    full_bundle: ModelBundle,
    comparison_bundles: list[ModelBundle],
    output_dir: Path,
    palette: Mapping[int, PaletteColor],
    device: torch.device,
    target_class: int | None,
    preview_top_k: int,
    dpi: int,
    save_pdf: bool,
) -> SampleRecord:
    sample = dataset[dataset_index]
    image_id = str(sample["image_id"])
    x = model_input_tensor(sample, full_bundle, device)
    logits, activations = capture_stage_activations(full_bundle.model, x)
    gradcam, class_index, gradcam_logits = gradcam_for_target(
        full_bundle.model,
        x,
        target_class=target_class,
    )
    paths = sample_figure_paths(output_dir, sample_number=sample_number, image_id=image_id)
    comparison_result = write_sample_figures(
        paths=paths,
        image_id=image_id,
        x=x,
        logits=logits,
        activations=activations,
        gradcam=gradcam,
        gradcam_logits=gradcam_logits,
        class_index=class_index,
        comparison_bundles=comparison_bundles,
        palette=palette,
        preview_top_k=preview_top_k,
        dpi=dpi,
        save_pdf=save_pdf,
    )

    return SampleRecord(
        sample_number=sample_number,
        dataset_index=dataset_index,
        image_id=image_id,
        stage_figure_path=paths.stage,
        preview_figure_path=paths.preview,
        comparison_figure_path=comparison_result,
        top1_palette_id=class_index,
        top1_probability=top_probability(logits, class_index),
    )


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    full_bundle: ModelBundle,
    comparison_bundles: list[ModelBundle],
    records: list[SampleRecord],
    overview_path: Path,
) -> None:
    lines = [
        "# TitLeNet Stage Visualization",
        "",
        "## Configuration",
        "",
        f"- checkpoint: `{full_bundle.checkpoint_path}`",
        f"- model_name: `{full_bundle.model_name}`",
        f"- model_dtype: `{full_bundle.model_dtype}`",
        f"- split: `{args.split}`",
        f"- target_class: `{args.target_class}`",
        "",
        "## Comparison Models",
        "",
    ]
    for bundle in comparison_bundles:
        lines.append(f"- `{bundle.model_name}`: `{bundle.checkpoint_path}`")

    lines.extend(
        [
            "",
            "## Paper Overview",
            "",
            f"![overview]({markdown_image_path(path, overview_path)})",
            "",
            "## Samples",
            "",
        ]
    )
    lines.append("")
    for record in records:
        lines.extend(
            [
                f"### Sample {record.sample_number:03d}: `{record.image_id}`",
                "",
                f"- dataset_index: `{record.dataset_index}`",
                f"- top1_palette_id: `{record.top1_palette_id}`",
                f"- top1_probability: `{record.top1_probability:.4f}`",
                "",
                "Rendered title preview:",
                "",
                f"![preview]({markdown_image_path(path, record.preview_figure_path)})",
                "",
                "Stage activation and top-1 Grad-CAM:",
                "",
                f"![stage]({markdown_image_path(path, record.stage_figure_path)})",
                "",
            ]
        )
        if record.comparison_figure_path is not None:
            lines.extend(
                [
                    "Ablation Grad-CAM comparison:",
                    "",
                    f"![comparison]({markdown_image_path(path, record.comparison_figure_path)})",
                    "",
                ]
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> list[SampleRecord]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = create_dataset(args)
    indices = select_dataset_indices(
        dataset,
        sample_count=args.sample_count,
        raw_indices=args.sample_indices,
        raw_image_ids=args.image_ids,
    )
    palette = load_palette(resolve_path(args.palette, must_exist=True))
    full_bundle = load_model_bundle(
        model_name=args.model_name,
        checkpoint_path=resolve_path(args.checkpoint, must_exist=True),
        label=FULL_TITLENET_LABEL,
        device=device,
        model_dtype=args.model_dtype,
    )
    comparison_bundles = load_comparison_bundles(
        args,
        full_bundle=full_bundle,
        device=device,
    )
    records = [
        visualize_sample(
            sample_number=sample_number,
            dataset_index=dataset_index,
            dataset=dataset,
            full_bundle=full_bundle,
            comparison_bundles=comparison_bundles,
            output_dir=output_dir,
            palette=palette,
            device=device,
            target_class=args.target_class,
            preview_top_k=args.preview_top_k,
            dpi=args.dpi,
            save_pdf=not args.no_pdf,
        )
        for sample_number, dataset_index in enumerate(indices, start=1)
    ]
    overview_path = paper_overview_path(args, output_dir)
    write_paper_overview_figure(
        overview_path,
        indices=indices,
        dataset=dataset,
        full_bundle=full_bundle,
        comparison_bundles=comparison_bundles,
        palette=palette,
        target_class=args.target_class,
        preview_top_k=args.preview_top_k,
        device=device,
        dpi=args.dpi,
        save_pdf=not args.no_pdf,
    )
    write_report(
        output_dir / "titlenet_stage_visualization_report.md",
        args=args,
        full_bundle=full_bundle,
        comparison_bundles=comparison_bundles,
        records=records,
        overview_path=overview_path,
    )
    LOGGER.info("wrote TitLeNet stage visualizations to %s", output_dir)
    return records


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
