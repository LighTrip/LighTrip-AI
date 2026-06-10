from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnx import helper

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.plot_utils import (
    load_pyplot,
    markdown_image_path,
)
from src.title_color_recommendation.data.dataset import TitleColorDataset


LOGGER = logging.getLogger(__name__)
DEFAULT_LOGITS_ONNX = Path(
    "outputs/title_color_recommendation/quantization/"
    "titlenet_student_warm_kd90_fp16_logits.onnx"
)
DEFAULT_TOP1_ONNX = Path(
    "outputs/title_color_recommendation/quantization/"
    "titlenet_student_warm_kd90_fp16_top1.onnx"
)
DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_PALETTE_PATH = Path("data/title_color_recommendation/processed/palette.json")
DEFAULT_OUTPUT_DIR = Path(
    "outputs/reports/titlenet_stage_visualization/"
    "student_fp16_ptq_onnx_activation_selected_8samples"
)
DEFAULT_OVERVIEW_NAME = "titlenet_student_fp16_ptq_onnx_activation_selected_8samples.png"
DEFAULT_IMAGE_IDS = (
    "city_highway_00348",
    "abstract_wave_00040",
    "city_parking_lot_00249",
    "abstract_industrial_area_00230",
    "interior_office_00046",
    "city_bridge_00169",
    "city_crosswalk_00220",
    "abstract_ice_floe_00253",
)
INPUT_HEIGHT = 36
INPUT_WIDTH = 136
ALIGN_CENTER = "center"
COLOR_WHITE = "white"
STAGE_OUTPUTS = {
    "stem": "/features/net/net.2/HardSwish_output_0",
    "stage1": "/features/net/net.3/block/block.5/HardSwish_output_0",
    "stage2": "/features/net/net.5/output_activation/HardSwish_output_0",
    "stage3": "/features/net/net.7/output_activation/HardSwish_output_0",
}


@dataclass(frozen=True)
class PaletteColor:
    palette_id: int
    name: str
    hex_code: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True)
class StageOutputInfo:
    name: str
    tensor_name: str
    elem_type: int
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class SampleRecord:
    sample_number: int
    dataset_index: int
    image_id: str
    top1_palette_id: int
    top1_probability: float
    top1_model_output: int | None
    top1_model_match: bool | None
    sample_figure_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize intermediate activation maps from a TitLeNet ONNX model."
    )
    parser.add_argument("--logits-onnx", type=Path, default=DEFAULT_LOGITS_ONNX)
    parser.add_argument("--top1-onnx", type=Path, default=DEFAULT_TOP1_ONNX)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--palette", type=Path, default=DEFAULT_PALETTE_PATH)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-count", type=int, default=len(DEFAULT_IMAGE_IDS))
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--image-ids", default=",".join(DEFAULT_IMAGE_IDS))
    parser.add_argument("--preview-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overview-path", type=Path, default=None)
    parser.add_argument("--stage-model-output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def resolve_path(path: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_inside_project(PROJECT_ROOT, path, must_exist=must_exist)


def parse_csv_values(raw_value: str) -> list[str]:
    return [value.strip() for value in raw_value.split(",") if value.strip()]


def parse_sample_indices(raw_value: str) -> list[int]:
    return [int(value) for value in parse_csv_values(raw_value)]


def create_dataset(args: argparse.Namespace) -> TitleColorDataset:
    return TitleColorDataset(
        args.split,
        data_root=resolve_path(args.data_root, must_exist=True),
        project_root=PROJECT_ROOT,
        augment=False,
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
        index_by_id = {
            item.image_id: index
            for index, item in enumerate(dataset.items)
        }
        missing_ids = [image_id for image_id in image_ids if image_id not in index_by_id]
        if missing_ids:
            raise ValueError(f"image ids not found in split manifest: {missing_ids}")
        return [index_by_id[image_id] for image_id in image_ids]

    return list(range(min(sample_count, len(dataset))))


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


def value_info_map(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int | str, ...]]]:
    output: dict[str, tuple[int, tuple[int | str, ...]]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        shape: list[int | str] = []
        for dimension in tensor_type.shape.dim:
            if dimension.dim_value:
                shape.append(int(dimension.dim_value))
            elif dimension.dim_param:
                shape.append(str(dimension.dim_param))
            else:
                shape.append("?")
        output[value.name] = (int(tensor_type.elem_type), tuple(shape))
    return output


def stage_output_infos(model: onnx.ModelProto) -> list[StageOutputInfo]:
    inferred_model = onnx.shape_inference.infer_shapes(model)
    values = value_info_map(inferred_model)
    infos: list[StageOutputInfo] = []
    for stage_name, tensor_name in STAGE_OUTPUTS.items():
        if tensor_name not in values:
            raise ValueError(f"stage output not found in ONNX graph: {stage_name}={tensor_name}")
        elem_type, shape = values[tensor_name]
        infos.append(
            StageOutputInfo(
                name=stage_name,
                tensor_name=tensor_name,
                elem_type=elem_type,
                shape=shape,
            )
        )
    return infos


def add_stage_outputs(model: onnx.ModelProto, stage_infos: list[StageOutputInfo]) -> onnx.ModelProto:
    existing_outputs = {output.name for output in model.graph.output}
    for info in stage_infos:
        if info.tensor_name in existing_outputs:
            continue
        model.graph.output.append(
            helper.make_tensor_value_info(
                info.tensor_name,
                info.elem_type,
                list(info.shape),
            )
        )
    return model


def save_stage_output_model(
    logits_onnx_path: Path,
    output_path: Path,
) -> tuple[Path, list[StageOutputInfo]]:
    model = onnx.load(logits_onnx_path)
    stage_infos = stage_output_infos(model)
    model = add_stage_outputs(model, stage_infos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path, stage_infos


def input_tensor(sample: Mapping[str, Any]) -> np.ndarray:
    return sample["x"].unsqueeze(0).numpy().astype(np.float32)


def rgb_array(x: np.ndarray) -> np.ndarray:
    return np.clip(np.transpose(x[0, :3], (1, 2, 0)), 0.0, 1.0).astype(np.float32)


def mask_array(x: np.ndarray) -> np.ndarray:
    return np.clip(x[0, 3], 0.0, 1.0).astype(np.float32)


def activation_heatmap(activation: np.ndarray) -> np.ndarray:
    value = torch.from_numpy(np.abs(activation).mean(axis=1, keepdims=True)).float()
    resized = F.interpolate(
        value,
        size=(INPUT_HEIGHT, INPUT_WIDTH),
        mode="bilinear",
        align_corners=False,
    )
    flattened = resized.flatten(start_dim=1)
    lower = flattened.min(dim=1).values.view(-1, 1, 1, 1)
    upper = flattened.max(dim=1).values.view(-1, 1, 1, 1)
    normalized = (resized - lower) / (upper - lower).clamp_min(1e-8)
    return normalized[0, 0].numpy().astype(np.float32)


def topk_predictions(logits: np.ndarray, *, limit: int) -> list[tuple[int, float]]:
    logits_tensor = torch.from_numpy(logits).float()
    probabilities = torch.softmax(logits_tensor, dim=-1)
    topk = probabilities.topk(min(limit, int(probabilities.shape[-1])), dim=-1)
    return [
        (int(palette_id), float(probability))
        for palette_id, probability in zip(topk.indices[0].tolist(), topk.values[0].tolist())
    ]


def rendered_title_preview(
    x: np.ndarray,
    color: PaletteColor,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    image = rgb_array(x)
    text_alpha = mask_array(x)[:, :, None] * alpha
    color_array = np.asarray(color.rgb, dtype=np.float32).reshape(1, 1, 3)
    return (image * (1.0 - text_alpha) + color_array * text_alpha).clip(0.0, 1.0)


def text_color_for_rgb(rgb: tuple[float, float, float]) -> str:
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return "black" if luminance > 0.55 else COLOR_WHITE


def add_image_axis(axis: Any, image: np.ndarray, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")


def add_mask_axis(axis: Any, mask: np.ndarray) -> None:
    axis.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title("Text mask")
    axis.axis("off")


def add_preview_axis(
    axis: Any,
    x: np.ndarray,
    palette: Mapping[int, PaletteColor],
    *,
    palette_id: int,
    probability: float,
    title: str,
    font_size: float = 8.0,
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


def safe_file_stem(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def save_figure(figure: Any, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")


def run_stage_session(
    session: ort.InferenceSession,
    x: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(output_names, {input_name: x})
    by_name = dict(zip(output_names, values))
    logits = by_name["logits"]
    activations = {
        stage_name: by_name[tensor_name]
        for stage_name, tensor_name in STAGE_OUTPUTS.items()
    }
    return logits, activations


def run_top1_session(session: ort.InferenceSession | None, x: np.ndarray) -> int | None:
    if session is None:
        return None
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    value = session.run([output_name], {input_name: x})[0]
    return int(value.reshape(-1)[0])


def write_sample_figure(
    path: Path,
    *,
    image_id: str,
    x: np.ndarray,
    logits: np.ndarray,
    activations: Mapping[str, np.ndarray],
    palette: Mapping[int, PaletteColor],
    preview_top_k: int,
    dpi: int,
) -> None:
    plt = load_pyplot(PROJECT_ROOT)
    image = rgb_array(x)
    predictions = topk_predictions(logits, limit=preview_top_k)
    figure, axes = plt.subplots(2, 4, figsize=(14.2, 4.2))
    figure.suptitle(
        f"ONNX FP16 PTQ activation heatmaps: {image_id}",
        fontweight="bold",
        fontsize=12,
    )
    add_image_axis(axes[0][0], image, "Input")
    add_mask_axis(axes[0][1], mask_array(x))
    for rank, (axis, prediction) in enumerate(zip(axes[0][2:], predictions[:2]), start=1):
        palette_id, probability = prediction
        add_preview_axis(
            axis,
            x,
            palette,
            palette_id=palette_id,
            probability=probability,
            title=f"Top-{rank}: {palette_id}",
        )
    for axis, stage_name in zip(axes[1], ("stem", "stage1", "stage2", "stage3")):
        add_overlay_axis(
            axis,
            image,
            activation_heatmap(activations[stage_name]),
            stage_name.replace("stage", "Stage ").title(),
        )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92), h_pad=0.8, w_pad=0.5)
    save_figure(figure, path, dpi=dpi)
    plt.close(figure)


def column_title(row_index: int, title: str) -> str:
    return title if row_index == 0 else ""


def add_overview_row(
    axes: Any,
    *,
    row_index: int,
    x: np.ndarray,
    logits: np.ndarray,
    activations: Mapping[str, np.ndarray],
    palette: Mapping[int, PaletteColor],
    preview_top_k: int,
) -> None:
    image = rgb_array(x)
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
        ("stem", "Stem"),
        ("stage1", "Stage 1"),
        ("stage2", "Stage 2"),
        ("stage3", "Stage 3"),
    )
    for axis, (stage_name, title) in zip(axes[4:], stage_columns):
        add_overlay_axis(
            axis,
            image,
            activation_heatmap(activations[stage_name]),
            column_title(row_index, title),
        )


def write_overview_figure(
    path: Path,
    *,
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    palette: Mapping[int, PaletteColor],
    preview_top_k: int,
    dpi: int,
) -> None:
    column_count = 8
    row_count = len(rows)
    plt = load_pyplot(PROJECT_ROOT)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(2.1 * column_count, max(2.8, 0.76 * row_count)),
        squeeze=False,
    )
    for row_index, (x, logits, activations) in enumerate(rows):
        add_overview_row(
            axes[row_index],
            row_index=row_index,
            x=x,
            logits=logits,
            activations=activations,
            palette=palette,
            preview_top_k=preview_top_k,
        )
    figure.tight_layout(w_pad=0.15, h_pad=0.0)
    save_figure(figure, path, dpi=dpi)
    plt.close(figure)


def write_report(
    path: Path,
    *,
    logits_onnx_path: Path,
    top1_onnx_path: Path | None,
    stage_model_path: Path,
    stage_infos: list[StageOutputInfo],
    records: list[SampleRecord],
    overview_path: Path,
) -> None:
    lines = [
        "# TitLeNet Student FP16 PTQ ONNX Activation Heatmaps",
        "",
        "## Configuration",
        "",
        f"- logits_onnx: `{logits_onnx_path}`",
        f"- top1_onnx: `{top1_onnx_path}`",
        f"- stage_output_model: `{stage_model_path}`",
        "- provider: `CPUExecutionProvider`",
        "",
        "## Stage Outputs",
        "",
        "| stage | tensor | elem_type | shape |",
        "| --- | --- | ---: | --- |",
    ]
    for info in stage_infos:
        lines.append(
            f"| `{info.name}` | `{info.tensor_name}` | {info.elem_type} | `{list(info.shape)}` |"
        )
    lines.extend(
        [
            "",
            "## Overview",
            "",
            f"![overview]({markdown_image_path(path, overview_path)})",
            "",
            "## Samples",
            "",
            "| sample | image_id | dataset_index | top1 | p(top1) | top1 model output | match | figure |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for record in records:
        top1_output = "-" if record.top1_model_output is None else str(record.top1_model_output)
        top1_match = "-" if record.top1_model_match is None else str(record.top1_model_match)
        lines.append(
            f"| {record.sample_number} | `{record.image_id}` | {record.dataset_index} | "
            f"{record.top1_palette_id} | {record.top1_probability:.4f} | {top1_output} | "
            f"{top1_match} | [figure]({markdown_image_path(path, record.sample_figure_path)}) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def overview_path(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.overview_path is not None:
        return resolve_path(args.overview_path)
    return output_dir / DEFAULT_OVERVIEW_NAME


def run(args: argparse.Namespace) -> list[SampleRecord]:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_onnx_path = resolve_path(args.logits_onnx, must_exist=True)
    top1_onnx_path = resolve_path(args.top1_onnx, must_exist=True)
    stage_model_path = (
        resolve_path(args.stage_model_output)
        if args.stage_model_output is not None
        else output_dir / "titlenet_student_fp16_ptq_logits_with_stage_outputs.onnx"
    )
    stage_model_path, stage_infos = save_stage_output_model(
        logits_onnx_path,
        stage_model_path,
    )
    stage_session = ort.InferenceSession(
        str(stage_model_path),
        providers=["CPUExecutionProvider"],
    )
    top1_session = ort.InferenceSession(
        str(top1_onnx_path),
        providers=["CPUExecutionProvider"],
    )
    dataset = create_dataset(args)
    indices = select_dataset_indices(
        dataset,
        sample_count=args.sample_count,
        raw_indices=args.sample_indices,
        raw_image_ids=args.image_ids,
    )
    palette = load_palette(resolve_path(args.palette, must_exist=True))
    rows_for_overview: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    records: list[SampleRecord] = []
    for sample_number, dataset_index in enumerate(indices, start=1):
        sample = dataset[dataset_index]
        image_id = str(sample["image_id"])
        x = input_tensor(sample)
        logits, activations = run_stage_session(stage_session, x)
        top1_output = run_top1_session(top1_session, x)
        predictions = topk_predictions(logits, limit=args.preview_top_k)
        top1_palette_id, top1_probability = predictions[0]
        sample_figure_path = (
            output_dir
            / f"sample_{sample_number:03d}_{safe_file_stem(image_id)}_onnx_activation_heatmap.png"
        )
        write_sample_figure(
            sample_figure_path,
            image_id=image_id,
            x=x,
            logits=logits,
            activations=activations,
            palette=palette,
            preview_top_k=args.preview_top_k,
            dpi=args.dpi,
        )
        rows_for_overview.append((x, logits, activations))
        records.append(
            SampleRecord(
                sample_number=sample_number,
                dataset_index=int(dataset_index),
                image_id=image_id,
                top1_palette_id=top1_palette_id,
                top1_probability=top1_probability,
                top1_model_output=top1_output,
                top1_model_match=None if top1_output is None else top1_output == top1_palette_id,
                sample_figure_path=sample_figure_path,
            )
        )
    overview_figure_path = overview_path(args, output_dir)
    write_overview_figure(
        overview_figure_path,
        rows=rows_for_overview,
        palette=palette,
        preview_top_k=args.preview_top_k,
        dpi=args.dpi,
    )
    write_report(
        output_dir / "titlenet_student_fp16_ptq_onnx_activation_report.md",
        logits_onnx_path=logits_onnx_path,
        top1_onnx_path=top1_onnx_path,
        stage_model_path=stage_model_path,
        stage_infos=stage_infos,
        records=records,
        overview_path=overview_figure_path,
    )
    metrics = {
        "logits_onnx": str(logits_onnx_path),
        "top1_onnx": str(top1_onnx_path),
        "stage_output_model": str(stage_model_path),
        "stage_outputs": [
            {
                "stage": info.name,
                "tensor": info.tensor_name,
                "elem_type": info.elem_type,
                "shape": list(info.shape),
            }
            for info in stage_infos
        ],
        "sample_count": len(records),
        "top1_model_match_count": sum(
            1 for record in records if record.top1_model_match is True
        ),
        "records": [
            {
                "sample_number": record.sample_number,
                "dataset_index": record.dataset_index,
                "image_id": record.image_id,
                "top1_palette_id": record.top1_palette_id,
                "top1_probability": record.top1_probability,
                "top1_model_output": record.top1_model_output,
                "top1_model_match": record.top1_model_match,
                "sample_figure_path": str(record.sample_figure_path),
            }
            for record in records
        ],
    }
    (output_dir / "titlenet_student_fp16_ptq_onnx_activation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("wrote ONNX activation heatmaps to %s", output_dir)
    return records


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
