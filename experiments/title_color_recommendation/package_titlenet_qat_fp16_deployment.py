from __future__ import annotations

import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.title_color_recommendation.path_utils import (
    resolve_project_path as resolve_inside_project,
)
from experiments.title_color_recommendation.prepare_titlenet_student_quantization_baseline import (
    path_size_mb,
)
from scripts.title_color_recommendation.export_titlenet_onnx import (
    DEFAULT_INPUT_SHAPE,
    DEFAULT_NUM_CLASSES,
    LOGITS_OUTPUT_NAME,
    TOP1_OUTPUT_NAME,
)


SOURCE_LOGITS_ONNX = Path(
    "outputs/title_color_recommendation/quantization/qat_kd90/"
    "titlenet_student_warm_kd90_fp16_logits.onnx"
)
SOURCE_TOP1_ONNX = Path(
    "outputs/title_color_recommendation/quantization/qat_kd90/"
    "titlenet_student_warm_kd90_fp16_top1.onnx"
)
SOURCE_PALETTE = Path("data/title_color_recommendation/processed/palette.json")
SOURCE_QUANTIZATION_METRICS = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_qat_kd90_quantization_metrics.json"
)
SOURCE_QAT_METRICS = Path(
    "outputs/reports/model_evaluation/"
    "titlenet_student_qat_kd_90_10_metrics.json"
)
DEPLOYMENT_DIR = Path("outputs/title_color_recommendation/deployment")
DEPLOYMENT_LOGITS_ONNX = DEPLOYMENT_DIR / "titlenet_student_qat_fp16_logits.onnx"
DEPLOYMENT_TOP1_ONNX = DEPLOYMENT_DIR / "titlenet_student_qat_fp16_top1.onnx"
DEPLOYMENT_PALETTE = DEPLOYMENT_DIR / "palette.json"
DEPLOYMENT_METADATA = DEPLOYMENT_DIR / "titlenet_student_qat_fp16_metadata.json"
DEPLOYMENT_REPORT = Path(
    "outputs/reports/model_evaluation/onnx/"
    "titlenet_student_qat_fp16_deployment_report.md"
)
DEPLOYMENT_METADATA_OUTPUT_PATH = PROJECT_ROOT / DEPLOYMENT_METADATA
DEPLOYMENT_REPORT_OUTPUT_PATH = PROJECT_ROOT / DEPLOYMENT_REPORT


@dataclass(frozen=True)
class OnnxTensorSpec:
    name: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class DeploymentValidation:
    logits_input: OnnxTensorSpec
    logits_output: OnnxTensorSpec
    top1_input: OnnxTensorSpec
    top1_output: OnnxTensorSpec
    palette_ids_valid: bool
    palette_count: int


def project_path(path: Path) -> Path:
    return resolve_inside_project(PROJECT_ROOT, path)


def display_path(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_json(path: Path) -> Any:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def deployment_metadata_path() -> Path:
    return DEPLOYMENT_METADATA_OUTPUT_PATH


def deployment_report_path() -> Path:
    return DEPLOYMENT_REPORT_OUTPUT_PATH


def require_mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must be a mapping")
    return value


def tensor_dtype_name(elem_type: int) -> str:
    import onnx

    return onnx.TensorProto.DataType.Name(elem_type).lower()


def tensor_spec(value_info: Any) -> OnnxTensorSpec:
    tensor_type = value_info.type.tensor_type
    return OnnxTensorSpec(
        name=str(value_info.name),
        shape=[int(dimension.dim_value) for dimension in tensor_type.shape.dim],
        dtype=tensor_dtype_name(int(tensor_type.elem_type)),
    )


def validate_onnx_pair(
    *,
    logits_path: Path,
    top1_path: Path,
) -> tuple[OnnxTensorSpec, OnnxTensorSpec, OnnxTensorSpec, OnnxTensorSpec]:
    import onnx

    logits_model = onnx.load(str(project_path(logits_path)))
    top1_model = onnx.load(str(project_path(top1_path)))
    onnx.checker.check_model(logits_model)
    onnx.checker.check_model(top1_model)

    logits_input = tensor_spec(logits_model.graph.input[0])
    logits_output = tensor_spec(logits_model.graph.output[0])
    top1_input = tensor_spec(top1_model.graph.input[0])
    top1_output = tensor_spec(top1_model.graph.output[0])

    expected_input = list(DEFAULT_INPUT_SHAPE)
    if logits_input.name != "input" or top1_input.name != "input":
        raise ValueError("deployment ONNX input name must be 'input'")
    if logits_input.shape != expected_input or top1_input.shape != expected_input:
        raise ValueError(
            f"deployment ONNX input shape must be {expected_input}: "
            f"logits={logits_input.shape}, top1={top1_input.shape}"
        )
    if logits_input.dtype != "float" or top1_input.dtype != "float":
        raise TypeError("deployment ONNX input dtype must be float32")
    if logits_output.name != LOGITS_OUTPUT_NAME:
        raise ValueError(f"logits output name must be {LOGITS_OUTPUT_NAME}")
    if logits_output.shape != [1, DEFAULT_NUM_CLASSES]:
        raise ValueError(f"logits output shape must be [1, {DEFAULT_NUM_CLASSES}]")
    if logits_output.dtype != "float":
        raise TypeError("logits output dtype must be float32")
    if top1_output.name != TOP1_OUTPUT_NAME:
        raise ValueError(f"top1 output name must be {TOP1_OUTPUT_NAME}")
    if top1_output.shape != [1]:
        raise ValueError("top1 output shape must be [1]")
    if top1_output.dtype != "int64":
        raise TypeError("top1 output dtype must be int64")

    return logits_input, logits_output, top1_input, top1_output


def validate_palette(path: Path) -> tuple[bool, int]:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise TypeError("palette payload must be a list")
    ids = {int(item["id"]) for item in payload}
    expected_ids = set(range(DEFAULT_NUM_CLASSES))
    return ids == expected_ids, len(payload)


def fp16_quantization_result(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for item in payload.get("results", []):
        if isinstance(item, Mapping) and item.get("name") == "fp16":
            return item
    raise KeyError("fp16 quantization result not found")


def copy_deployment_artifacts() -> None:
    project_path(DEPLOYMENT_DIR).mkdir(parents=True, exist_ok=True)
    shutil.copy2(project_path(SOURCE_LOGITS_ONNX), project_path(DEPLOYMENT_LOGITS_ONNX))
    shutil.copy2(project_path(SOURCE_TOP1_ONNX), project_path(DEPLOYMENT_TOP1_ONNX))
    shutil.copy2(project_path(SOURCE_PALETTE), project_path(DEPLOYMENT_PALETTE))
    normalize_onnx_graph(DEPLOYMENT_LOGITS_ONNX)
    normalize_onnx_graph(DEPLOYMENT_TOP1_ONNX)


def normalize_onnx_graph(path: Path) -> None:
    import onnx

    model = onnx.load(str(project_path(path)))
    available = {
        value.name
        for value in (
            *model.graph.input,
            *model.graph.initializer,
            *model.graph.sparse_initializer,
        )
    }
    unsorted_nodes = list(model.graph.node)
    sorted_nodes = []
    while unsorted_nodes:
        progress = False
        remaining = []
        for node in unsorted_nodes:
            if all(input_name == "" or input_name in available for input_name in node.input):
                sorted_nodes.append(node)
                available.update(node.output)
                progress = True
            else:
                remaining.append(node)
        if not progress:
            missing = sorted(
                {
                    input_name
                    for node in remaining
                    for input_name in node.input
                    if input_name and input_name not in available
                }
            )
            raise ValueError(
                f"cannot topologically sort ONNX graph {path}: missing={missing[:5]}"
            )
        unsorted_nodes = remaining

    del model.graph.node[:]
    model.graph.node.extend(sorted_nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(project_path(path)))


def build_metadata(validation: DeploymentValidation) -> dict[str, Any]:
    quantization_metrics = require_mapping(
        load_json(SOURCE_QUANTIZATION_METRICS),
        description="QAT quantization metrics",
    )
    qat_metrics = require_mapping(
        load_json(SOURCE_QAT_METRICS),
        description="QAT training metrics",
    )
    fp16_result = fp16_quantization_result(quantization_metrics)
    fp16_validation = require_mapping(
        fp16_result.get("validation"),
        description="FP16 validation metrics",
    )
    return {
        "model_id": "titlenet_student_qat_fp16",
        "source_model": "titlenet_student_qat_kd90",
        "precision": "fp16_internal_float32_io",
        "deployment_candidate_rank": 1,
        "files": {
            "logits_onnx": display_path(project_path(DEPLOYMENT_LOGITS_ONNX)),
            "top1_onnx": display_path(project_path(DEPLOYMENT_TOP1_ONNX)),
            "palette": display_path(project_path(DEPLOYMENT_PALETTE)),
        },
        "input": {
            "name": "input",
            "shape": list(DEFAULT_INPUT_SHAPE),
            "dtype": "float32",
            "channel_order": ["R", "G", "B", "mask"],
            "rgb_range": [0.0, 1.0],
            "mask_range": [0.0, 1.0],
        },
        "outputs": {
            "logits": {
                "name": LOGITS_OUTPUT_NAME,
                "shape": [1, DEFAULT_NUM_CLASSES],
                "dtype": "float32",
                "meaning": "raw logits for 32 palette colors",
            },
            "top1": {
                "name": TOP1_OUTPUT_NAME,
                "shape": [1],
                "dtype": "int64",
                "meaning": "top-1 palette color index in 0..31",
            },
        },
        "palette": {
            "count": validation.palette_count,
            "ids": "0..31",
            "id_mapping": "top1_index maps to palette.json item.id",
        },
        "validation": {
            "sample_count": fp16_validation.get("sample_count"),
            "top1_agreement": fp16_validation.get("top1_model_agreement"),
            "top3_agreement": fp16_validation.get("top3_agreement"),
            "top5_agreement": fp16_validation.get("top5_agreement"),
            "ndcg_at_3": fp16_validation.get("quantized_ndcg_at_3"),
            "ndcg_at_5": fp16_validation.get("quantized_ndcg_at_5"),
            "ndcg_at_3_drop": fp16_validation.get("ndcg_at_3_drop"),
            "ndcg_at_5_drop": fp16_validation.get("ndcg_at_5_drop"),
            "max_abs_diff": fp16_validation.get("max_abs_diff"),
            "mean_abs_diff": fp16_validation.get("mean_abs_diff"),
            "passed": fp16_validation.get("passed"),
        },
        "runtime_reference": {
            "python_onnxruntime_cpu_batch1_ms": fp16_result.get("latency", {})
            .get("batch1_logits", {})
            .get("inference_time_ms"),
            "logits_size_mb": path_size_mb(project_path(DEPLOYMENT_LOGITS_ONNX)),
            "top1_size_mb": path_size_mb(project_path(DEPLOYMENT_TOP1_ONNX)),
        },
        "qat_training": {
            "best_epoch": qat_metrics.get("best_epoch"),
            "best_metric": qat_metrics.get("best_metric"),
            "best_metric_value": qat_metrics.get("best_metric_value"),
            "test_metrics": qat_metrics.get("test_metrics"),
            "teacher_agreement": qat_metrics.get("test_teacher_agreement"),
        },
        "onnx_validation": asdict(validation),
        "mobile_integration_notes": [
            "Run top1 ONNX for final deployment output.",
            "Keep preprocessing outside the model: resize/crop ROI to 36x136 and stack RGB/mask channels.",
            "Input tensor remains float32 even though the model uses FP16 internally.",
            "Verify actual React Native runtime support and device latency separately.",
        ],
    }


def write_metadata(payload: Mapping[str, Any]) -> None:
    metadata_path = deployment_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_report(metadata: Mapping[str, Any]) -> None:
    validation = require_mapping(metadata["validation"], description="validation")
    runtime = require_mapping(metadata["runtime_reference"], description="runtime")
    files = require_mapping(metadata["files"], description="files")
    lines = [
        "# TitLeNet Student QAT FP16 Deployment Package",
        "",
        "## Summary",
        "",
        f"- model_id: `{metadata['model_id']}`",
        f"- source_model: `{metadata['source_model']}`",
        f"- precision: `{metadata['precision']}`",
        "- deployment_candidate_rank: `1`",
        "",
        "## Files",
        "",
        "| item | path |",
        "| --- | --- |",
        f"| logits_onnx | `{files['logits_onnx']}` |",
        f"| top1_onnx | `{files['top1_onnx']}` |",
        f"| palette | `{files['palette']}` |",
        f"| metadata | `{display_path(project_path(DEPLOYMENT_METADATA))}` |",
        "",
        "## Interface",
        "",
        "| item | value |",
        "| --- | --- |",
        "| input_shape | `[1, 4, 36, 136]` |",
        "| input_dtype | `float32` |",
        "| channels | `R, G, B, mask` |",
        "| rgb_range | `0..1` |",
        "| mask_range | `0/1` |",
        "| logits_output | `[1, 32] float32` |",
        "| top1_output | `[1] int64` |",
        "| final_output | `palette index 0..31` |",
        "",
        "## Validation",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| sample_count | {format_metric(validation.get('sample_count'))} |",
        f"| top1_agreement | {format_metric(validation.get('top1_agreement'))} |",
        f"| top3_agreement | {format_metric(validation.get('top3_agreement'))} |",
        f"| top5_agreement | {format_metric(validation.get('top5_agreement'))} |",
        f"| NDCG@3 | {format_metric(validation.get('ndcg_at_3'))} |",
        f"| NDCG@5 | {format_metric(validation.get('ndcg_at_5'))} |",
        f"| NDCG@5_drop | {format_metric(validation.get('ndcg_at_5_drop'))} |",
        f"| max_abs_diff | {format_metric(validation.get('max_abs_diff'))} |",
        "",
        "## Runtime Reference",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| logits_size_mb | {format_metric(runtime.get('logits_size_mb'))} |",
        f"| top1_size_mb | {format_metric(runtime.get('top1_size_mb'))} |",
        (
            f"| Python ONNX Runtime CPU batch1 ms | "
            f"{format_metric(runtime.get('python_onnxruntime_cpu_batch1_ms'))} |"
        ),
        "",
        "## Mobile Notes",
        "",
        "- Use the top-1 ONNX model for the final app output.",
        "- Confirm React Native runtime support for FP16 internal ops before final release.",
        "- Measure latency on the target Android/iOS device; Python CPU latency is only a reference.",
    ]
    report_path = deployment_report_path()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_deployment() -> Mapping[str, Any]:
    copy_deployment_artifacts()
    logits_input, logits_output, top1_input, top1_output = validate_onnx_pair(
        logits_path=DEPLOYMENT_LOGITS_ONNX,
        top1_path=DEPLOYMENT_TOP1_ONNX,
    )
    palette_ids_valid, palette_count = validate_palette(DEPLOYMENT_PALETTE)
    if not palette_ids_valid:
        raise ValueError("deployment palette ids must be exactly 0..31")
    validation = DeploymentValidation(
        logits_input=logits_input,
        logits_output=logits_output,
        top1_input=top1_input,
        top1_output=top1_output,
        palette_ids_valid=palette_ids_valid,
        palette_count=palette_count,
    )
    metadata = build_metadata(validation)
    write_metadata(metadata)
    write_report(metadata)
    return metadata


def main() -> int:
    metadata = package_deployment()
    files = require_mapping(metadata["files"], description="files")
    print("Packaged TitLeNet QAT FP16 deployment artifacts")
    print(f"Logits ONNX: {files['logits_onnx']}")
    print(f"Top1 ONNX: {files['top1_onnx']}")
    print(f"Palette: {files['palette']}")
    print(f"Metadata: {display_path(project_path(DEPLOYMENT_METADATA))}")
    print(f"Report: {display_path(project_path(DEPLOYMENT_REPORT))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
