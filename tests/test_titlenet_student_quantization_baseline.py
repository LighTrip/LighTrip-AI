from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.title_color_recommendation import (
    prepare_titlenet_student_quantization_baseline as prep,
)
from scripts.title_color_recommendation import export_titlenet_onnx


def test_export_paths_allow_student_summary_output(tmp_path: Path) -> None:
    paths = export_titlenet_onnx.export_paths(
        output_dir=tmp_path,
        logits_output=tmp_path / "student_logits.onnx",
        top1_output=tmp_path / "student_top1.onnx",
        summary_output=tmp_path / "student_summary.json",
    )

    assert paths.logits_output.name == "student_logits.onnx"
    assert paths.top1_output.name == "student_top1.onnx"
    assert paths.summary_output.name == "student_summary.json"


class FakeCalibrationDataset:
    def __getitem__(self, index: int) -> dict[str, object]:
        x = torch.zeros((4, 36, 136), dtype=torch.float32)
        x[:3] = 0.5
        x[3] = float(index % 2)
        return {"x": x, "image_id": f"sample/{index}"}


def test_write_calibration_samples_creates_manifest_and_arrays(tmp_path: Path) -> None:
    summary = prep.write_calibration_samples(
        dataset=FakeCalibrationDataset(),  # type: ignore[arg-type]
        indices=[2, 0],
        sample_dir=tmp_path / "samples",
        manifest_path=tmp_path / "manifest.json",
        checkpoint_path=Path("outputs/checkpoints/student.pt"),
        split="val",
        seed=42,
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    first_sample = np.load(tmp_path / "samples" / "sample_0000.npy")

    assert summary.sample_count == 2
    assert manifest["model_id"] == prep.MODEL_ID
    assert manifest["sample_count"] == 2
    assert manifest["input"]["shape"] == [1, 4, 36, 136]
    assert first_sample.shape == (1, 4, 36, 136)
    assert first_sample.dtype == np.float32


def test_validate_input_array_rejects_non_binary_mask() -> None:
    array = np.zeros((1, 4, 36, 136), dtype=np.float32)
    array[:, 3:] = 0.5

    with pytest.raises(ValueError, match="mask channel"):
        prep.validate_input_array(array, image_id="bad-mask")
