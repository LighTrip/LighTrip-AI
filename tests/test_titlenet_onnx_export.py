from __future__ import annotations

import json
from pathlib import Path

from scripts.title_color_recommendation import export_titlenet_onnx


def test_default_export_paths_use_expected_filenames(tmp_path: Path) -> None:
    paths = export_titlenet_onnx.export_paths(
        output_dir=tmp_path,
        logits_output=None,
        top1_output=None,
    )

    assert paths.logits_output == tmp_path / "titlenet_logits.onnx"
    assert paths.top1_output == tmp_path / "titlenet_top1.onnx"


def test_mobile_inference_config_declares_logits_and_top1_exports() -> None:
    config_path = Path("configs/title_color_recommendation/titlenet_mobile_inference.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["model_input"]["shape"] == [1, 4, 36, 136]
    assert config["model_outputs"]["logits"]["shape"] == [1, 32]
    assert config["model_outputs"]["top1"]["shape"] == [1]
    assert config["model_outputs"]["top1"]["dtype"] == "int64"
    assert config["model_outputs"]["top1"]["postprocess"] == "argmax"
