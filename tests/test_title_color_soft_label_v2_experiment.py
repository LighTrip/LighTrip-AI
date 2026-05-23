from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def write_tiny_labels(path: Path, fieldnames: list[str]) -> None:
    rows = [
        ("sample_a", 0, "ivory", "#FFFDF7", "cream", 0.9, 0.8, 0.6, 0.7, 0.0, 0.60, 0.55, 1),
        ("sample_a", 1, "black", "#000000", "neutral_dark", 0.2, 0.7, 0.9, 0.8, 0.5, 0.20, 0.25, 0),
        ("sample_a", 2, "blue", "#2563EB", "accent", 0.5, 0.9, 0.8, 0.4, 0.1, 0.40, 0.20, 1),
        ("sample_b", 0, "ivory", "#FFFDF7", "cream", 0.1, 0.8, 0.6, 0.7, 0.6, 0.15, 0.20, 0),
        ("sample_b", 1, "black", "#000000", "neutral_dark", 0.8, 0.7, 0.8, 0.8, 0.0, 0.70, 0.60, 1),
        ("sample_b", 2, "blue", "#2563EB", "accent", 0.4, 0.9, 0.9, 0.4, 0.2, 0.35, 0.20, 1),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for (
            image_id,
            palette_id,
            name,
            hex_value,
            group,
            readability,
            aesthetic,
            tone,
            simplicity,
            penalty,
            pseudo,
            probability,
            wcag,
        ) in rows:
            writer.writerow(
                {
                    "id": image_id,
                    "split": "test",
                    "palette_id": palette_id,
                    "color_name": name,
                    "color_hex": hex_value,
                    "color_group": group,
                    "readability_score": readability,
                    "aesthetic_prior": aesthetic,
                    "tone_match_score": tone,
                    "simplicity_score": simplicity,
                    "fail_penalty": penalty,
                    "pseudo_score": pseudo,
                    "target_probability": probability,
                    "temperature": 0.2,
                    "rank": palette_id + 1,
                    "wcag_pass": wcag,
                }
            )


def write_preview_inputs(root: Path) -> Path:
    roi_dir = root / "rois"
    mask_dir = root / "masks"
    roi_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for image_id, color in (("sample_a", (20, 30, 40)), ("sample_b", (220, 220, 220))):
        roi_path = roi_dir / f"{image_id}.jpg"
        mask_path = mask_dir / f"{image_id}.png"
        Image.new("RGB", (24, 12), color).save(roi_path)
        Image.new("L", (24, 12), 255).save(mask_path)
        rows.append(
            {
                "id": image_id,
                "roi_path": roi_path.relative_to(root).as_posix(),
                "mask_path": mask_path.relative_to(root).as_posix(),
            }
        )

    manifest_path = root / "test.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "roi_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_soft_label_v2_experiment_generates_labels_reports_and_preview(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    module = __import__(
        "experiments.title_color_recommendation.run_soft_label_v2_experiment",
        fromlist=["run"],
    )
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)

    config_path = tmp_path / "soft_label_v2.yaml"
    config_path.write_text(
        "\n".join(
            [
                "labeling:",
                "  temperature: 0.15",
                "  weights:",
                "    readability_score: 0.65",
                "    aesthetic_prior: 0.15",
                "    tone_match_score: 0.15",
                "    simplicity_score: 0.05",
                "    fail_penalty: 0.35",
            ]
        ),
        encoding="utf-8",
    )
    v1_labels = tmp_path / "labels_soft.csv"
    v1_matrix = tmp_path / "labels_matrix.npy"
    v2_labels = tmp_path / "labels_soft_v2.csv"
    v2_matrix = tmp_path / "labels_matrix_v2.npy"
    write_tiny_labels(v1_labels, list(module.LABEL_COLUMNS))
    np.save(
        v1_matrix,
        np.asarray([[0.55, 0.25, 0.20], [0.20, 0.60, 0.20]], dtype=np.float32),
    )
    preview_manifest = write_preview_inputs(tmp_path)

    args = module.parse_args(
        [
            "--config-path",
            str(config_path),
            "--v1-labels-soft",
            str(v1_labels),
            "--v1-labels-matrix",
            str(v1_matrix),
            "--v2-labels-soft",
            str(v2_labels),
            "--v2-labels-matrix",
            str(v2_matrix),
            "--summary-path",
            str(tmp_path / "soft_label_v2_summary.json"),
            "--analysis-report-path",
            str(tmp_path / "soft_label_v2_analysis.md"),
            "--comparison-report-path",
            str(tmp_path / "soft_label_v1_vs_v2_comparison.md"),
            "--preview-dir",
            str(tmp_path / "preview"),
            "--preview-manifest",
            str(preview_manifest),
            "--preview-count",
            "1",
            "--progress-every",
            "0",
        ]
    )

    result = module.run(args)

    matrix = np.load(v2_matrix)
    assert matrix.shape == (2, 3)
    assert np.allclose(matrix.sum(axis=1), np.ones(2), atol=1e-6)
    assert result.generation_result is not None
    assert result.generation_result.processed_images == 2
    assert result.analysis_report_path.exists()
    assert result.comparison_report_path.exists()
    assert result.preview_manifest_path is not None
    assert result.preview_manifest_path.exists()

    payload = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert payload["generation"]["processed_images"] == 2
    assert payload["v2"]["matrix_shape"] == [2, 3]
