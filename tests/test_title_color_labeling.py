from __future__ import annotations

import numpy as np
from PIL import Image

from src.title_color_recommendation.labeling.soft_labels import (
    PaletteColor,
    SoftLabelConfig,
    compute_pseudo_score,
    compute_fail_penalty,
    compute_image_soft_labels,
    soft_label_config_from_mapping,
    softmax,
)


def sample_palette() -> list[PaletteColor]:
    return [
        PaletteColor(
            id=0,
            name="white",
            hex="#FFFFFF",
            group="neutral_light",
            rgb=(255, 255, 255),
            lab=(100.0, 0.0, 0.0),
            relative_luminance=1.0,
            aesthetic_prior=0.95,
        ),
        PaletteColor(
            id=1,
            name="black",
            hex="#000000",
            group="neutral_dark",
            rgb=(0, 0, 0),
            lab=(0.0, 0.0, 0.0),
            relative_luminance=0.0,
            aesthetic_prior=0.95,
        ),
        PaletteColor(
            id=2,
            name="blue",
            hex="#3B82F6",
            group="accent",
            rgb=(59, 130, 246),
            lab=(56.56, 20.16, -63.61),
            relative_luminance=0.235,
            aesthetic_prior=0.90,
        ),
    ]


def test_compute_image_soft_labels_prefers_light_text_on_dark_background() -> None:
    roi = Image.new("RGB", (32, 16), (0, 0, 0))
    mask = Image.new("L", (32, 16), 255)
    result = compute_image_soft_labels(
        roi,
        mask,
        sample_palette(),
        SoftLabelConfig(temperature=0.2),
    )

    assert result.contrast_p05[0] > 20.0
    assert np.isclose(float(result.contrast_p05[1]), 1.0)
    assert bool(result.wcag_pass[0])
    assert not bool(result.wcag_pass[1])
    assert int(np.argmax(result.target_distribution)) == 0
    assert np.isclose(float(result.target_distribution.sum()), 1.0)


def test_compute_image_soft_labels_uses_text_mask_pixels_for_contrast() -> None:
    roi = Image.new("RGB", (8, 4), (255, 255, 255))
    mask = Image.new("L", (8, 4), 0)
    for x in range(4):
        for y in range(2):
            roi.putpixel((x, y), (0, 0, 0))
            mask.putpixel((x, y), 255)

    result = compute_image_soft_labels(
        roi,
        mask,
        sample_palette(),
        SoftLabelConfig(temperature=0.2),
    )

    assert result.background.text_pixel_count == 8
    assert result.contrast_p05[0] > 20.0
    assert np.isclose(float(result.contrast_p05[1]), 1.0)


def test_fail_penalty_increases_when_low_percentiles_fail() -> None:
    config = SoftLabelConfig(contrast_threshold=4.5, min_contrast_p05=3.0)
    penalty = compute_fail_penalty(
        np.asarray([3.2, 2.0], dtype=np.float32),
        np.asarray([4.8, 2.5], dtype=np.float32),
        config,
    )

    assert np.isclose(float(penalty[0]), 0.0)
    assert penalty[1] > 0.0


def test_softmax_temperature_controls_sharpness() -> None:
    scores = np.asarray([0.1, 0.3, 0.9], dtype=np.float32)

    cold = softmax(scores, temperature=0.2)
    warm = softmax(scores, temperature=1.0)

    assert np.isclose(float(cold.sum()), 1.0)
    assert np.isclose(float(warm.sum()), 1.0)
    assert float(cold.max()) > float(warm.max())


def test_soft_label_config_accepts_top_level_fail_penalty_weight() -> None:
    config = soft_label_config_from_mapping(
        {
            "labeling": {
                "temperature": 0.15,
                "fail_penalty": 0.35,
                "weights": {
                    "readability_score": 0.65,
                    "aesthetic_prior": 0.15,
                    "tone_match_score": 0.15,
                    "simplicity_score": 0.05,
                },
            }
        }
    )

    assert np.isclose(config.temperature, 0.15)
    assert np.isclose(config.fail_penalty_weight, 0.35)


def test_compute_pseudo_score_uses_configured_component_weights() -> None:
    config = SoftLabelConfig(
        readability_weight=0.65,
        aesthetic_weight=0.15,
        tone_match_weight=0.15,
        simplicity_weight=0.05,
        fail_penalty_weight=0.35,
    )

    pseudo_score = compute_pseudo_score(
        readability_score=np.asarray([0.8], dtype=np.float32),
        aesthetic_prior=np.asarray([0.6], dtype=np.float32),
        tone_match_score=np.asarray([0.7], dtype=np.float32),
        simplicity_score=np.asarray([0.5], dtype=np.float32),
        fail_penalty=np.asarray([0.2], dtype=np.float32),
        config=config,
    )

    expected = (0.65 * 0.8) + (0.15 * 0.6) + (0.15 * 0.7) + (0.05 * 0.5) - (0.35 * 0.2)
    assert np.isclose(float(pseudo_score[0]), expected)
