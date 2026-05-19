from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PaletteColor:
    id: int
    name: str
    hex: str
    group: str
    rgb: tuple[int, int, int]
    lab: tuple[float, float, float]
    relative_luminance: float
    aesthetic_prior: float


@dataclass(frozen=True)
class SoftLabelConfig:
    temperature: float = 0.2
    contrast_threshold: float = 4.5
    min_contrast_p05: float = 3.0
    contrast_norm_min: float = 1.0
    contrast_norm_max: float = 7.0
    readability_p05_weight: float = 0.50
    readability_p10_weight: float = 0.35
    readability_mean_weight: float = 0.15
    readability_weight: float = 0.70
    aesthetic_weight: float = 0.15
    tone_match_weight: float = 0.10
    simplicity_weight: float = 0.05
    fail_penalty_weight: float = 0.30


@dataclass(frozen=True)
class BackgroundStats:
    text_pixel_count: int
    roi_pixel_count: int
    mean_luminance: float
    std_luminance: float
    p05_luminance: float
    p95_luminance: float
    mean_lab: tuple[float, float, float]
    chroma: float
    colorfulness: float
    complexity: float


@dataclass(frozen=True)
class ImageSoftLabelResult:
    contrast_p05: np.ndarray
    contrast_p10: np.ndarray
    contrast_mean: np.ndarray
    wcag_pass: np.ndarray
    readability_score: np.ndarray
    aesthetic_prior: np.ndarray
    tone_match_score: np.ndarray
    simplicity_score: np.ndarray
    fail_penalty: np.ndarray
    pseudo_score: np.ndarray
    target_distribution: np.ndarray
    background: BackgroundStats


def soft_label_config_from_mapping(mapping: Mapping[str, Any]) -> SoftLabelConfig:
    labeling = mapping.get("labeling") or {}
    weights = labeling.get("weights") or {}
    readability = weights.get("readability") or {}
    normalization = labeling.get("contrast_normalization") or {}

    return SoftLabelConfig(
        temperature=float(labeling.get("temperature", 0.2)),
        contrast_threshold=float(labeling.get("contrast_threshold", 4.5)),
        min_contrast_p05=float(labeling.get("min_contrast_p05", 3.0)),
        contrast_norm_min=float(normalization.get("min", 1.0)),
        contrast_norm_max=float(normalization.get("max", 7.0)),
        readability_p05_weight=float(readability.get("p05", 0.50)),
        readability_p10_weight=float(readability.get("p10", 0.35)),
        readability_mean_weight=float(readability.get("mean", 0.15)),
        readability_weight=float(weights.get("readability_score", 0.70)),
        aesthetic_weight=float(weights.get("aesthetic_prior", 0.15)),
        tone_match_weight=float(weights.get("tone_match_score", 0.10)),
        simplicity_weight=float(weights.get("simplicity_score", 0.05)),
        fail_penalty_weight=float(weights.get("fail_penalty", 0.30)),
    )


def load_palette(path: Path) -> list[PaletteColor]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"palette must be a list: {path}")

    palette: list[PaletteColor] = []
    for item in payload:
        palette.append(
            PaletteColor(
                id=int(item["id"]),
                name=str(item["name"]),
                hex=str(item["hex"]).upper(),
                group=str(item["group"]),
                rgb=tuple(int(channel) for channel in item["rgb"]),
                lab=tuple(float(value) for value in item["lab"]),
                relative_luminance=float(item["relative_luminance"]),
                aesthetic_prior=float(item["aesthetic_prior"]),
            )
        )
    palette.sort(key=lambda color: color.id)
    validate_palette(palette)
    return palette


def validate_palette(palette: list[PaletteColor]) -> None:
    ids = [color.id for color in palette]
    if ids != list(range(len(palette))):
        raise ValueError(f"palette ids must be continuous from 0: {ids}")
    if len(palette) == 0:
        raise ValueError("palette must contain at least one color")


def palette_luminance_array(palette: Iterable[PaletteColor]) -> np.ndarray:
    return np.asarray([color.relative_luminance for color in palette], dtype=np.float32)


def adjusted_aesthetic_array(palette: Iterable[PaletteColor]) -> np.ndarray:
    values: list[float] = []
    for color in palette:
        score = color.aesthetic_prior
        if color.group in {"neutral_light", "neutral_dark"}:
            extreme = abs(color.relative_luminance - 0.5) * 2.0
            if extreme > 0.80:
                score -= ((extreme - 0.80) / 0.20) * 0.40
        elif color.group == "cream":
            extreme = abs(color.relative_luminance - 0.5) * 2.0
            if extreme > 0.88:
                score -= ((extreme - 0.88) / 0.12) * 0.18
        values.append(min(1.0, max(0.0, score)))
    return np.asarray(values, dtype=np.float32)


def srgb_to_linear_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32) / 255.0
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def relative_luminance_image(rgb: np.ndarray) -> np.ndarray:
    linear = srgb_to_linear_array(rgb)
    return (
        (0.2126 * linear[..., 0])
        + (0.7152 * linear[..., 1])
        + (0.0722 * linear[..., 2])
    ).astype(np.float32)


def srgb_channel_to_linear(channel: float) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def rgb_to_lab(rgb: Iterable[float]) -> tuple[float, float, float]:
    red, green, blue = [srgb_channel_to_linear(float(channel)) for channel in rgb]

    x = (0.4124564 * red) + (0.3575761 * green) + (0.1804375 * blue)
    y = (0.2126729 * red) + (0.7151522 * green) + (0.0721750 * blue)
    z = (0.0193339 * red) + (0.1191920 * green) + (0.9503041 * blue)

    x /= 0.95047
    z /= 1.08883

    delta = 6 / 29

    def lab_f(value: float) -> float:
        if value > delta**3:
            return value ** (1 / 3)
        return (value / (3 * delta**2)) + (4 / 29)

    fx = lab_f(x)
    fy = lab_f(y)
    fz = lab_f(z)
    return ((116 * fy) - 16, 500 * (fx - fy), 200 * (fy - fz))


def wcag_contrast_against_background(
    candidate_luminance: np.ndarray,
    background_luminance: np.ndarray,
) -> np.ndarray:
    candidate = candidate_luminance[:, None]
    background = background_luminance[None, :]
    lighter = np.maximum(candidate, background)
    darker = np.minimum(candidate, background)
    return (lighter + 0.05) / (darker + 0.05)


def normalize_contrast(values: np.ndarray, config: SoftLabelConfig) -> np.ndarray:
    denominator = max(1e-6, config.contrast_norm_max - config.contrast_norm_min)
    return np.clip((values - config.contrast_norm_min) / denominator, 0.0, 1.0)


def softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive: {temperature}")
    logits = values.astype(np.float64) / temperature
    logits = logits - float(np.max(logits))
    exp = np.exp(logits)
    total = float(np.sum(exp))
    if total <= 0 or not math.isfinite(total):
        raise ValueError("softmax produced an invalid denominator")
    return (exp / total).astype(np.float32)


def colorfulness_score(rgb: np.ndarray) -> float:
    values = rgb.astype(np.float32)
    red = values[..., 0]
    green = values[..., 1]
    blue = values[..., 2]
    rg = red - green
    yb = (0.5 * (red + green)) - blue
    std_root = math.sqrt(float(np.std(rg)) ** 2 + float(np.std(yb)) ** 2)
    mean_root = math.sqrt(float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2)
    return min(1.0, max(0.0, (std_root + (0.3 * mean_root)) / 255.0))


def compute_background_stats(
    roi_rgb: np.ndarray,
    mask: np.ndarray,
    luminance: np.ndarray,
) -> tuple[BackgroundStats, np.ndarray]:
    text_mask = mask > 0
    if int(np.count_nonzero(text_mask)) < 8:
        text_mask = np.ones(mask.shape, dtype=bool)

    text_luminance = luminance[text_mask].astype(np.float32)
    full_luminance = luminance.reshape(-1).astype(np.float32)
    mean_rgb = roi_rgb.reshape(-1, 3).mean(axis=0)
    mean_lab = rgb_to_lab(mean_rgb)
    chroma = math.sqrt((mean_lab[1] ** 2) + (mean_lab[2] ** 2))
    colorfulness = colorfulness_score(roi_rgb)
    std_luminance = float(np.std(full_luminance))
    complexity = min(1.0, max(0.0, (0.70 * (std_luminance / 0.28)) + (0.30 * colorfulness)))

    stats = BackgroundStats(
        text_pixel_count=int(text_luminance.size),
        roi_pixel_count=int(full_luminance.size),
        mean_luminance=float(np.mean(full_luminance)),
        std_luminance=std_luminance,
        p05_luminance=float(np.percentile(full_luminance, 5)),
        p95_luminance=float(np.percentile(full_luminance, 95)),
        mean_lab=mean_lab,
        chroma=float(chroma),
        colorfulness=float(colorfulness),
        complexity=float(complexity),
    )
    return stats, text_luminance


def candidate_hue(lab: tuple[float, float, float]) -> float:
    return math.degrees(math.atan2(lab[2], lab[1])) % 360.0


def hue_distance(a: float, b: float) -> float:
    distance = abs(a - b) % 360.0
    return min(distance, 360.0 - distance)


def group_tone_score(group: str, background_luminance: float) -> float:
    if background_luminance < 0.33:
        scores = {
            "neutral_light": 0.95,
            "cream": 0.92,
            "pastel": 0.86,
            "accent": 0.72,
            "muted": 0.58,
            "deep": 0.28,
            "neutral_dark": 0.20,
        }
    elif background_luminance > 0.67:
        scores = {
            "neutral_dark": 0.95,
            "deep": 0.92,
            "muted": 0.78,
            "accent": 0.70,
            "pastel": 0.48,
            "cream": 0.28,
            "neutral_light": 0.24,
        }
    else:
        scores = {
            "accent": 0.84,
            "deep": 0.80,
            "muted": 0.72,
            "neutral_dark": 0.72,
            "neutral_light": 0.70,
            "cream": 0.68,
            "pastel": 0.64,
        }
    return scores.get(group, 0.60)


def compute_tone_match_scores(
    palette: list[PaletteColor],
    background: BackgroundStats,
) -> np.ndarray:
    bg_hue = candidate_hue(background.mean_lab)
    bg_chroma = background.chroma
    values: list[float] = []
    for color in palette:
        group_score = group_tone_score(color.group, background.mean_luminance)
        candidate_chroma = math.sqrt((color.lab[1] ** 2) + (color.lab[2] ** 2))
        if bg_chroma < 8 or candidate_chroma < 8:
            harmony = 0.68
        else:
            distance = hue_distance(bg_hue, candidate_hue(color.lab))
            analogous = math.exp(-((distance / 55.0) ** 2))
            complementary = math.exp(-(((distance - 180.0) / 65.0) ** 2))
            harmony = 0.50 + (0.50 * max(analogous, complementary * 0.90))
        values.append(min(1.0, max(0.0, (0.70 * group_score) + (0.30 * harmony))))
    return np.asarray(values, dtype=np.float32)


def compute_simplicity_scores(
    palette: list[PaletteColor],
    background: BackgroundStats,
) -> np.ndarray:
    simple_scores = {
        "neutral_light": 0.95,
        "cream": 0.90,
        "neutral_dark": 0.95,
        "deep": 0.82,
        "muted": 0.78,
        "pastel": 0.58,
        "accent": 0.48,
    }
    expressive_scores = {
        "accent": 0.86,
        "pastel": 0.76,
        "deep": 0.68,
        "muted": 0.62,
        "cream": 0.58,
        "neutral_light": 0.56,
        "neutral_dark": 0.56,
    }
    complexity = background.complexity
    values = [
        (complexity * simple_scores.get(color.group, 0.60))
        + ((1.0 - complexity) * expressive_scores.get(color.group, 0.60))
        for color in palette
    ]
    return np.asarray(values, dtype=np.float32)


def compute_fail_penalty(
    contrast_p10: np.ndarray,
    contrast_mean: np.ndarray,
    config: SoftLabelConfig,
) -> np.ndarray:
    p10_deficit = np.clip(
        (config.min_contrast_p05 - contrast_p10) / max(config.min_contrast_p05, 1e-6),
        0.0,
        1.0,
    )
    mean_deficit = np.clip(
        (config.contrast_threshold - contrast_mean) / max(config.contrast_threshold, 1e-6),
        0.0,
        1.0,
    )
    return np.clip((0.70 * p10_deficit) + (0.30 * mean_deficit), 0.0, 1.0).astype(
        np.float32
    )


def compute_image_soft_labels(
    roi_image: Image.Image,
    mask_image: Image.Image,
    palette: list[PaletteColor],
    config: SoftLabelConfig,
    *,
    temperature: float | None = None,
) -> ImageSoftLabelResult:
    roi_rgb = np.asarray(roi_image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
    if mask.shape != roi_rgb.shape[:2]:
        raise ValueError(f"mask size must match ROI size: mask={mask.shape}, roi={roi_rgb.shape[:2]}")

    luminance = relative_luminance_image(roi_rgb)
    background, text_luminance = compute_background_stats(roi_rgb, mask, luminance)
    palette_luminance = palette_luminance_array(palette)
    contrasts = wcag_contrast_against_background(palette_luminance, text_luminance)

    contrast_p05 = np.percentile(contrasts, 5, axis=1).astype(np.float32)
    contrast_p10 = np.percentile(contrasts, 10, axis=1).astype(np.float32)
    contrast_mean = np.mean(contrasts, axis=1).astype(np.float32)
    wcag_pass = (contrast_mean >= config.contrast_threshold) & (
        contrast_p10 >= config.min_contrast_p05
    )

    readability_score = (
        (config.readability_p05_weight * normalize_contrast(contrast_p05, config))
        + (config.readability_p10_weight * normalize_contrast(contrast_p10, config))
        + (config.readability_mean_weight * normalize_contrast(contrast_mean, config))
    ).astype(np.float32)

    aesthetic_prior = adjusted_aesthetic_array(palette)
    tone_match_score = compute_tone_match_scores(palette, background)
    simplicity_score = compute_simplicity_scores(palette, background)
    fail_penalty = compute_fail_penalty(contrast_p10, contrast_mean, config)
    pseudo_score = (
        (config.readability_weight * readability_score)
        + (config.aesthetic_weight * aesthetic_prior)
        + (config.tone_match_weight * tone_match_score)
        + (config.simplicity_weight * simplicity_score)
        - (config.fail_penalty_weight * fail_penalty)
    ).astype(np.float32)
    target_distribution = softmax(
        pseudo_score,
        temperature=config.temperature if temperature is None else temperature,
    )

    return ImageSoftLabelResult(
        contrast_p05=contrast_p05,
        contrast_p10=contrast_p10,
        contrast_mean=contrast_mean,
        wcag_pass=wcag_pass,
        readability_score=readability_score,
        aesthetic_prior=aesthetic_prior,
        tone_match_score=tone_match_score,
        simplicity_score=simplicity_score,
        fail_penalty=fail_penalty,
        pseudo_score=pseudo_score,
        target_distribution=target_distribution,
        background=background,
    )


def distribution_entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def normalized_distribution_entropy(probabilities: np.ndarray) -> float:
    if probabilities.size <= 1:
        return 0.0
    return distribution_entropy(probabilities) / math.log(float(probabilities.size))
