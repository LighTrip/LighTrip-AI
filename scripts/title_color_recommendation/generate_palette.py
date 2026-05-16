#!/usr/bin/env python3
"""Generate the fixed title color palette and preview image."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PALETTE_PATH = ROOT / "data" / "title_color_recommendation" / "processed" / "palette.json"
PREVIEW_PATH = ROOT / "outputs" / "title_color_recommendation" / "reports" / "palette_preview.png"
PALETTE_SIZE = 32

ALLOWED_GROUPS = {
    "neutral_light",
    "cream",
    "neutral_dark",
    "pastel",
    "accent",
    "deep",
    "muted",
}

EXPECTED_GROUP_COUNTS = {
    "neutral_light": 4,
    "cream": 4,
    "neutral_dark": 4,
    "pastel": 6,
    "accent": 8,
    "deep": 4,
    "muted": 2,
}

RAW_PALETTE = [
    ("pure_white", "#FFFFFF", "neutral_light", 0.98),
    ("soft_white", "#FAFAFA", "neutral_light", 0.97),
    ("cloud_white", "#F8FAFC", "neutral_light", 0.96),
    ("pale_slate", "#E2E8F0", "neutral_light", 0.90),
    ("ivory", "#FFFDF7", "cream", 0.96),
    ("warm_cream", "#FFF4D6", "cream", 0.93),
    ("vanilla", "#FFEFC2", "cream", 0.92),
    ("peach_cream", "#FCE7C8", "cream", 0.90),
    ("pure_black", "#000000", "neutral_dark", 0.96),
    ("ink_black", "#111111", "neutral_dark", 0.95),
    ("charcoal", "#1F2937", "neutral_dark", 0.93),
    ("navy_black", "#0F172A", "neutral_dark", 0.94),
    ("pastel_pink", "#FBCFE8", "pastel", 0.82),
    ("peach_pastel", "#FED7AA", "pastel", 0.81),
    ("mint_pastel", "#BBF7D0", "pastel", 0.80),
    ("sky_pastel", "#BAE6FD", "pastel", 0.83),
    ("lavender", "#DDD6FE", "pastel", 0.83),
    ("jade_mist", "#D1FAE5", "pastel", 0.82),
    ("vivid_red", "#EF4444", "accent", 0.88),
    ("vivid_orange", "#F97316", "accent", 0.87),
    ("amber", "#F59E0B", "accent", 0.84),
    ("green", "#22C55E", "accent", 0.84),
    ("teal", "#14B8A6", "accent", 0.86),
    ("sky", "#0EA5E9", "accent", 0.87),
    ("blue", "#3B82F6", "accent", 0.90),
    ("purple", "#A855F7", "accent", 0.87),
    ("deep_red", "#7F1D1D", "deep", 0.88),
    ("deep_green", "#14532D", "deep", 0.87),
    ("deep_blue", "#1E3A8A", "deep", 0.92),
    ("deep_violet", "#4C1D95", "deep", 0.90),
    ("slate", "#64748B", "muted", 0.78),
    ("stone", "#78716C", "muted", 0.76),
]


def hex_to_rgb(hex_color: str) -> list[int]:
    value = hex_color[1:] if hex_color.startswith("#") else hex_color
    return [int(value[index : index + 2], 16) for index in (0, 2, 4)]


def srgb_channel_to_linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: Iterable[int]) -> float:
    red, green, blue = [srgb_channel_to_linear(channel) for channel in rgb]
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def rgb_to_lab(rgb: Iterable[int]) -> list[float]:
    red, green, blue = [srgb_channel_to_linear(channel) for channel in rgb]

    x = (0.4124564 * red) + (0.3575761 * green) + (0.1804375 * blue)
    y = (0.2126729 * red) + (0.7151522 * green) + (0.0721750 * blue)
    z = (0.0193339 * red) + (0.1191920 * green) + (0.9503041 * blue)

    x /= 0.95047
    y /= 1.00000
    z /= 1.08883

    delta = 6 / 29

    def lab_f(value: float) -> float:
        if value > delta**3:
            return math.copysign(abs(value) ** (1 / 3), value)
        return (value / (3 * delta**2)) + (4 / 29)

    fx = lab_f(x)
    fy = lab_f(y)
    fz = lab_f(z)

    def round_lab(value: float) -> float:
        rounded = round(value, 2)
        return 0.0 if abs(rounded) == 0 else rounded

    return [
        round_lab((116 * fy) - 16),
        round_lab(500 * (fx - fy)),
        round_lab(200 * (fy - fz)),
    ]


def build_palette() -> list[dict[str, object]]:
    palette = []
    for item_id, (name, hex_color, group, aesthetic_prior) in enumerate(RAW_PALETTE):
        rgb = hex_to_rgb(hex_color)
        palette.append(
            {
                "id": item_id,
                "hex": hex_color.upper(),
                "name": name,
                "group": group,
                "rgb": rgb,
                "lab": rgb_to_lab(rgb),
                "relative_luminance": round(relative_luminance(rgb), 6),
                "aesthetic_prior": aesthetic_prior,
            }
        )
    return palette


def validate_palette(palette: list[dict[str, object]]) -> None:
    if len(palette) != PALETTE_SIZE:
        raise ValueError(f"Expected {PALETTE_SIZE} colors, got {len(palette)}")

    ids = [item["id"] for item in palette]
    if ids != list(range(PALETTE_SIZE)):
        raise ValueError(f"Palette ids must be continuous from 0 to {PALETTE_SIZE - 1}")

    required = {
        "id",
        "hex",
        "name",
        "group",
        "rgb",
        "lab",
        "relative_luminance",
        "aesthetic_prior",
    }
    hex_values = []
    names = []
    for item in palette:
        missing = required - set(item)
        if missing:
            raise ValueError(f"Missing fields for id={item.get('id')}: {sorted(missing)}")
        if item["group"] not in ALLOWED_GROUPS:
            raise ValueError(f"Invalid group for id={item['id']}: {item['group']}")
        if not 0 <= float(item["aesthetic_prior"]) <= 1:
            raise ValueError(f"Invalid aesthetic_prior for id={item['id']}")
        hex_values.append(item["hex"])
        names.append(item["name"])

    if len(set(hex_values)) != len(hex_values):
        raise ValueError("Duplicate HEX values found")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate color names found")

    group_counts = Counter(str(item["group"]) for item in palette)
    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        raise ValueError(f"Unexpected group counts: {dict(sorted(group_counts.items()))}")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_palette_preview(palette: list[dict[str, object]]) -> None:
    columns = 8
    tile_width = 172
    tile_height = 104
    gap = 8
    margin = 32
    header_height = 74
    rows = math.ceil(len(palette) / columns)

    width = (margin * 2) + (columns * tile_width) + ((columns - 1) * gap)
    height = (margin * 2) + header_height + (rows * tile_height) + ((rows - 1) * gap)

    image = Image.new("RGB", (width, height), (246, 247, 249))
    draw = ImageDraw.Draw(image)

    title_font = load_font(28)
    text_font = load_font(14)
    small_font = load_font(12)

    draw.text(
        (margin, 24),
        f"LighTrip Title Color Palette - {PALETTE_SIZE} Fixed Classes",
        fill=(24, 29, 39),
        font=title_font,
    )
    counts = Counter(str(item["group"]) for item in palette)
    summary = "  ".join(f"{group}: {counts[group]}" for group in sorted(ALLOWED_GROUPS))
    draw.text((margin, 57), summary, fill=(75, 85, 99), font=small_font)

    start_y = margin + header_height
    for index, item in enumerate(palette):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        y = start_y + row * (tile_height + gap)

        rgb = tuple(item["rgb"])
        luminance = float(item["relative_luminance"])
        text_color = (17, 24, 39) if luminance > 0.45 else (255, 255, 255)
        muted_text_color = (55, 65, 81) if luminance > 0.45 else (229, 231, 235)
        border_color = (210, 214, 220) if luminance > 0.70 else rgb

        draw.rounded_rectangle(
            (x, y, x + tile_width, y + tile_height),
            radius=6,
            fill=rgb,
            outline=border_color,
            width=1,
        )

        line_1 = f"{item['id']:02d} {item['name']}"
        line_2 = str(item["hex"])
        line_3 = f"{item['group']}  L={luminance:.3f}"
        draw.text((x + 10, y + 12), line_1, fill=text_color, font=text_font)
        draw.text((x + 10, y + 40), line_2, fill=text_color, font=text_font)
        draw.text((x + 10, y + 70), line_3, fill=muted_text_color, font=small_font)

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    image.save(PREVIEW_PATH)


def main() -> None:
    palette = build_palette()
    validate_palette(palette)

    PALETTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PALETTE_PATH.write_text(
        json.dumps(palette, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    draw_palette_preview(palette)

    counts = Counter(item["group"] for item in palette)
    print(f"Wrote {PALETTE_PATH}")
    print(f"Wrote {PREVIEW_PATH}")
    print(f"Group counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
