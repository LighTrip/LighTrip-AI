from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image, ImageDraw, ImageFont


class RandIntGenerator(Protocol):
    def randint(self, a: int, b: int) -> int:
        """Return a random integer N such that a <= N <= b."""


@dataclass(frozen=True)
class ImageSize:
    width: int
    height: int


@dataclass(frozen=True)
class RelativeROI:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class TitleSpec:
    center_x: float
    center_y: float
    font_size: int
    text: str = "TITLE"
    font_path: str = ""


@dataclass(frozen=True)
class TitleROIResult:
    cropped_image: Image.Image
    roi_image: Image.Image
    text_mask: Image.Image
    resized_size: ImageSize
    crop_box: tuple[int, int, int, int]
    roi_box: tuple[int, int, int, int]
    title_center_in_roi: tuple[float, float]


def image_size_from_config(config: Mapping[str, Any]) -> ImageSize:
    return ImageSize(
        width=int(config["width"]),
        height=int(config["height"]),
    )


def relative_roi_from_config(config: Mapping[str, Any]) -> RelativeROI:
    return RelativeROI(
        x1=float(config["x1"]),
        y1=float(config["y1"]),
        x2=float(config["x2"]),
        y2=float(config["y2"]),
    )


def title_spec_from_config(config: Mapping[str, Any]) -> TitleSpec:
    return TitleSpec(
        center_x=float(config["center_x"]),
        center_y=float(config["center_y"]),
        font_size=int(config["font_size"]),
        text=str(config.get("text") or "TITLE"),
        font_path=str(config.get("font_path") or ""),
    )


def validate_image_size(size: ImageSize, *, name: str) -> None:
    if size.width <= 0 or size.height <= 0:
        raise ValueError(f"{name} width/height must be positive: {size}")


def validate_relative_roi(roi: RelativeROI) -> None:
    if not 0 <= roi.x1 < roi.x2 <= 1:
        raise ValueError(f"ROI x coordinates must satisfy 0 <= x1 < x2 <= 1: {roi}")
    if not 0 <= roi.y1 < roi.y2 <= 1:
        raise ValueError(f"ROI y coordinates must satisfy 0 <= y1 < y2 <= 1: {roi}")


def resampling_lanczos() -> int:
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


def resampling_nearest() -> int:
    try:
        return Image.Resampling.NEAREST
    except AttributeError:
        return Image.NEAREST


def resized_size_for_crop(original_size: ImageSize, target_size: ImageSize) -> ImageSize:
    validate_image_size(original_size, name="original_size")
    validate_image_size(target_size, name="target_size")
    scale = max(
        target_size.width / original_size.width,
        target_size.height / original_size.height,
    )
    return ImageSize(
        width=max(target_size.width, int(math.ceil(original_size.width * scale))),
        height=max(target_size.height, int(math.ceil(original_size.height * scale))),
    )


def resize_keep_aspect_for_crop(
    image: Image.Image,
    target_size: ImageSize,
) -> Image.Image:
    resized_size = resized_size_for_crop(
        ImageSize(width=image.width, height=image.height),
        target_size,
    )
    if image.size == (resized_size.width, resized_size.height):
        return image.copy()
    return image.resize(
        (resized_size.width, resized_size.height),
        resample=resampling_lanczos(),
    )


def stable_int(seed: int, key: str, *, max_value: int) -> int:
    if max_value <= 0:
        return 0
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (max_value + 1)


def split_to_crop_mode(split: str) -> str:
    return "random" if split.strip().lower() == "train" else "center"


def crop_offsets(
    source_size: ImageSize,
    target_size: ImageSize,
    *,
    mode: str,
    rng: RandIntGenerator | None = None,
    seed: int | None = None,
    key: str = "",
) -> tuple[int, int]:
    validate_image_size(source_size, name="source_size")
    validate_image_size(target_size, name="target_size")
    if source_size.width < target_size.width or source_size.height < target_size.height:
        raise ValueError(
            "source_size must be greater than or equal to target_size: "
            f"source={source_size}, target={target_size}"
        )

    max_x = source_size.width - target_size.width
    max_y = source_size.height - target_size.height

    if mode == "center":
        return max_x // 2, max_y // 2
    if mode == "random":
        if seed is not None and key:
            return (
                stable_int(seed, f"{key}:x", max_value=max_x),
                stable_int(seed, f"{key}:y", max_value=max_y),
            )
        if rng is not None:
            return rng.randint(0, max_x), rng.randint(0, max_y)
        return secrets.randbelow(max_x + 1), secrets.randbelow(max_y + 1)
    raise ValueError(f"Unsupported crop mode: {mode}")


def crop_to_input_size(
    image: Image.Image,
    target_size: ImageSize,
    *,
    mode: str,
    rng: RandIntGenerator | None = None,
    seed: int | None = None,
    key: str = "",
) -> tuple[Image.Image, ImageSize, tuple[int, int, int, int]]:
    resized = resize_keep_aspect_for_crop(image, target_size)
    resized_size = ImageSize(width=resized.width, height=resized.height)
    left, top = crop_offsets(
        resized_size,
        target_size,
        mode=mode,
        rng=rng,
        seed=seed,
        key=key,
    )
    box = (left, top, left + target_size.width, top + target_size.height)
    return resized.crop(box), resized_size, box


def roi_box_from_relative(
    image_size: ImageSize,
    roi: RelativeROI,
) -> tuple[int, int, int, int]:
    validate_image_size(image_size, name="image_size")
    validate_relative_roi(roi)
    x1 = int(math.floor(image_size.width * roi.x1))
    y1 = int(math.floor(image_size.height * roi.y1))
    x2 = int(math.ceil(image_size.width * roi.x2))
    y2 = int(math.ceil(image_size.height * roi.y2))

    x1 = min(max(x1, 0), image_size.width - 1)
    y1 = min(max(y1, 0), image_size.height - 1)
    x2 = min(max(x2, x1 + 1), image_size.width)
    y2 = min(max(y2, y1 + 1), image_size.height)
    return x1, y1, x2, y2


def crop_roi(
    cropped_image: Image.Image,
    roi: RelativeROI,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    roi_box = roi_box_from_relative(
        ImageSize(width=cropped_image.width, height=cropped_image.height),
        roi,
    )
    return cropped_image.crop(roi_box), roi_box


def title_center_in_roi(
    crop_size: ImageSize,
    roi_box: tuple[int, int, int, int],
    title: TitleSpec,
) -> tuple[float, float]:
    x1, y1, _, _ = roi_box
    return (crop_size.width * title.center_x) - x1, (crop_size.height * title.center_y) - y1


@lru_cache(maxsize=16)
def load_font(
    size: int,
    font_path: str = "",
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if font_path:
        path = Path(font_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"font_path 파일을 찾을 수 없습니다: {font_path}")
        return ImageFont.truetype(str(path), size)

    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def generate_text_mask(
    roi_size: ImageSize,
    title_center: tuple[float, float],
    title: TitleSpec,
    *,
    text: str | None = None,
) -> Image.Image:
    validate_image_size(roi_size, name="roi_size")
    mask = Image.new("L", (roi_size.width, roi_size.height), 0)
    draw = ImageDraw.Draw(mask)
    font = load_font(title.font_size, title.font_path)
    title_text = text if text is not None else title.text
    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    left = title_center[0] - (text_width / 2) - bbox[0]
    top = title_center[1] - (text_height / 2) - bbox[1]
    draw.text((left, top), title_text, fill=255, font=font)
    return mask.point(lambda value: 255 if value else 0)


def prepare_title_roi(
    image: Image.Image,
    *,
    input_size: ImageSize,
    roi: RelativeROI,
    title: TitleSpec,
    crop_mode: str,
    rng: RandIntGenerator | None = None,
    seed: int | None = None,
    key: str = "",
    text: str | None = None,
) -> TitleROIResult:
    cropped_image, resized_size, crop_box = crop_to_input_size(
        image,
        input_size,
        mode=crop_mode,
        rng=rng,
        seed=seed,
        key=key,
    )
    roi_image, roi_box = crop_roi(cropped_image, roi)
    roi_size = ImageSize(width=roi_image.width, height=roi_image.height)
    center = title_center_in_roi(input_size, roi_box, title)
    text_mask = generate_text_mask(roi_size, center, title, text=text)
    return TitleROIResult(
        cropped_image=cropped_image,
        roi_image=roi_image,
        text_mask=text_mask,
        resized_size=resized_size,
        crop_box=crop_box,
        roi_box=roi_box,
        title_center_in_roi=center,
    )
