from __future__ import annotations

from PIL import Image

from src.title_color_recommendation.data.roi_preprocessing import (
    ImageSize,
    RelativeROI,
    TitleSpec,
    crop_offsets,
    prepare_title_roi,
    resized_size_for_crop,
    roi_box_from_relative,
    split_to_crop_mode,
)


def test_resized_size_preserves_aspect_and_covers_target() -> None:
    target = ImageSize(width=150, height=200)

    portrait = resized_size_for_crop(ImageSize(width=400, height=800), target)
    assert portrait == ImageSize(width=150, height=300)

    landscape = resized_size_for_crop(ImageSize(width=800, height=400), target)
    assert landscape == ImageSize(width=400, height=200)


def test_crop_offsets_center_and_random_are_valid() -> None:
    source = ImageSize(width=400, height=200)
    target = ImageSize(width=150, height=200)

    assert crop_offsets(source, target, mode="center") == (125, 0)

    left, top = crop_offsets(source, target, mode="random", seed=7, key="sample")
    assert 0 <= left <= 250
    assert top == 0


def test_roi_box_matches_config_example() -> None:
    roi = RelativeROI(x1=0.05, y1=0.18, x2=0.95, y2=0.36)

    assert roi_box_from_relative(ImageSize(width=150, height=200), roi) == (
        7,
        36,
        143,
        72,
    )


def test_prepare_title_roi_keeps_native_roi_size_and_binary_mask() -> None:
    image = Image.new("RGB", (400, 800), (64, 128, 192))
    result = prepare_title_roi(
        image,
        input_size=ImageSize(width=150, height=200),
        roi=RelativeROI(x1=0.05, y1=0.18, x2=0.95, y2=0.36),
        title=TitleSpec(center_x=0.5, center_y=0.264, font_size=22, text="TITLE"),
        crop_mode="center",
    )

    assert result.cropped_image.size == (150, 200)
    assert result.roi_image.size == (136, 36)
    assert result.text_mask.size == result.roi_image.size
    assert set(result.text_mask.getdata()) <= {0, 255}
    assert result.text_mask.getbbox() is not None


def test_split_to_crop_mode() -> None:
    assert split_to_crop_mode("train") == "random"
    assert split_to_crop_mode("valid") == "center"
    assert split_to_crop_mode("test") == "center"
    assert split_to_crop_mode("") == "center"
