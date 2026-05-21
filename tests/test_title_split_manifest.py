from __future__ import annotations

import pytest

from src.title_color_recommendation.data.split_manifest import (
    SplitRatios,
    apply_split_to_rows,
    category_distribution,
    image_counts_by_split,
    split_group_counts,
    stratified_image_split,
)


def test_split_group_counts_keeps_eval_splits_for_small_buckets() -> None:
    assert split_group_counts(3, SplitRatios()) == {
        "train": 1,
        "val": 1,
        "test": 1,
    }
    assert split_group_counts(10, SplitRatios()) == {
        "train": 8,
        "val": 1,
        "test": 1,
    }


def test_stratified_image_split_preserves_category_counts() -> None:
    rows = [
        {"id": f"abstract_{index}", "category_slug": "abstract"}
        for index in range(10)
    ] + [
        {"id": f"food_{index}", "category_slug": "food"}
        for index in range(10)
    ]

    assignments = stratified_image_split(rows, seed=7, ratios=SplitRatios())

    assert image_counts_by_split(assignments) == {
        "train": 16,
        "val": 2,
        "test": 2,
    }
    assert category_distribution(rows, assignments) == {
        "abstract": {"total": 10, "train": 8, "val": 1, "test": 1},
        "food": {"total": 10, "train": 8, "val": 1, "test": 1},
    }


def test_apply_split_keeps_same_image_rows_together() -> None:
    rows = [
        {"id": "image_a", "category_slug": "fashion", "roi_path": "a_1.jpg"},
        {"id": "image_a", "category_slug": "fashion", "roi_path": "a_2.jpg"},
        {"id": "image_b", "category_slug": "fashion", "roi_path": "b_1.jpg"},
        {"id": "image_c", "category_slug": "fashion", "roi_path": "c_1.jpg"},
    ]

    assignments = stratified_image_split(rows, seed=3, ratios=SplitRatios())
    split_rows = apply_split_to_rows(rows, assignments, seed=3)
    image_a_splits = {
        row["split"]
        for rows_for_split in split_rows.values()
        for row in rows_for_split
        if row["id"] == "image_a"
    }

    assert len(image_a_splits) == 1


def test_stratified_image_split_rejects_conflicting_image_categories() -> None:
    rows = [
        {"id": "image_a", "category_slug": "fashion"},
        {"id": "image_a", "category_slug": "food"},
    ]

    with pytest.raises(ValueError, match="multiple categories"):
        stratified_image_split(rows)
