from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def validate(self) -> None:
        ratios = self.as_dict()
        for split, ratio in ratios.items():
            if ratio < 0:
                raise ValueError(f"{split} ratio must be non-negative: {ratio}")
        total = sum(ratios.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"split ratios must sum to 1.0: {total}")
        if self.train <= 0:
            raise ValueError("train ratio must be positive")


@dataclass
class ImageGroup:
    image_id: str
    category: str
    rows: list[Mapping[str, Any]]


def split_hash(seed: int, category: str, image_id: str) -> str:
    payload = f"{seed}:{category}:{image_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_group_counts(total: int, ratios: SplitRatios) -> dict[str, int]:
    """Return train/val/test counts for one stratification bucket."""
    ratios.validate()
    if total < 0:
        raise ValueError(f"total must be non-negative: {total}")
    if total == 0:
        return {split: 0 for split in SPLIT_NAMES}

    ratio_map = ratios.as_dict()
    raw_counts = {split: total * ratio_map[split] for split in SPLIT_NAMES}
    counts = {split: int(math.floor(raw_counts[split])) for split in SPLIT_NAMES}

    active_eval_splits = [
        split for split in ("val", "test") if ratio_map[split] > 0
    ]
    minimums = {split: 0 for split in SPLIT_NAMES}
    if total >= len(active_eval_splits) + 1:
        for split in active_eval_splits:
            minimums[split] = 1
            counts[split] = max(counts[split], 1)

    while sum(counts.values()) > total:
        if counts["train"] > minimums["train"]:
            counts["train"] -= 1
            continue
        candidates = [
            split for split in SPLIT_NAMES if counts[split] > minimums[split]
        ]
        if not candidates:
            break
        split_to_reduce = max(candidates, key=lambda split: counts[split])
        counts[split_to_reduce] -= 1

    remainder_order = sorted(
        SPLIT_NAMES,
        key=lambda split: (
            -(raw_counts[split] - math.floor(raw_counts[split])),
            SPLIT_NAMES.index(split),
        ),
    )
    while sum(counts.values()) < total:
        for split in remainder_order:
            counts[split] += 1
            if sum(counts.values()) == total:
                break

    return counts


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required for split manifest rows")
    return value


def group_rows_by_image(
    rows: list[Mapping[str, Any]],
    *,
    image_key: str = "id",
    category_key: str = "category_slug",
) -> dict[str, ImageGroup]:
    groups: dict[str, ImageGroup] = {}
    for row in rows:
        image_id = _required_text(row, image_key)
        category = _required_text(row, category_key)
        group = groups.get(image_id)
        if group is None:
            groups[image_id] = ImageGroup(
                image_id=image_id,
                category=category,
                rows=[row],
            )
            continue
        if group.category != category:
            raise ValueError(
                f"image {image_id} has multiple categories: "
                f"{group.category}, {category}"
            )
        group.rows.append(row)
    return groups


def stratified_image_split(
    rows: list[Mapping[str, Any]],
    *,
    seed: int = 42,
    ratios: SplitRatios | None = None,
    image_key: str = "id",
    category_key: str = "category_slug",
) -> dict[str, str]:
    """Assign one split per image id while preserving category proportions."""
    split_ratios = ratios or SplitRatios()
    split_ratios.validate()

    groups = group_rows_by_image(
        rows,
        image_key=image_key,
        category_key=category_key,
    )
    category_to_ids: dict[str, list[str]] = defaultdict(list)
    for group in groups.values():
        category_to_ids[group.category].append(group.image_id)

    assignments: dict[str, str] = {}
    for category in sorted(category_to_ids):
        image_ids = category_to_ids[category]
        image_ids.sort(key=lambda image_id: split_hash(seed, category, image_id))
        counts = split_group_counts(len(image_ids), split_ratios)

        cursor = 0
        for split in SPLIT_NAMES:
            split_count = counts[split]
            for image_id in image_ids[cursor : cursor + split_count]:
                assignments[image_id] = split
            cursor += split_count

    return assignments


def apply_split_to_rows(
    rows: list[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    seed: int = 42,
    image_key: str = "id",
    category_key: str = "category_slug",
) -> dict[str, list[dict[str, Any]]]:
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLIT_NAMES}
    for row in rows:
        image_id = _required_text(row, image_key)
        split = assignments.get(image_id)
        if split not in SPLIT_NAMES:
            raise ValueError(f"missing split assignment for image: {image_id}")
        row_with_split = dict(row)
        row_with_split["split"] = split
        split_rows[split].append(row_with_split)

    for split in SPLIT_NAMES:
        split_rows[split].sort(
            key=lambda row: split_hash(
                seed,
                str(row.get(category_key) or ""),
                str(row.get(image_key) or ""),
            )
        )
    return split_rows


def image_counts_by_split(assignments: Mapping[str, str]) -> dict[str, int]:
    counter = Counter(assignments.values())
    return {split: counter.get(split, 0) for split in SPLIT_NAMES}


def category_distribution(
    rows: list[Mapping[str, Any]],
    assignments: Mapping[str, str],
    *,
    image_key: str = "id",
    category_key: str = "category_slug",
) -> dict[str, dict[str, int]]:
    groups = group_rows_by_image(
        rows,
        image_key=image_key,
        category_key=category_key,
    )
    distribution: dict[str, dict[str, int]] = {}
    for image_id, group in groups.items():
        split = assignments.get(image_id)
        if split not in SPLIT_NAMES:
            raise ValueError(f"missing split assignment for image: {image_id}")
        category_counts = distribution.setdefault(
            group.category,
            {"total": 0, "train": 0, "val": 0, "test": 0},
        )
        category_counts["total"] += 1
        category_counts[split] += 1
    return dict(sorted(distribution.items()))
