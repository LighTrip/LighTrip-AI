from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch.
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]

from src.title_color_recommendation.data.dataset import (
    DEFAULT_DATA_ROOT,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_CLASSES,
    SPLIT_NAMES,
    TitleColorAugmentationConfig,
    TitleColorDataset,
    load_label_matrix,
    load_pseudo_scores,
    manifest_items_from_rows,
    normalize_split,
    read_manifest_rows,
    require_torch,
    resolve_path,
)


def require_dataloader() -> Any:
    if torch is None or DataLoader is None:
        raise ModuleNotFoundError(
            "PyTorch is required for title color DataLoader creation. "
            "Install torch before constructing dataloaders."
        )
    return DataLoader


def create_title_color_dataset(
    split: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    project_root: str | Path = ".",
    augmentation: TitleColorAugmentationConfig | None = None,
    **dataset_kwargs: Any,
) -> TitleColorDataset:
    if augmentation is not None:
        dataset_kwargs.setdefault("augmentation", augmentation)
    return TitleColorDataset(
        split=split,
        data_root=data_root,
        project_root=project_root,
        **dataset_kwargs,
    )


def create_title_color_dataloader(
    split: str,
    *,
    batch_size: int = 32,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    project_root: str | Path = ".",
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int | None = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    loader_cls = require_dataloader()
    split = normalize_split(split)
    dataset = TitleColorDataset(
        split=split,
        data_root=data_root,
        project_root=project_root,
        **dict(dataset_kwargs or {}),
    )

    generator = None
    if seed is not None:
        torch_module = require_torch()
        generator = torch_module.Generator()
        generator.manual_seed(seed)

    if shuffle is None:
        shuffle = split == "train"

    return loader_cls(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
    )


def create_title_color_datasets(
    *,
    splits: Sequence[str] = SPLIT_NAMES,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    project_root: str | Path = ".",
    labels_matrix_path: str | Path | None = None,
    labels_soft_path: str | Path | None = None,
    mmap_mode: str | None = "r",
    augmentation: TitleColorAugmentationConfig | None = None,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> dict[str, TitleColorDataset]:
    project_root_path = Path(project_root).expanduser().resolve()
    data_root_path = resolve_path(data_root, project_root_path)
    normalized_splits = tuple(normalize_split(split) for split in splits)

    labels_matrix_path = (
        resolve_path(labels_matrix_path, project_root_path)
        if labels_matrix_path is not None
        else data_root_path / "processed" / "labels" / "labels_matrix.npy"
    )
    labels_soft_path = (
        resolve_path(labels_soft_path, project_root_path)
        if labels_soft_path is not None
        else data_root_path / "processed" / "labels" / "labels_soft.csv"
    )
    labels_matrix = load_label_matrix(labels_matrix_path, mmap_mode=mmap_mode)
    num_classes = int(labels_matrix.shape[1])

    rows_by_split = {
        split: read_manifest_rows(data_root_path / "splits" / f"{split}.csv")
        for split in normalized_splits
    }
    all_image_ids = [
        item.image_id
        for split, rows in rows_by_split.items()
        for item in manifest_items_from_rows(
            rows,
            split=split,
            project_root=project_root_path,
        )
    ]
    pseudo_scores_by_id = load_pseudo_scores(
        labels_soft_path,
        all_image_ids,
        num_classes=num_classes,
    )

    return {
        split: TitleColorDataset(
            split=split,
            data_root=data_root_path,
            project_root=project_root_path,
            rows=rows,
            labels_matrix=labels_matrix,
            pseudo_scores_by_id=pseudo_scores_by_id,
            image_size=image_size,
            augmentation=augmentation,
        )
        for split, rows in rows_by_split.items()
    }


def create_title_color_dataloaders(
    *,
    batch_size: int = 32,
    splits: Sequence[str] = SPLIT_NAMES,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    project_root: str | Path = ".",
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int | None = None,
    dataset_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    loader_cls = require_dataloader()
    datasets = create_title_color_datasets(
        splits=splits,
        data_root=data_root,
        project_root=project_root,
        **dict(dataset_kwargs or {}),
    )

    generator = None
    if seed is not None:
        torch_module = require_torch()
        generator = torch_module.Generator()
        generator.manual_seed(seed)

    return {
        split: loader_cls(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            generator=generator,
        )
        for split, dataset in datasets.items()
    }


def validate_title_color_batch(
    batch: Mapping[str, Any],
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> None:
    width, height = image_size
    x = batch["x"]
    pseudo_scores = batch["pseudo_scores"]
    target_distribution = batch["target_distribution"]

    if tuple(x.shape[1:]) != (4, height, width):
        raise ValueError(f"x batch shape must be [B, 4, {height}, {width}]: {x.shape}")
    if tuple(pseudo_scores.shape[1:]) != (num_classes,):
        raise ValueError(
            f"pseudo_scores batch shape must be [B, {num_classes}]: "
            f"{pseudo_scores.shape}"
        )
    if tuple(target_distribution.shape[1:]) != (num_classes,):
        raise ValueError(
            f"target_distribution batch shape must be [B, {num_classes}]: "
            f"{target_distribution.shape}"
        )

    sums = target_distribution.sum(dim=1)
    torch_module = require_torch()
    if not torch_module.allclose(
        sums,
        torch_module.ones_like(sums),
        atol=1e-4,
    ):
        raise ValueError("target_distribution rows must sum to 1.0")
