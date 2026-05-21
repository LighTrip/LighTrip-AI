from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch
else:
    torch = None

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is not installed in this environment.",
)

from src.title_color_recommendation.data.dataloader import (  # noqa: E402
    create_title_color_dataloader,
    create_title_color_dataloaders,
    validate_title_color_batch,
)
from src.title_color_recommendation.data.dataset import (  # noqa: E402
    TitleColorAugmentationConfig,
    TitleColorDataset,
)


MANIFEST_FIELDS = ["id", "split", "roi_path", "mask_path", "label_matrix_index"]


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _make_roi(path: Path) -> None:
    image = Image.new("RGB", (136, 36), (0, 0, 255))
    for x in range(68):
        for y in range(36):
            image.putpixel((x, y), (255, 0, 0))
    image.save(path)


def _make_mask(path: Path) -> None:
    mask = Image.new("L", (136, 36), 0)
    for x in range(68):
        for y in range(36):
            mask.putpixel((x, y), 255)
    mask.save(path)


def _make_dataset_files(tmp_path: Path) -> Path:
    data_root = tmp_path / "data" / "title_color_recommendation"
    splits_dir = data_root / "splits"
    roi_dir = data_root / "processed" / "rois"
    mask_dir = data_root / "processed" / "masks"
    labels_dir = data_root / "processed" / "labels"
    for path in (splits_dir, roi_dir, mask_dir, labels_dir):
        path.mkdir(parents=True, exist_ok=True)

    split_to_ids = {
        "train": ["train_0", "train_1"],
        "val": ["val_0"],
        "test": ["test_0"],
    }
    all_ids = [
        image_id
        for image_ids in split_to_ids.values()
        for image_id in image_ids
    ]

    for image_id in all_ids:
        _make_roi(roi_dir / f"{image_id}.png")
        _make_mask(mask_dir / f"{image_id}.png")

    matrix = []
    for index, _image_id in enumerate(all_ids):
        values = np.arange(1, 33, dtype=np.float32) + index
        matrix.append(values / values.sum())
    np.save(labels_dir / "labels_matrix.npy", np.stack(matrix))

    with (labels_dir / "labels_soft.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "split", "palette_id", "pseudo_score"],
        )
        writer.writeheader()
        for split, image_ids in split_to_ids.items():
            for image_id in image_ids:
                for palette_id in range(32):
                    writer.writerow(
                        {
                            "id": image_id,
                            "split": split,
                            "palette_id": str(palette_id),
                            "pseudo_score": f"{palette_id / 31:.6f}",
                        }
                    )

    matrix_index_by_id = {
        image_id: str(index)
        for index, image_id in enumerate(all_ids)
    }
    for split, image_ids in split_to_ids.items():
        rows = []
        for image_id in image_ids:
            rows.append(
                {
                    "id": image_id,
                    "split": split,
                    "roi_path": str((roi_dir / f"{image_id}.png").relative_to(tmp_path)),
                    "mask_path": str((mask_dir / f"{image_id}.png").relative_to(tmp_path)),
                    "label_matrix_index": matrix_index_by_id[image_id],
                }
            )
        _write_manifest(splits_dir / f"{split}.csv", rows)

    return data_root


def test_title_color_dataset_loads_four_channel_sample(tmp_path: Path) -> None:
    data_root = _make_dataset_files(tmp_path)
    dataset = TitleColorDataset(
        "train",
        data_root=data_root,
        project_root=tmp_path,
        augment=False,
        mmap_mode=None,
    )

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["image_id"] == "train_0"
    assert tuple(sample["x"].shape) == (4, 36, 136)
    assert tuple(sample["pseudo_scores"].shape) == (32,)
    assert tuple(sample["target_distribution"].shape) == (32,)
    assert torch.isclose(sample["target_distribution"].sum(), torch.tensor(1.0))
    assert set(sample["x"][3].unique().tolist()) <= {0.0, 1.0}


def test_title_color_dataloader_batches_have_expected_shapes(tmp_path: Path) -> None:
    data_root = _make_dataset_files(tmp_path)
    loader = create_title_color_dataloader(
        "train",
        batch_size=2,
        data_root=data_root,
        project_root=tmp_path,
        shuffle=False,
        num_workers=0,
        dataset_kwargs={"augment": False, "mmap_mode": None},
    )

    batch = next(iter(loader))

    validate_title_color_batch(batch)
    assert tuple(batch["x"].shape) == (2, 4, 36, 136)
    assert tuple(batch["pseudo_scores"].shape) == (2, 32)
    assert tuple(batch["target_distribution"].shape) == (2, 32)
    assert torch.allclose(batch["target_distribution"].sum(dim=1), torch.ones(2))
    assert batch["image_id"] == ["train_0", "train_1"]


def test_create_train_val_test_dataloaders_from_split_manifests(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset_files(tmp_path)
    loaders = create_title_color_dataloaders(
        batch_size=2,
        data_root=data_root,
        project_root=tmp_path,
        num_workers=0,
        dataset_kwargs={"mmap_mode": None},
    )

    assert set(loaders) == {"train", "val", "test"}
    assert len(loaders["train"].dataset) == 2
    assert len(loaders["val"].dataset) == 1
    assert len(loaders["test"].dataset) == 1
    assert loaders["train"].sampler.__class__.__name__ == "RandomSampler"
    assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"
    assert loaders["test"].sampler.__class__.__name__ == "SequentialSampler"


def test_train_only_augmentation_flips_roi_and_mask_together(
    tmp_path: Path,
) -> None:
    data_root = _make_dataset_files(tmp_path)
    augmentation = TitleColorAugmentationConfig(
        flip_p=1.0,
        brightness=0.0,
        contrast=0.0,
    )
    train_dataset = TitleColorDataset(
        "train",
        data_root=data_root,
        project_root=tmp_path,
        augmentation=augmentation,
        mmap_mode=None,
    )
    val_dataset = TitleColorDataset(
        "val",
        data_root=data_root,
        project_root=tmp_path,
        augmentation=augmentation,
        mmap_mode=None,
    )

    train_x = train_dataset[0]["x"]
    val_x = val_dataset[0]["x"]

    assert train_x[0, 0, 0] == 0.0
    assert train_x[2, 0, 0] == 1.0
    assert train_x[3, 0, 0] == 0.0
    assert val_x[0, 0, 0] == 1.0
    assert val_x[2, 0, 0] == 0.0
    assert val_x[3, 0, 0] == 1.0
