from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch.
    torch = None  # type: ignore[assignment]

    class TorchDataset:  # type: ignore[no-redef]
        pass


DEFAULT_DATA_ROOT = Path("data/title_color_recommendation")
DEFAULT_IMAGE_SIZE = (136, 36)
DEFAULT_NUM_CLASSES = 32
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class TitleColorAugmentationConfig:
    flip_p: float = 0.5
    brightness: float = 0.08
    contrast: float = 0.08


@dataclass(frozen=True)
class TitleColorManifestItem:
    image_id: str
    roi_path: Path
    mask_path: Path
    label_matrix_index: int


def require_torch() -> Any:
    if torch is None:
        raise ModuleNotFoundError(
            "PyTorch is required for TitleColorDataset. "
            "Install torch before constructing datasets or dataloaders."
        )
    return torch


def resolve_path(path: str | Path, base_dir: str | Path = ".") -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(base_dir).expanduser().resolve() / path


def manifest_path_for_split(
    split: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    project_root: str | Path = ".",
) -> Path:
    split = normalize_split(split)
    data_root_path = resolve_path(data_root, project_root)
    return data_root_path / "splits" / f"{split}.csv"


def normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    if normalized not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}: {split!r}")
    return normalized


def read_manifest_rows(manifest_path: str | Path) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"split manifest not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"split manifest is empty: {path}")
    return rows


def load_label_matrix(
    labels_matrix_path: str | Path,
    *,
    mmap_mode: str | None = "r",
) -> np.ndarray:
    path = Path(labels_matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"labels_matrix.npy not found: {path}")

    matrix = np.load(path, mmap_mode=mmap_mode)
    if matrix.ndim != 2:
        raise ValueError(f"labels matrix must be 2D: shape={matrix.shape}")
    if matrix.shape[1] <= 0:
        raise ValueError(f"labels matrix must have at least one class: {matrix.shape}")
    return matrix


def load_pseudo_scores(
    labels_soft_path: str | Path,
    image_ids: Iterable[str],
    *,
    num_classes: int = DEFAULT_NUM_CLASSES,
    score_column: str = "pseudo_score",
) -> dict[str, np.ndarray]:
    requested_ids = {str(image_id) for image_id in image_ids}
    if not requested_ids:
        return {}

    path = Path(labels_soft_path)
    if not path.exists():
        raise FileNotFoundError(f"labels_soft.csv not found: {path}")

    scores_by_id = {
        image_id: np.zeros(num_classes, dtype=np.float32)
        for image_id in requested_ids
    }
    seen_palette_ids: dict[str, set[int]] = {image_id: set() for image_id in requested_ids}
    completed_ids: set[str] = set()

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"id", "palette_id", score_column}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"labels_soft.csv missing columns: {sorted(missing_columns)}"
            )

        for row in reader:
            image_id = str(row["id"])
            if image_id not in requested_ids:
                continue

            palette_id = int(row["palette_id"])
            if not 0 <= palette_id < num_classes:
                raise ValueError(
                    f"palette_id out of range for {image_id}: "
                    f"{palette_id} not in [0, {num_classes})"
                )

            scores_by_id[image_id][palette_id] = np.float32(row[score_column])
            seen_palette_ids[image_id].add(palette_id)
            if len(seen_palette_ids[image_id]) == num_classes:
                completed_ids.add(image_id)
                if len(completed_ids) == len(requested_ids):
                    break

    incomplete_ids = [
        image_id
        for image_id, palette_ids in seen_palette_ids.items()
        if len(palette_ids) != num_classes
    ]
    if incomplete_ids:
        sample = sorted(incomplete_ids)[:5]
        raise ValueError(
            "labels_soft.csv does not contain complete pseudo scores for "
            f"{len(incomplete_ids)} images; sample={sample}"
        )

    return scores_by_id


def _required_value(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required in split manifest rows")
    return value


def _label_matrix_index(row: Mapping[str, Any]) -> int:
    for key in ("label_matrix_index", "matrix_index"):
        raw_value = str(row.get(key) or "").strip()
        if raw_value:
            return int(raw_value)
    raise ValueError("label_matrix_index is required in split manifest rows")


def manifest_items_from_rows(
    rows: list[Mapping[str, Any]],
    *,
    split: str,
    project_root: str | Path = ".",
) -> list[TitleColorManifestItem]:
    split = normalize_split(split)
    seen_ids: set[str] = set()
    items: list[TitleColorManifestItem] = []

    for row in rows:
        image_id = _required_value(row, "id")
        row_split = str(row.get("split") or "").strip().lower()
        if row_split and row_split != split:
            raise ValueError(
                f"manifest row split mismatch for {image_id}: "
                f"expected={split}, actual={row_split}"
            )
        if image_id in seen_ids:
            raise ValueError(f"duplicate image_id in split manifest: {image_id}")
        seen_ids.add(image_id)

        items.append(
            TitleColorManifestItem(
                image_id=image_id,
                roi_path=resolve_path(_required_value(row, "roi_path"), project_root),
                mask_path=resolve_path(_required_value(row, "mask_path"), project_root),
                label_matrix_index=_label_matrix_index(row),
            )
        )

    return items


class TitleColorDataset(TorchDataset):
    """Load native-size title ROI RGB, text mask, and 32-way soft labels."""

    def __init__(
        self,
        split: str,
        *,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        project_root: str | Path = ".",
        manifest_path: str | Path | None = None,
        labels_matrix_path: str | Path | None = None,
        labels_soft_path: str | Path | None = None,
        rows: list[Mapping[str, Any]] | None = None,
        labels_matrix: np.ndarray | None = None,
        pseudo_scores_by_id: Mapping[str, np.ndarray] | None = None,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        augment: bool | None = None,
        augmentation: TitleColorAugmentationConfig | None = None,
        mmap_mode: str | None = "r",
    ) -> None:
        super().__init__()
        require_torch()

        self.split = normalize_split(split)
        self.project_root = Path(project_root).expanduser().resolve()
        self.data_root = resolve_path(data_root, self.project_root)
        self.image_size = image_size
        self.augment = self.split == "train" if augment is None else bool(augment)
        self.augmentation = augmentation or TitleColorAugmentationConfig()

        manifest_path = (
            resolve_path(manifest_path, self.project_root)
            if manifest_path is not None
            else self.data_root / "splits" / f"{self.split}.csv"
        )
        manifest_rows = rows if rows is not None else read_manifest_rows(manifest_path)
        self.items = manifest_items_from_rows(
            list(manifest_rows),
            split=self.split,
            project_root=self.project_root,
        )

        labels_matrix_path = (
            resolve_path(labels_matrix_path, self.project_root)
            if labels_matrix_path is not None
            else self.data_root / "processed" / "labels" / "labels_matrix.npy"
        )
        self.labels_matrix = (
            labels_matrix
            if labels_matrix is not None
            else load_label_matrix(labels_matrix_path, mmap_mode=mmap_mode)
        )
        self.num_classes = int(self.labels_matrix.shape[1])

        labels_soft_path = (
            resolve_path(labels_soft_path, self.project_root)
            if labels_soft_path is not None
            else self.data_root / "processed" / "labels" / "labels_soft.csv"
        )
        self.pseudo_scores_by_id = (
            dict(pseudo_scores_by_id)
            if pseudo_scores_by_id is not None
            else load_pseudo_scores(
                labels_soft_path,
                (item.image_id for item in self.items),
                num_classes=self.num_classes,
            )
        )
        self._validate_label_assets()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch_module = require_torch()
        item = self.items[index]

        roi = self._load_image(item.roi_path, mode="RGB", image_id=item.image_id)
        mask = self._load_image(item.mask_path, mode="L", image_id=item.image_id)
        self._validate_native_size(roi, mask, image_id=item.image_id)

        if self.augment:
            roi, mask = self._augment_pair(roi, mask)

        width, height = roi.size
        rgb_tensor = (
            torch_module.tensor(list(roi.getdata()), dtype=torch_module.float32)
            .view(height, width, 3)
            .permute(2, 0, 1)
            .contiguous()
            / 255.0
        )
        mask_tensor = (
            torch_module.tensor(list(mask.getdata()), dtype=torch_module.float32)
            .view(1, height, width)
            .gt(127.5)
            .float()
        )
        x = torch_module.cat((rgb_tensor, mask_tensor), dim=0).float()

        target_distribution = np.asarray(
            self.labels_matrix[item.label_matrix_index],
            dtype=np.float32,
        )
        pseudo_scores = np.asarray(
            self.pseudo_scores_by_id[item.image_id],
            dtype=np.float32,
        )

        return {
            "x": x,
            "pseudo_scores": torch_module.tensor(
                pseudo_scores.tolist(),
                dtype=torch_module.float32,
            ),
            "target_distribution": torch_module.tensor(
                target_distribution.tolist(),
                dtype=torch_module.float32,
            ),
            "image_id": item.image_id,
        }

    def _validate_label_assets(self) -> None:
        matrix_rows = int(self.labels_matrix.shape[0])
        missing_pseudo_scores: list[str] = []

        for item in self.items:
            if not 0 <= item.label_matrix_index < matrix_rows:
                raise IndexError(
                    f"label_matrix_index out of range for {item.image_id}: "
                    f"{item.label_matrix_index} not in [0, {matrix_rows})"
                )

            pseudo_scores = self.pseudo_scores_by_id.get(item.image_id)
            if pseudo_scores is None:
                missing_pseudo_scores.append(item.image_id)
                continue
            if len(pseudo_scores) != self.num_classes:
                raise ValueError(
                    f"pseudo_scores for {item.image_id} must have "
                    f"{self.num_classes} values: shape={np.shape(pseudo_scores)}"
                )

        if missing_pseudo_scores:
            raise ValueError(
                "missing pseudo_scores for image ids: "
                f"{missing_pseudo_scores[:5]}"
            )

    def _load_image(self, path: Path, *, mode: str, image_id: str) -> Image.Image:
        if not path.exists():
            raise FileNotFoundError(f"{mode} image for {image_id} not found: {path}")
        with Image.open(path) as image:
            return image.convert(mode)

    def _validate_native_size(
        self,
        roi: Image.Image,
        mask: Image.Image,
        *,
        image_id: str,
    ) -> None:
        if roi.size != self.image_size:
            raise ValueError(
                f"ROI image for {image_id} must keep native size "
                f"{self.image_size}: actual={roi.size}"
            )
        if mask.size != self.image_size:
            raise ValueError(
                f"text mask for {image_id} must keep native size "
                f"{self.image_size}: actual={mask.size}"
            )

    def _augment_pair(
        self,
        roi: Image.Image,
        mask: Image.Image,
    ) -> tuple[Image.Image, Image.Image]:
        torch_module = require_torch()
        config = self.augmentation
        if float(torch_module.rand(()).item()) < config.flip_p:
            roi = ImageOps.mirror(roi)
            mask = ImageOps.mirror(mask)

        if config.brightness > 0:
            factor = self._jitter_factor(config.brightness)
            roi = ImageEnhance.Brightness(roi).enhance(factor)
        if config.contrast > 0:
            factor = self._jitter_factor(config.contrast)
            roi = ImageEnhance.Contrast(roi).enhance(factor)

        return roi, mask

    def _jitter_factor(self, strength: float) -> float:
        torch_module = require_torch()
        low = 1.0 - strength
        high = 1.0 + strength
        return float(torch_module.empty(()).uniform_(low, high).item())
