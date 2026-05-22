from __future__ import annotations

from typing import Any

import pytest


class TinyTitleColorDataset:
    def __init__(self, torch_module: Any, *, length: int) -> None:
        self.torch = torch_module
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        x = self.torch.full((4, 4, 4), float(index + 1) / 10.0)
        target = self.torch.zeros(32, dtype=self.torch.float32)
        target[index % 32] = 0.7
        target[(index + 1) % 32] = 0.3
        pseudo_scores = self.torch.linspace(0.0, 1.0, steps=32)
        wcag_pass = self.torch.zeros(32, dtype=self.torch.float32)
        wcag_pass[::2] = 1.0
        return {
            "x": x,
            "target_distribution": target,
            "pseudo_scores": pseudo_scores,
            "wcag_pass": wcag_pass,
            "image_id": f"sample_{index}",
        }


def tiny_classifier(nn_module: Any) -> Any:
    return nn_module.Sequential(
        nn_module.Flatten(),
        nn_module.Linear(4 * 4 * 4, 32),
    )


def tiny_loader(
    torch_module: Any,
    *,
    length: int,
    batch_size: int = 2,
) -> Any:
    data_module = pytest.importorskip("torch.utils.data")
    dataset = TinyTitleColorDataset(torch_module, length=length)
    return data_module.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def tiny_split_loaders(torch_module: Any) -> dict[str, Any]:
    return {
        "train": tiny_loader(torch_module, length=4),
        "val": tiny_loader(torch_module, length=2),
        "test": tiny_loader(torch_module, length=2),
    }
