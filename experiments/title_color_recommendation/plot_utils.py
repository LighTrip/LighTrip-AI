from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_pyplot(project_root: Path) -> Any:
    matplotlib_config_dir = project_root / "outputs" / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def markdown_image_path(report_path: Path, image_path: Path) -> str:
    try:
        return image_path.relative_to(report_path.parent).as_posix()
    except ValueError:
        return image_path.as_posix()


def top_color_rows(distribution: list[float], *, limit: int = 8) -> list[str]:
    pairs = sorted(
        enumerate(distribution),
        key=lambda item: item[1],
        reverse=True,
    )
    rows = ["| rank | palette_id | share |", "| ---: | ---: | ---: |"]
    for rank, (palette_id, share) in enumerate(pairs[:limit], start=1):
        rows.append(f"| {rank} | {palette_id} | {share:.4f} |")
    return rows
