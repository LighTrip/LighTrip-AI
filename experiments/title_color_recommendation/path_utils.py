from __future__ import annotations

from pathlib import Path


def is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def resolve_project_path(
    project_root: Path,
    value: str | Path,
    *,
    must_exist: bool = False,
    description: str = "path",
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve(strict=False)
    resolved_project_root = project_root.resolve()
    if not is_relative_to(resolved, resolved_project_root):
        raise ValueError(f"{description} must be inside project root: {value}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {value}")
    return resolved
