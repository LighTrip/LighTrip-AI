from __future__ import annotations

import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""} and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import dhash, remove_tree_inside_root, sha256_file


TITLE_DATA_ROOT = PROJECT_ROOT / "data" / "title_color_recommendation"
TITLE_CONFIG_ROOT = PROJECT_ROOT / "configs" / "title_color_recommendation"
TITLE_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "title_color_recommendation"
TMP_ROOT = Path("/tmp")

ALLOWED_READ_ROOTS = (
    PROJECT_ROOT.resolve(),
    TMP_ROOT.resolve(),
)
ALLOWED_WRITE_ROOTS = (
    TITLE_DATA_ROOT.resolve(),
    TITLE_OUTPUT_ROOT.resolve(),
    TMP_ROOT.resolve(),
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def project_relative(path: Path) -> str:
    resolved_project_root = PROJECT_ROOT.resolve()
    try:
        return path.resolve(strict=False).relative_to(resolved_project_root).as_posix()
    except ValueError:
        return path.as_posix()


def is_relative_to(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def first_matching_root(path: Path, roots: Sequence[Path]) -> Path | None:
    for root in roots:
        resolved_root = root.resolve(strict=False)
        if is_relative_to(path, resolved_root):
            return resolved_root
    return None


def require_within_roots(
    path: Path,
    roots: Sequence[Path],
    *,
    description: str,
) -> Path:
    resolved = path.resolve(strict=False)
    if first_matching_root(resolved, roots) is None:
        allowed = ", ".join(project_relative(root) for root in roots)
        raise ValueError(f"{description} 허용 경로 밖입니다: {path} (허용: {allowed})")
    return resolved


def resolve_project_path(
    value: str | Path,
    *,
    allowed_roots: Sequence[Path] = ALLOWED_READ_ROOTS,
    description: str = "path",
    must_exist: bool = False,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    resolved = require_within_roots(path, allowed_roots, description=description)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{description} 파일을 찾을 수 없습니다: {path}")
    return resolved


def resolve_config_path(value: str | Path) -> Path:
    return resolve_project_path(
        value,
        allowed_roots=(TITLE_CONFIG_ROOT, PROJECT_ROOT / "configs", TMP_ROOT),
        description="config path",
        must_exist=True,
    )


def resolve_output_path(value: str | Path, *, description: str) -> Path:
    return resolve_project_path(
        value,
        allowed_roots=ALLOWED_WRITE_ROOTS,
        description=description,
    )


def ensure_output_dir(path: Path) -> Path:
    resolved = require_within_roots(
        path,
        ALLOWED_WRITE_ROOTS,
        description="output directory",
    )
    if path.exists() and path.is_symlink():
        raise ValueError(f"심볼릭 링크 출력 경로는 허용하지 않습니다: {path}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def clear_output_dir(path: Path) -> Path:
    if path.exists() and path.is_symlink():
        raise ValueError(f"심볼릭 링크 폴더는 삭제하지 않습니다: {path}")

    resolved = require_within_roots(
        path,
        ALLOWED_WRITE_ROOTS,
        description="clear-output directory",
    )
    if resolved.exists():
        root = first_matching_root(resolved, ALLOWED_WRITE_ROOTS)
        if root is None or resolved == root:
            raise ValueError(f"삭제 대상이 허용된 출력 루트와 같습니다: {path}")
        remove_tree_inside_root(resolved, root)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def safe_path_segment(value: object, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in text
    ).strip("._")
    return (safe or fallback)[:128]


def safe_child_path(root: Path, *segments: object) -> Path:
    resolved_root = root.resolve(strict=False)
    path = resolved_root.joinpath(
        *(safe_path_segment(segment) for segment in segments)
    )
    return require_within_roots(path, (resolved_root,), description="output file")


def resolve_input_image_path(value: str, *, raw_dir: Path) -> Path:
    if not value.strip():
        raise ValueError("image_path가 비어 있습니다.")

    raw_root = raw_dir.resolve(strict=False)
    supplied = Path(value).expanduser()
    candidates = (
        [supplied]
        if supplied.is_absolute()
        else [
            PROJECT_ROOT / supplied,
            raw_root / supplied,
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if is_relative_to(resolved, raw_root):
            return resolved

    raise ValueError(f"raw_dir 밖의 image_path입니다: {value}")


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def write_csv_rows(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    require_within_roots(path, ALLOWED_WRITE_ROOTS, description="metadata output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    require_within_roots(path, ALLOWED_WRITE_ROOTS, description="summary output")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
