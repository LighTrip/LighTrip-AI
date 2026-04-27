from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import read_jsonl, write_jsonl


def stratified_split(
    rows: list[dict[str, Any]],
    seed: int,
    valid_ratio: float,
    test_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label"])].append(row)

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []

    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        total = len(label_rows)
        valid_count = max(1, int(total * valid_ratio)) if total >= 3 else 0
        test_count = max(1, int(total * test_ratio)) if total >= 3 else 0

        valid.extend(label_rows[:valid_count])
        test.extend(label_rows[valid_count : valid_count + test_count])
        train.extend(label_rows[valid_count + test_count :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "image", "generated_text", "label"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="검증된 JSONL을 train/valid/test로 계층 분할합니다.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/accepted_drafts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    train, valid, test = stratified_split(
        rows,
        seed=args.seed,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "valid.jsonl", valid)
    write_jsonl(args.output_dir / "test.jsonl", test)

    if args.csv:
        write_csv(args.output_dir / "train.csv", train)
        write_csv(args.output_dir / "valid.csv", valid)
        write_csv(args.output_dir / "test.csv", test)

    print(f"train: {len(train)}")
    print(f"valid: {len(valid)}")
    print(f"test: {len(test)}")


if __name__ == "__main__":
    main()
