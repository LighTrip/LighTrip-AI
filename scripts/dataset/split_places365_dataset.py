from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import read_jsonl, stratified_split, write_split_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="검증된 JSONL을 train/valid/test로 계층 분할합니다.")
    parser.add_argument("--input", type=Path, default=Path("data_places365/interim/places365_generated_drafts.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_places365/processed"))
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

    write_split_outputs(
        output_dir=args.output_dir,
        train=train,
        valid=valid,
        test=test,
        include_csv=args.csv,
    )

    print(f"train: {len(train)}")
    print(f"valid: {len(valid)}")
    print(f"test: {len(test)}")


if __name__ == "__main__":
    main()
