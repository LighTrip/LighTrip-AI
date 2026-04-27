from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset.common import load_categories


def parse_category_filter(value: str, categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not value.strip():
        return categories

    requested = {item.strip() for item in value.split(",") if item.strip()}
    selected = [
        category
        for category in categories
        if str(category["slug"]) in requested or str(category["label"]) in requested
    ]
    if len(selected) != len(requested):
        allowed = ", ".join(
            [str(category["slug"]) for category in categories]
            + [str(category["label"]) for category in categories]
        )
        raise ValueError(f"알 수 없는 카테고리가 있습니다. 사용 가능: {allowed}")

    return selected


def get_sample_source_label(sample: Any, classes: set[str]) -> str:
    labels = get_sample_labels(sample)
    for label_name in labels:
        if label_name in classes:
            return label_name

    return ""


def get_sample_labels(sample: Any) -> set[str]:
    sample_labels: set[str] = set()
    for field_name in ("detections", "classifications", "positive_labels", "ground_truth"):
        try:
            field = sample[field_name]
        except (KeyError, AttributeError):
            continue

        if field is None:
            continue

        labels = getattr(field, "detections", None) or getattr(field, "classifications", None)
        if labels is None and isinstance(field, list):
            labels = field if isinstance(field, list) else []
        if labels is None and hasattr(field, "label"):
            labels = [field]
        if labels is None:
            continue

        for label in labels:
            label_name = getattr(label, "label", "")
            if label_name:
                sample_labels.add(str(label_name))

    return sample_labels


def has_excluded_label(sample: Any, excluded_classes: set[str]) -> bool:
    if not excluded_classes:
        return False

    return bool(get_sample_labels(sample) & excluded_classes)


def main() -> None:
    parser = argparse.ArgumentParser(description="FiftyOne으로 Open Images V7 일부 이미지를 수집합니다.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-samples-per-category", type=int, default=180)
    parser.add_argument("--label-types", default="detections,classifications")
    parser.add_argument("--categories", default="")
    parser.add_argument("--no-balance-classes", action="store_true")
    parser.add_argument("--oversample-factor", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-output", action="store_true")
    args = parser.parse_args()

    try:
        import fiftyone.zoo as foz
    except ImportError as exc:
        raise RuntimeError(
            "fiftyone이 설치되어 있지 않습니다. `pip install fiftyone` 후 다시 실행하세요."
        ) from exc

    label_types = [item.strip() for item in args.label_types.split(",") if item.strip()]

    for category in parse_category_filter(args.categories, load_categories()):
        label = str(category["label"])
        slug = str(category["slug"])
        classes = [str(item) for item in category["open_images_classes"]]
        excluded_classes = {
            str(item) for item in category.get("exclude_open_images_classes", [])
        }
        target_dir = args.output_dir / slug
        if args.overwrite_output and target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        per_class_limit = math.ceil(args.max_samples_per_category / len(classes))
        class_groups = [classes] if args.no_balance_classes else [[class_name] for class_name in classes]

        for class_group in class_groups:
            if copied >= args.max_samples_per_category:
                break

            class_set = set(class_group)
            dataset_name_class = "-".join(
                class_name.lower().replace(" ", "-") for class_name in class_group
            )
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=args.split,
                dataset_name=f"open-images-v7-{args.split}-{slug}-{dataset_name_class}",
                label_types=label_types,
                classes=class_group,
                max_samples=min(
                    per_class_limit * args.oversample_factor,
                    (args.max_samples_per_category - copied) * args.oversample_factor,
                ),
                shuffle=True,
                seed=args.seed,
                only_matching=True,
            )

            for sample in dataset:
                if copied >= args.max_samples_per_category:
                    break

                source_path = Path(sample.filepath)
                if not source_path.exists():
                    continue
                if has_excluded_label(sample, excluded_classes):
                    continue

                source_label = get_sample_source_label(sample, class_set) or class_group[0]
                source_label_slug = source_label.lower().replace(" ", "_") or "unknown"
                copied += 1
                target_path = target_dir / f"{slug}_{copied:05d}_{source_label_slug}{source_path.suffix.lower()}"
                shutil.copy2(source_path, target_path)

        print(f"[{label}] {copied} images from {classes} -> {target_dir}")


if __name__ == "__main__":
    main()
