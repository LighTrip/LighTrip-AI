from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from experiments.category_classifier._bootstrap import bootstrap_project_root

bootstrap_project_root()

from experiments.category_classifier.src.data import load_text_label_dataset
from experiments.category_classifier.src.evaluate import (
    evaluate_predictions,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
)
from experiments.category_classifier.src.models import (
    add_model_hyperparameter_arguments,
    add_single_model_argument,
    build_pipeline_from_args,
    model_params_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF 기반 카테고리 분류 모델을 학습합니다."
    )
    add_single_model_argument(parser)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data_places365/processed/train.jsonl"),
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path("data_places365/processed/valid.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data_places365/processed/test.jsonl"),
    )
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=Path("experiments/category_classifier/stopwords_ko.txt"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("experiments/category_classifier/artifacts"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("experiments/category_classifier/reports"),
    )
    add_model_hyperparameter_arguments(parser)
    return parser.parse_args()


def split_metrics(
    model_name: str,
    split_name: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    report_dir: Path,
) -> dict[str, Any]:
    metrics = evaluate_predictions(y_true, y_pred, labels)
    save_classification_report(
        report_dir / f"{model_name}_{split_name}_classification_report.txt",
        y_true,
        y_pred,
        labels,
    )
    save_confusion_matrix(
        report_dir / f"{model_name}_{split_name}_confusion_matrix.csv",
        y_true,
        y_pred,
        labels,
    )
    return metrics


def main() -> None:
    args = parse_args()

    train_texts, train_labels = load_text_label_dataset(
        args.train,
        text_field=args.text_field,
        label_field=args.label_field,
        drop_empty_text=True,
    )
    valid_texts, valid_labels = load_text_label_dataset(
        args.valid,
        text_field=args.text_field,
        label_field=args.label_field,
    )
    test_texts, test_labels = load_text_label_dataset(
        args.test,
        text_field=args.text_field,
        label_field=args.label_field,
    )

    labels = sorted(set(train_labels) | set(valid_labels) | set(test_labels))
    model_params = model_params_from_args(args)
    pipeline = build_pipeline_from_args(args, args.model)
    pipeline.fit(train_texts, np.asarray(train_labels, dtype=object))
    confidence_type = "probability" if hasattr(pipeline, "predict_proba") else "decision_function"

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    valid_pred = pipeline.predict(valid_texts).tolist()
    test_pred = pipeline.predict(test_texts).tolist()
    metrics = {
        "model": args.model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "labels": labels,
        "dataset": {
            "train": str(args.train),
            "valid": str(args.valid),
            "test": str(args.test),
            "train_size": len(train_texts),
            "valid_size": len(valid_texts),
            "test_size": len(test_texts),
        },
        "params": model_params,
        "valid": split_metrics(
            args.model,
            "valid",
            valid_labels,
            valid_pred,
            labels,
            args.report_dir,
        ),
        "test": split_metrics(
            args.model,
            "test",
            test_labels,
            test_pred,
            labels,
            args.report_dir,
        ),
    }

    artifact_path = args.artifact_dir / f"{args.model}_tfidf.joblib"
    joblib.dump(
        {
            "pipeline": pipeline,
            "metadata": {
                "model": args.model,
                "labels": labels,
                "text_field": args.text_field,
                "label_field": args.label_field,
                "params": model_params,
                "confidence_type": confidence_type,
            },
        },
        artifact_path,
    )

    metrics_path = args.report_dir / f"{args.model}_metrics.json"
    save_metrics(metrics_path, metrics)

    summary = {
        "artifact": str(artifact_path),
        "metrics": str(metrics_path),
        "valid_accuracy": metrics["valid"]["accuracy"],
        "test_accuracy": metrics["test"]["accuracy"],
        "test_f1_macro": metrics["test"]["f1_macro"],
        "test_f1_weighted": metrics["test"]["f1_weighted"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
