from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.category_classifier.src.data import load_text_label_dataset
from experiments.category_classifier.src.evaluate import (
    evaluate_predictions,
    save_classification_report,
    save_confusion_matrix,
    save_metrics,
)
from experiments.category_classifier.src.models import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TF-IDF + Naive Bayes 카테고리 분류 baseline을 학습합니다."
    )
    parser.add_argument("--model", default="nb", choices=["nb"])
    parser.add_argument("--train", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("data/processed/valid.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--stopwords", type=Path, default=Path("experiments/category_classifier/stopwords_ko.txt"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("experiments/category_classifier/artifacts"))
    parser.add_argument("--report-dir", type=Path, default=Path("experiments/category_classifier/reports"))
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    return parser.parse_args()


def split_metrics(
    split_name: str,
    y_true: list[str],
    y_pred: list[str],
    labels: list[str],
    report_dir: Path,
) -> dict[str, Any]:
    metrics = evaluate_predictions(y_true, y_pred, labels)
    save_classification_report(
        report_dir / f"{split_name}_classification_report.txt",
        y_true,
        y_pred,
        labels,
    )
    save_confusion_matrix(
        report_dir / f"{split_name}_confusion_matrix.csv",
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
    pipeline = build_pipeline(
        args.model,
        stopwords_path=args.stopwords,
        max_features=args.max_features,
        min_df=args.min_df,
        max_df=args.max_df,
        ngram_max=args.ngram_max,
        alpha=args.alpha,
    )
    pipeline.fit(train_texts, train_labels)

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
        "params": {
            "max_features": args.max_features,
            "min_df": args.min_df,
            "max_df": args.max_df,
            "ngram_max": args.ngram_max,
            "alpha": args.alpha,
        },
        "valid": split_metrics("valid", valid_labels, valid_pred, labels, args.report_dir),
        "test": split_metrics("test", test_labels, test_pred, labels, args.report_dir),
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
                "params": metrics["params"],
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
