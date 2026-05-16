from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import accuracy_score, f1_score

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from experiments.category_classifier._bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.category_classifier.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="confidence threshold를 튜닝해 low-confidence 샘플을 기타로 fallback합니다."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(
            "experiments/category_classifier/artifacts/"
            "places365_2_manual_full_calibrated/calibrated_linear_svm_tfidf.joblib"
        ),
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path("data/category_classifier/places365_v2/processed/valid.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/category_classifier/places365_v2/processed/test.jsonl"),
    )
    parser.add_argument(
        "--excluded",
        type=Path,
        default=Path("data/category_classifier/places365_v2/manual_review_full/excluded_drafts.jsonl"),
        help="모호/부적합 샘플 fallback rate 확인용 JSONL입니다.",
    )
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--label-field", default="label")
    parser.add_argument("--unknown-label", default="기타")
    parser.add_argument("--threshold-min", type=float, default=0.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="threshold 선택 시 validation에서 유지할 최소 non-fallback 비율입니다.",
    )
    parser.add_argument(
        "--lowest-confidence-limit",
        type=int,
        default=100,
        help="split별 최저 confidence 분석 파일에 저장할 최대 샘플 수입니다.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("experiments/category_classifier/reports/unknown_threshold"),
    )
    return parser.parse_args()


def require_probability_model(pipeline: Any) -> None:
    if not hasattr(pipeline, "predict_proba"):
        raise ValueError(
            "threshold tuning에는 predict_proba를 지원하는 모델이 필요합니다. "
            "calibrated_linear_svm artifact를 사용하세요."
        )
    if not hasattr(pipeline, "classes_"):
        raise ValueError("모델에서 classes_를 찾을 수 없습니다.")


def load_rows(
    path: Path,
    *,
    text_field: str,
    label_field: str,
    require_label: bool,
    drop_empty_text: bool,
) -> tuple[list[dict[str, Any]], list[str | None], int]:
    rows: list[dict[str, Any]] = []
    labels: list[str | None] = []
    skipped_empty_text = 0

    for index, row in enumerate(read_jsonl(path), start=1):
        text = row.get(text_field)
        if not isinstance(text, str) or not text.strip():
            if drop_empty_text:
                skipped_empty_text += 1
                continue
            raise ValueError(f"{path}:{index} '{text_field}' 값이 비어 있습니다.")

        label = row.get(label_field)
        if require_label and (not isinstance(label, str) or not label.strip()):
            raise ValueError(f"{path}:{index} '{label_field}' 값이 비어 있습니다.")

        rows.append(row)
        labels.append(label.strip() if isinstance(label, str) and label.strip() else None)

    if not rows:
        raise ValueError(f"분석 가능한 row가 없습니다: {path}")
    return rows, labels, skipped_empty_text


def probability_details(
    classes: list[str],
    probabilities: Any,
) -> tuple[str, float, str | None, float | None, float | None, dict[str, float]]:
    probability_values = [float(value) for value in probabilities]
    ranked_indexes = sorted(
        range(len(probability_values)),
        key=lambda index: probability_values[index],
        reverse=True,
    )
    top1_index = ranked_indexes[0]
    top2_index = ranked_indexes[1] if len(ranked_indexes) > 1 else None
    top1_probability = probability_values[top1_index]
    top2_probability = probability_values[top2_index] if top2_index is not None else None
    margin = (
        top1_probability - top2_probability
        if top2_probability is not None
        else None
    )
    probability_map = {
        label: probability
        for label, probability in zip(classes, probability_values)
    }
    return (
        classes[top1_index],
        top1_probability,
        classes[top2_index] if top2_index is not None else None,
        top2_probability,
        margin,
        probability_map,
    )


def predict_records(
    pipeline: Any,
    rows: list[dict[str, Any]],
    labels: list[str | None],
    *,
    text_field: str,
) -> list[dict[str, Any]]:
    texts = [str(row[text_field]) for row in rows]
    raw_predictions = [str(label) for label in pipeline.predict(texts).tolist()]
    probabilities = pipeline.predict_proba(texts)
    classes = [str(label) for label in pipeline.classes_]

    records: list[dict[str, Any]] = []
    for row, label, raw_prediction, probability_row in zip(
        rows,
        labels,
        raw_predictions,
        probabilities,
    ):
        (
            probability_label,
            confidence,
            second_label,
            second_confidence,
            confidence_margin,
            probability_map,
        ) = probability_details(classes, probability_row)

        record = dict(row)
        if label is not None:
            record["label"] = label
            record["is_correct"] = raw_prediction == label
        record["raw_predicted_label"] = raw_prediction
        record["probability_argmax_label"] = probability_label
        record["confidence"] = confidence
        record["second_label"] = second_label
        record["second_confidence"] = second_confidence
        record["confidence_margin"] = confidence_margin
        record["probabilities"] = probability_map
        records.append(record)

    return records


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def thresholded_label(record: dict[str, Any], threshold: float, unknown_label: str) -> str:
    if float(record["confidence"]) < threshold:
        return unknown_label
    return str(record["raw_predicted_label"])


def label_fallback_rates(
    records: list[dict[str, Any]],
    fallback_mask: list[bool],
) -> dict[str, dict[str, float | int]]:
    label_counts = Counter(str(record["label"]) for record in records)
    fallback_counts = Counter(
        str(record["label"])
        for record, is_fallback in zip(records, fallback_mask)
        if is_fallback
    )
    return {
        label: {
            "total": count,
            "fallback_count": fallback_counts[label],
            "fallback_rate": fallback_counts[label] / count,
        }
        for label, count in sorted(label_counts.items())
    }


def evaluate_accepted_threshold(
    records: list[dict[str, Any]],
    *,
    target_labels: list[str],
    threshold: float,
    unknown_label: str,
) -> dict[str, Any]:
    y_true = [str(record["label"]) for record in records]
    raw_pred = [str(record["raw_predicted_label"]) for record in records]
    fallback_mask = [
        float(record["confidence"]) < threshold
        for record in records
    ]
    y_pred = [
        unknown_label if is_fallback else prediction
        for prediction, is_fallback in zip(raw_pred, fallback_mask)
    ]
    raw_correct_mask = [
        truth == prediction
        for truth, prediction in zip(y_true, raw_pred)
    ]
    known_indexes = [
        index
        for index, is_fallback in enumerate(fallback_mask)
        if not is_fallback
    ]
    known_true = [y_true[index] for index in known_indexes]
    known_pred = [raw_pred[index] for index in known_indexes]

    total = len(records)
    fallback_count = sum(fallback_mask)
    known_count = total - fallback_count
    raw_correct_count = sum(raw_correct_mask)
    raw_wrong_count = total - raw_correct_count
    correct_to_unknown_count = sum(
        is_fallback and is_correct
        for is_fallback, is_correct in zip(fallback_mask, raw_correct_mask)
    )
    wrong_to_unknown_count = sum(
        is_fallback and not is_correct
        for is_fallback, is_correct in zip(fallback_mask, raw_correct_mask)
    )
    known_correct_count = raw_correct_count - correct_to_unknown_count
    known_error_count = known_count - known_correct_count

    return {
        "threshold": threshold,
        "total": total,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / total,
        "coverage": known_count / total,
        "known_count": known_count,
        "raw_accuracy": accuracy_score(y_true, raw_pred),
        "accuracy_with_unknown_as_wrong": accuracy_score(y_true, y_pred),
        "macro_f1_known_labels": f1_score(
            y_true,
            y_pred,
            labels=target_labels,
            average="macro",
            zero_division=0,
        ),
        "known_accuracy": safe_rate(known_correct_count, known_count),
        "known_macro_f1": (
            f1_score(
                known_true,
                known_pred,
                labels=target_labels,
                average="macro",
                zero_division=0,
            )
            if known_count > 0
            else None
        ),
        "known_error_count": known_error_count,
        "known_error_rate": safe_rate(known_error_count, known_count),
        "raw_wrong_count": raw_wrong_count,
        "wrong_to_unknown_count": wrong_to_unknown_count,
        "wrong_capture_rate": safe_rate(wrong_to_unknown_count, raw_wrong_count),
        "correct_to_unknown_count": correct_to_unknown_count,
        "correct_reject_rate": safe_rate(correct_to_unknown_count, raw_correct_count),
        "fallback_precision_against_raw_errors": safe_rate(
            wrong_to_unknown_count,
            fallback_count,
        ),
        "label_fallback_rates": label_fallback_rates(records, fallback_mask),
    }


def threshold_candidates(minimum: float, maximum: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("threshold-step은 0보다 커야 합니다.")
    if minimum < 0 or maximum > 1 or minimum > maximum:
        raise ValueError("threshold 범위는 0.0 이상 1.0 이하이며 min <= max 이어야 합니다.")

    values: list[float] = []
    current = minimum
    while current <= maximum + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def score_for_selection(metric: dict[str, Any]) -> tuple[float, float, float, float]:
    known_accuracy = metric["known_accuracy"]
    wrong_capture_rate = metric["wrong_capture_rate"]
    correct_reject_rate = metric["correct_reject_rate"]
    return (
        known_accuracy if known_accuracy is not None else -1.0,
        wrong_capture_rate if wrong_capture_rate is not None else -1.0,
        -(correct_reject_rate if correct_reject_rate is not None else 1.0),
        metric["threshold"],
    )


def select_threshold(
    threshold_metrics: list[dict[str, Any]],
    *,
    min_coverage: float,
) -> dict[str, Any]:
    candidates = [
        metric
        for metric in threshold_metrics
        if metric["coverage"] >= min_coverage and metric["known_count"] > 0
    ]
    if not candidates:
        candidates = [
            metric
            for metric in threshold_metrics
            if metric["known_count"] > 0
        ]
    return max(candidates, key=score_for_selection)


def evaluate_reject_set(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    unknown_label: str,
) -> dict[str, Any]:
    fallback_mask = [
        thresholded_label(record, threshold, unknown_label) == unknown_label
        for record in records
    ]
    fallback_count = sum(fallback_mask)
    label_counts = Counter(str(record.get("label", "")) for record in records)
    fallback_by_label = Counter(
        str(record.get("label", ""))
        for record, is_fallback in zip(records, fallback_mask)
        if is_fallback
    )

    return {
        "total": len(records),
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / len(records),
        "label_counts": dict(sorted(label_counts.items())),
        "fallback_by_label": dict(sorted(fallback_by_label.items())),
    }


def apply_selected_threshold(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    unknown_label: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["predicted_label"] = thresholded_label(record, threshold, unknown_label)
        item["used_unknown_fallback"] = item["predicted_label"] == unknown_label
        output.append(item)
    return output


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_threshold_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "threshold",
        "fallback_rate",
        "coverage",
        "raw_accuracy",
        "accuracy_with_unknown_as_wrong",
        "macro_f1_known_labels",
        "known_accuracy",
        "known_error_rate",
        "wrong_capture_rate",
        "correct_reject_rate",
        "fallback_precision_against_raw_errors",
        "fallback_count",
        "known_count",
        "wrong_to_unknown_count",
        "correct_to_unknown_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_low_confidence_outputs(
    report_dir: Path,
    split_name: str,
    records: list[dict[str, Any]],
    *,
    threshold: float,
    unknown_label: str,
    limit: int,
) -> None:
    thresholded_records = apply_selected_threshold(
        records,
        threshold=threshold,
        unknown_label=unknown_label,
    )
    sorted_records = sorted(
        thresholded_records,
        key=lambda record: float(record["confidence"]),
    )
    fallback_candidates = [
        record
        for record in sorted_records
        if record["used_unknown_fallback"]
    ]

    write_jsonl(
        report_dir / f"{split_name}_fallback_candidates.jsonl",
        fallback_candidates,
    )
    write_jsonl(
        report_dir / f"{split_name}_lowest_confidence.jsonl",
        sorted_records[:limit],
    )


def main() -> None:
    args = parse_args()
    artifact = joblib.load(args.model_path)
    pipeline = artifact["pipeline"] if isinstance(artifact, dict) else artifact
    metadata = artifact.get("metadata", {}) if isinstance(artifact, dict) else {}
    require_probability_model(pipeline)

    target_labels = [
        str(label)
        for label in metadata.get("labels", list(pipeline.classes_))
        if str(label) != args.unknown_label
    ]

    valid_rows, valid_labels, valid_skipped = load_rows(
        args.valid,
        text_field=args.text_field,
        label_field=args.label_field,
        require_label=True,
        drop_empty_text=False,
    )
    test_rows, test_labels, test_skipped = load_rows(
        args.test,
        text_field=args.text_field,
        label_field=args.label_field,
        require_label=True,
        drop_empty_text=False,
    )
    valid_records = predict_records(
        pipeline,
        valid_rows,
        valid_labels,
        text_field=args.text_field,
    )
    test_records = predict_records(
        pipeline,
        test_rows,
        test_labels,
        text_field=args.text_field,
    )

    thresholds = threshold_candidates(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    valid_threshold_metrics = [
        evaluate_accepted_threshold(
            valid_records,
            target_labels=target_labels,
            threshold=threshold,
            unknown_label=args.unknown_label,
        )
        for threshold in thresholds
    ]
    selected_valid_metric = select_threshold(
        valid_threshold_metrics,
        min_coverage=args.min_coverage,
    )
    selected_threshold = float(selected_valid_metric["threshold"])

    test_metric = evaluate_accepted_threshold(
        test_records,
        target_labels=target_labels,
        threshold=selected_threshold,
        unknown_label=args.unknown_label,
    )

    excluded_summary: dict[str, Any] | None = None
    excluded_skipped = 0
    if args.excluded:
        excluded_rows, excluded_labels, excluded_skipped = load_rows(
            args.excluded,
            text_field=args.text_field,
            label_field=args.label_field,
            require_label=False,
            drop_empty_text=True,
        )
        excluded_records = predict_records(
            pipeline,
            excluded_rows,
            excluded_labels,
            text_field=args.text_field,
        )
        excluded_summary = evaluate_reject_set(
            excluded_records,
            threshold=selected_threshold,
            unknown_label=args.unknown_label,
        )
        write_low_confidence_outputs(
            args.report_dir,
            "excluded",
            excluded_records,
            threshold=selected_threshold,
            unknown_label=args.unknown_label,
            limit=args.lowest_confidence_limit,
        )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    write_threshold_csv(
        args.report_dir / "valid_threshold_summary.csv",
        valid_threshold_metrics,
    )
    write_low_confidence_outputs(
        args.report_dir,
        "valid",
        valid_records,
        threshold=selected_threshold,
        unknown_label=args.unknown_label,
        limit=args.lowest_confidence_limit,
    )
    write_low_confidence_outputs(
        args.report_dir,
        "test",
        test_records,
        threshold=selected_threshold,
        unknown_label=args.unknown_label,
        limit=args.lowest_confidence_limit,
    )

    selected_threshold_payload = {
        "unknown_threshold": selected_threshold,
        "unknown_label": args.unknown_label,
        "selected_on": "valid",
        "min_coverage": args.min_coverage,
        "selection_metric_order": [
            "known_accuracy",
            "wrong_capture_rate",
            "lower_correct_reject_rate",
            "higher_threshold",
        ],
    }
    write_json(args.report_dir / "selected_threshold.json", selected_threshold_payload)

    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "model_metadata": metadata,
        "data": {
            "valid": str(args.valid),
            "test": str(args.test),
            "excluded": str(args.excluded) if args.excluded else None,
            "valid_size": len(valid_records),
            "test_size": len(test_records),
            "valid_skipped_empty_text": valid_skipped,
            "test_skipped_empty_text": test_skipped,
            "excluded_skipped_empty_text": excluded_skipped,
        },
        "threshold_grid": {
            "min": args.threshold_min,
            "max": args.threshold_max,
            "step": args.threshold_step,
        },
        "selected_threshold": selected_threshold_payload,
        "valid_selected": selected_valid_metric,
        "test_at_selected_threshold": test_metric,
        "excluded_at_selected_threshold": excluded_summary,
        "valid_threshold_metrics": valid_threshold_metrics,
        "outputs": {
            "valid_threshold_summary": str(args.report_dir / "valid_threshold_summary.csv"),
            "selected_threshold": str(args.report_dir / "selected_threshold.json"),
            "valid_fallback_candidates": str(args.report_dir / "valid_fallback_candidates.jsonl"),
            "test_fallback_candidates": str(args.report_dir / "test_fallback_candidates.jsonl"),
            "excluded_fallback_candidates": str(
                args.report_dir / "excluded_fallback_candidates.jsonl"
            )
            if args.excluded
            else None,
        },
    }
    write_json(args.report_dir / "threshold_tuning_metrics.json", metrics)

    print(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "selected_threshold": selected_threshold,
                "valid": {
                    "coverage": selected_valid_metric["coverage"],
                    "known_accuracy": selected_valid_metric["known_accuracy"],
                    "fallback_rate": selected_valid_metric["fallback_rate"],
                    "wrong_capture_rate": selected_valid_metric["wrong_capture_rate"],
                },
                "test": {
                    "coverage": test_metric["coverage"],
                    "known_accuracy": test_metric["known_accuracy"],
                    "fallback_rate": test_metric["fallback_rate"],
                    "wrong_capture_rate": test_metric["wrong_capture_rate"],
                },
                "excluded": excluded_summary,
                "metrics": str(args.report_dir / "threshold_tuning_metrics.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
