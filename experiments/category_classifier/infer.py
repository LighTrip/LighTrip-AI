from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.category_classifier.src.data import load_texts_from_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="저장된 카테고리 분류 모델로 추론합니다.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("experiments/category_classifier/artifacts/nb_tfidf.joblib"),
    )
    parser.add_argument("--text", action="append", help="분류할 텍스트입니다. 여러 번 입력할 수 있습니다.")
    parser.add_argument("--input-jsonl", type=Path, help="분류할 JSONL 파일입니다.")
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        help="predict_proba 최대값이 이 값보다 낮으면 기타로 반환합니다.",
    )
    return parser.parse_args()


def max_probabilities(model: Any, texts: list[str]) -> list[float | None]:
    if not hasattr(model, "predict_proba"):
        return [None for _ in texts]
    probabilities = model.predict_proba(texts)
    return [float(row.max()) for row in probabilities]


def apply_unknown_threshold(
    labels: list[str],
    probabilities: list[float | None],
    threshold: float | None,
) -> list[str]:
    if threshold is None:
        return labels
    return [
        "기타" if probability is not None and probability < threshold else label
        for label, probability in zip(labels, probabilities)
    ]


def predict_rows(
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
    text_field: str,
    unknown_threshold: float | None,
) -> list[dict[str, Any]]:
    pipeline = artifact["pipeline"]
    texts = [str(row[text_field]) for row in rows]
    raw_labels = pipeline.predict(texts).tolist()
    probabilities = max_probabilities(pipeline, texts)
    labels = apply_unknown_threshold(raw_labels, probabilities, unknown_threshold)

    results: list[dict[str, Any]] = []
    for row, raw_label, label, probability in zip(rows, raw_labels, labels, probabilities):
        result = dict(row)
        result["predicted_label"] = label
        result["raw_predicted_label"] = raw_label
        if probability is not None:
            result["confidence"] = probability
        results.append(result)
    return results


def main() -> None:
    args = parse_args()
    if not args.text and not args.input_jsonl:
        raise SystemExit("--text 또는 --input-jsonl 중 하나는 필요합니다.")

    artifact = joblib.load(args.model_path)

    rows: list[dict[str, Any]] = []
    if args.text:
        rows.extend({args.text_field: text} for text in args.text)
    if args.input_jsonl:
        rows.extend(load_texts_from_jsonl(args.input_jsonl, text_field=args.text_field))

    results = predict_rows(artifact, rows, args.text_field, args.unknown_threshold)

    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w", encoding="utf-8") as file:
            for result in results:
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
        return

    for result in results:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
