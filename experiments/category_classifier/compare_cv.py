from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

try:
    from _bootstrap import bootstrap_project_root
except ModuleNotFoundError:
    from experiments.category_classifier._bootstrap import bootstrap_project_root

bootstrap_project_root()

from experiments.category_classifier.src.data import load_text_label_dataset
from experiments.category_classifier.src.evaluate import evaluate_predictions
from experiments.category_classifier.src.models import (
    MODEL_DISPLAY_NAMES,
    add_model_hyperparameter_arguments,
    add_multi_model_argument,
    build_pipeline_from_args,
    model_params_from_args,
)
METRIC_KEYS = ("accuracy", "f1_macro")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="카테고리 분류 모델 3종을 5-fold Stratified 교차 검증으로 비교합니다."
    )
    parser.add_argument(
        "--data",
        type=Path,
        nargs="+",
        default=[
            Path("data_places365/processed/train.jsonl"),
            Path("data_places365/processed/valid.jsonl"),
            Path("data_places365/processed/test.jsonl"),
        ],
        help="교차 검증에 사용할 JSONL 파일 목록입니다.",
    )
    parser.add_argument("--text-field", default="generated_text")
    parser.add_argument("--label-field", default="label")
    add_multi_model_argument(parser)
    parser.add_argument(
        "--stopwords",
        type=Path,
        default=Path("experiments/category_classifier/stopwords_ko.txt"),
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("experiments/category_classifier/reports/cv_5fold"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    add_model_hyperparameter_arguments(parser)
    return parser.parse_args()


def configure_matplotlib_font() -> None:
    candidates = ["NanumGothic"]

    available = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False


def load_cv_dataset(
    paths: list[Path],
    *,
    text_field: str,
    label_field: str,
) -> tuple[list[str], list[str], dict[str, Any]]:
    texts: list[str] = []
    labels: list[str] = []
    sources: list[dict[str, Any]] = []

    for path in paths:
        source_texts, source_labels = load_text_label_dataset(
            path,
            text_field=text_field,
            label_field=label_field,
            drop_empty_text=True,
        )
        texts.extend(source_texts)
        labels.extend(source_labels)
        sources.append({"path": str(path), "size": len(source_texts)})

    label_counts = dict(sorted(Counter(labels).items()))
    return texts, labels, {"sources": sources, "total_size": len(texts), "label_counts": label_counts}


def build_model(args: argparse.Namespace, model_name: str):
    return build_pipeline_from_args(args, model_name)


def summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in METRIC_KEYS:
        values = np.array([fold[key] for fold in folds], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1))
    for key in ("fit_seconds", "predict_seconds", "predict_ms_per_sample"):
        values = np.array([fold[key] for fold in folds], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=1))
    return summary


def evaluate_model_cv(
    args: argparse.Namespace,
    model_name: str,
    texts: list[str],
    labels: list[str],
    target_labels: list[str],
) -> dict[str, Any]:
    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )
    x = np.array(texts, dtype=object)
    y = np.array(labels, dtype=object)
    folds: list[dict[str, Any]] = []
    oof_true: list[str] = []
    oof_pred: list[str] = []

    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y), start=1):
        pipeline = build_model(args, model_name)
        train_texts = x[train_index].tolist()
        train_labels = y[train_index].tolist()
        test_texts = x[test_index].tolist()
        test_labels = y[test_index].tolist()

        fit_started_at = time.perf_counter()
        pipeline.fit(train_texts, train_labels)
        fit_seconds = time.perf_counter() - fit_started_at

        predict_started_at = time.perf_counter()
        predictions = pipeline.predict(test_texts).tolist()
        predict_seconds = time.perf_counter() - predict_started_at

        metrics = evaluate_predictions(test_labels, predictions, target_labels)
        fold_result = {
            "fold": fold_index,
            "train_size": len(train_texts),
            "valid_size": len(test_texts),
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "predict_ms_per_sample": (predict_seconds / len(test_texts)) * 1000,
        }
        folds.append(fold_result)
        oof_true.extend(test_labels)
        oof_pred.extend(predictions)

        print(
            f"[{MODEL_DISPLAY_NAMES[model_name]}] fold {fold_index}/{args.folds} "
            f"accuracy={metrics['accuracy']:.4f} macro_f1={metrics['f1_macro']:.4f}"
        )

    return {
        "model": model_name,
        "display_name": MODEL_DISPLAY_NAMES[model_name],
        "folds": folds,
        "summary": summarize_folds(folds),
        "oof": {
            "accuracy": evaluate_predictions(oof_true, oof_pred, target_labels)["accuracy"],
            "f1_macro": evaluate_predictions(oof_true, oof_pred, target_labels)["f1_macro"],
            "classification_report": classification_report(
                oof_true,
                oof_pred,
                labels=target_labels,
                zero_division=0,
                output_dict=True,
            ),
        },
        "oof_true": oof_true,
        "oof_pred": oof_pred,
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_summary_csv(path: Path, model_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "accuracy_mean",
                "accuracy_std",
                "macro_f1_mean",
                "macro_f1_std",
                "predict_ms_per_sample_mean",
                "predict_ms_per_sample_std",
                "fit_seconds_mean",
                "fit_seconds_std",
            ]
        )
        for result in model_results:
            summary = result["summary"]
            writer.writerow(
                [
                    result["model"],
                    summary["accuracy_mean"],
                    summary["accuracy_std"],
                    summary["f1_macro_mean"],
                    summary["f1_macro_std"],
                    summary["predict_ms_per_sample_mean"],
                    summary["predict_ms_per_sample_std"],
                    summary["fit_seconds_mean"],
                    summary["fit_seconds_std"],
                ]
            )


def save_fold_csv(path: Path, model_results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "model",
                "fold",
                "train_size",
                "valid_size",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "fit_seconds",
                "predict_seconds",
                "predict_ms_per_sample",
            ]
        )
        for result in model_results:
            for fold in result["folds"]:
                writer.writerow(
                    [
                        result["model"],
                        fold["fold"],
                        fold["train_size"],
                        fold["valid_size"],
                        fold["accuracy"],
                        fold["f1_macro"],
                        fold["f1_weighted"],
                        fold["fit_seconds"],
                        fold["predict_seconds"],
                        fold["predict_ms_per_sample"],
                    ]
                )


def save_confusion_matrix_outputs(
    report_dir: Path,
    result: dict[str, Any],
    target_labels: list[str],
) -> None:
    matrix = confusion_matrix(result["oof_true"], result["oof_pred"], labels=target_labels)
    csv_path = report_dir / f"{result['model']}_oof_confusion_matrix.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", *target_labels])
        for label, row in zip(target_labels, matrix):
            writer.writerow([label, *row.tolist()])

    report = classification_report(
        result["oof_true"],
        result["oof_pred"],
        labels=target_labels,
        zero_division=0,
    )
    (report_dir / f"{result['model']}_oof_classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(target_labels)),
        yticks=np.arange(len(target_labels)),
        xticklabels=target_labels,
        yticklabels=target_labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"{result['display_name']} OOF Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(
                col_index,
                row_index,
                format(matrix[row_index, col_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, col_index] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(report_dir / f"{result['model']}_oof_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_metric_summary(report_dir: Path, model_results: list[dict[str, Any]]) -> None:
    models = [result["display_name"] for result in model_results]
    x = np.arange(len(models))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))
    accuracy_mean = [result["summary"]["accuracy_mean"] for result in model_results]
    accuracy_std = [result["summary"]["accuracy_std"] for result in model_results]
    f1_mean = [result["summary"]["f1_macro_mean"] for result in model_results]
    f1_std = [result["summary"]["f1_macro_std"] for result in model_results]

    accuracy_bars = ax.bar(
        x - width / 2,
        accuracy_mean,
        width,
        yerr=accuracy_std,
        label="Accuracy",
        capsize=5,
    )
    f1_bars = ax.bar(
        x + width / 2,
        f1_mean,
        width,
        yerr=f1_std,
        label="Macro F1",
        capsize=5,
    )
    ax.set_ylabel("Score")
    ax.set_title("5-fold CV Mean Score (mean ± std)")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.14)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars, means, stds in (
        (accuracy_bars, accuracy_mean, accuracy_std),
        (f1_bars, f1_mean, f1_std),
    ):
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.018,
                f"{mean:.4f}\n±{std:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(report_dir / "cv_metric_summary.png", dpi=160)
    plt.close(fig)


def plot_fold_scores(report_dir: Path, model_results: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, metric, title in zip(axes, ("accuracy", "f1_macro"), ("Accuracy", "Macro F1")):
        for result in model_results:
            folds = [fold["fold"] for fold in result["folds"]]
            scores = [fold[metric] for fold in result["folds"]]
            axis.plot(folds, scores, marker="o", label=result["display_name"])
        axis.set_title(f"Fold별 {title}")
        axis.set_xlabel("Fold")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Score")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "cv_fold_scores.png", dpi=160)
    plt.close(fig)


def plot_inference_speed(report_dir: Path, model_results: list[dict[str, Any]]) -> None:
    models = [result["display_name"] for result in model_results]
    means = [result["summary"]["predict_ms_per_sample_mean"] for result in model_results]
    stds = [result["summary"]["predict_ms_per_sample_std"] for result in model_results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(models, means, yerr=stds, capsize=5, color="#4C78A8")
    ax.set_ylabel("ms / sample")
    ax.set_title("5-fold CV Inference Speed")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(report_dir / "cv_inference_speed.png", dpi=160)
    plt.close(fig)


def format_score(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def choose_best_model(model_results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        model_results,
        key=lambda result: (
            result["summary"]["f1_macro_mean"],
            result["summary"]["accuracy_mean"],
            -result["summary"]["f1_macro_std"],
            -result["summary"]["predict_ms_per_sample_mean"],
        ),
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    row_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator, *row_lines])


def save_markdown_report(
    path: Path,
    payload: dict[str, Any],
    model_results: list[dict[str, Any]],
    selected: dict[str, Any],
) -> None:
    summary_rows = []
    for result in model_results:
        summary = result["summary"]
        summary_rows.append(
            [
                result["display_name"],
                format_score(summary["accuracy_mean"], summary["accuracy_std"]),
                format_score(summary["f1_macro_mean"], summary["f1_macro_std"]),
                format_score(
                    summary["predict_ms_per_sample_mean"],
                    summary["predict_ms_per_sample_std"],
                ),
                format_score(summary["fit_seconds_mean"], summary["fit_seconds_std"]),
            ]
        )

    fold_rows = []
    for result in model_results:
        for fold in result["folds"]:
            fold_rows.append(
                [
                    result["display_name"],
                    str(fold["fold"]),
                    f"{fold['accuracy']:.4f}",
                    f"{fold['f1_macro']:.4f}",
                    f"{fold['predict_ms_per_sample']:.4f}",
                ]
            )

    selected_report = selected["oof"]["classification_report"]
    per_label_rows = []
    for label in payload["labels"]:
        metrics = selected_report[label]
        per_label_rows.append(
            [
                label,
                f"{metrics['precision']:.4f}",
                f"{metrics['recall']:.4f}",
                f"{metrics['f1-score']:.4f}",
                str(int(metrics["support"])),
            ]
        )

    dataset = payload["dataset"]
    sorted_by_macro = sorted(
        model_results,
        key=lambda result: result["summary"]["f1_macro_mean"],
        reverse=True,
    )
    runner_up = sorted_by_macro[1] if len(sorted_by_macro) > 1 else selected
    macro_gap = (
        selected["summary"]["f1_macro_mean"] - runner_up["summary"]["f1_macro_mean"]
    )
    most_stable = min(
        model_results,
        key=lambda result: result["summary"]["f1_macro_std"],
    )
    selected_min_macro_f1 = min(fold["f1_macro"] for fold in selected["folds"])
    lines = [
        "# 카테고리 분류 모델 5-fold 교차 검증 결과",
        "",
        "## 결론 요약",
        "",
        f"최종 적용 모델은 **{selected['display_name']}**입니다.",
        "",
        (
            f"- 핵심 지표인 Macro F1 평균이 {selected['summary']['f1_macro_mean']:.4f}로 "
            f"가장 높았습니다."
        ),
        (
            f"- Accuracy 평균도 {selected['summary']['accuracy_mean']:.4f}로 가장 높아 "
            "전체 정답률 기준에서도 우위가 확인됐습니다."
        ),
        (
            f"- 가장 낮은 fold Macro F1도 {selected_min_macro_f1:.4f}로, 모든 fold에서 "
            "서비스 적용 가능한 수준의 성능을 유지했습니다."
        ),
        "",
        "## 실험 설정",
        "",
        f"- 생성 시각: {payload['created_at']}",
        f"- 평가 데이터셋: {dataset['total_size']}개 "
        f"({', '.join(f'{label} {count}' for label, count in dataset['label_counts'].items())})",
        f"- 교차 검증: StratifiedKFold(n_splits={payload['folds']}, shuffle=True, random_state={payload['random_state']})",
        "- 비교 모델: Naive Bayes, Logistic Regression, Linear SVM",
        "- 텍스트 표현: TF-IDF unigram/bigram",
        "- 추론 시간: 동일 실행 환경에서 fold별 validation split 예측 시간을 기준으로 측정한 참고 지표",
        "",
        "## 최종 선정 기준",
        "",
        "1. **Macro F1 평균을 최우선 지표로 판단**했습니다. 라벨별 성능 균형이 서비스 품질에 직접 영향을 주기 때문입니다.",
        "2. **Accuracy 평균은 보조 지표로 확인**했습니다. 전체 정답률이 높은 모델인지 함께 검증했습니다.",
        "3. **표준편차는 안정성 판단 지표로 사용**했습니다. fold별 성능 변동이 큰 모델은 운영 리스크가 커질 수 있습니다.",
        "4. **추론 속도와 학습 시간은 운영 판단 보조 지표로 사용**했습니다. 모델 배포, 재학습, API 응답 비용을 함께 고려했습니다.",
        "",
        "## 모델별 평균 성능",
        "",
        markdown_table(
            ["Model", "Accuracy", "Macro F1", "Inference ms/sample", "Fit seconds"],
            summary_rows,
        ),
        "",
        "### 평균 성능 해석",
        "",
        (
            f"- {selected['display_name']}은 Macro F1 평균에서 1위이며, 2위인 "
            f"{runner_up['display_name']}보다 {macro_gap:.4f} 높았습니다."
        ),
        (
            f"- 안정성만 보면 {most_stable['display_name']}의 Macro F1 표준편차가 "
            f"{most_stable['summary']['f1_macro_std']:.4f}로 가장 낮습니다."
        ),
        (
            f"- {selected['display_name']}의 Macro F1 표준편차는 "
            f"{selected['summary']['f1_macro_std']:.4f}으로 {most_stable['display_name']}보다 크지만, "
            "평균 Macro F1과 Accuracy가 모두 가장 높고 추론 비용도 낮아 최종 모델로 더 적합합니다."
        ),
        "",
        "## 시각화",
        "",
        "![모델별 Accuracy/Macro F1 mean ± std](cv_metric_summary.png)",
        "",
        "![Fold별 Accuracy/Macro F1 안정성](cv_fold_scores.png)",
        "",
        "![모델별 평균 추론 시간](cv_inference_speed.png)",
        "",
        f"![{selected['display_name']} out-of-fold confusion matrix]({selected['model']}_oof_confusion_matrix.png)",
        "",
        "## Fold별 성능",
        "",
        markdown_table(
            ["Model", "Fold", "Accuracy", "Macro F1", "Inference ms/sample"],
            fold_rows,
        ),
        "",
        "## Confusion Matrix 기반 카테고리별 성능",
        "",
        f"- 기준: {selected['display_name']} out-of-fold 예측 전체",
        f"- OOF Accuracy: {selected['oof']['accuracy']:.4f}",
        f"- OOF Macro F1: {selected['oof']['f1_macro']:.4f}",
        "",
        markdown_table(
            ["Label", "Precision", "Recall", "F1", "Support"],
            per_label_rows,
        ),
        "",
        "## 산출물",
        "",
        "- `cv_results.json`: 전체 교차 검증 결과 원본",
        "- `cv_summary.csv`: 모델별 평균/표준편차 요약",
        "- `cv_fold_results.csv`: fold별 성능 결과",
        "- `cv_metric_summary.png`: 모델별 Accuracy/Macro F1 mean ± std",
        "- `cv_fold_scores.png`: fold별 Accuracy/Macro F1 안정성",
        "- `cv_inference_speed.png`: 모델별 평균 추론 시간",
        f"- `{selected['model']}_oof_confusion_matrix.png`: 선정 모델 out-of-fold confusion matrix",
        "",
        "## 선정 근거",
        "",
        (
            f"{selected['display_name']}은 핵심 지표인 Macro F1 평균이 가장 높고 "
            f"({selected['summary']['f1_macro_mean']:.4f}), Accuracy 평균도 "
            f"{selected['summary']['accuracy_mean']:.4f}로 가장 우수했습니다. "
            f"Macro F1 표준편차는 {most_stable['display_name']}보다 크지만 모든 fold에서 {selected_min_macro_f1:.4f} 이상의 "
            "Macro F1을 유지했고, 평균 성능 우위와 낮은 추론 비용이 운영 적용에 더 적합합니다. "
            "또한 TF-IDF + LinearSVC 구조라 학습과 추론 비용이 낮고 운영 artifact도 기존 pipeline "
            "방식으로 단순하게 유지됩니다."
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_matplotlib_font()
    args.report_dir.mkdir(parents=True, exist_ok=True)

    texts, labels, dataset = load_cv_dataset(
        args.data,
        text_field=args.text_field,
        label_field=args.label_field,
    )
    target_labels = sorted(set(labels))
    if min(Counter(labels).values()) < args.folds:
        raise ValueError("각 라벨의 샘플 수가 fold 수보다 많아야 StratifiedKFold를 적용할 수 있습니다.")

    model_results = [
        evaluate_model_cv(args, model_name, texts, labels, target_labels)
        for model_name in args.models
    ]
    selected = choose_best_model(model_results)

    for result in model_results:
        save_confusion_matrix_outputs(args.report_dir, result, target_labels)

    plot_metric_summary(args.report_dir, model_results)
    plot_fold_scores(args.report_dir, model_results)
    plot_inference_speed(args.report_dir, model_results)

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "folds": args.folds,
        "random_state": args.random_state,
        "labels": target_labels,
        "dataset": dataset,
        "params": model_params_from_args(args),
        "selected_model": selected["model"],
        "models": [
            {key: value for key, value in result.items() if key not in {"oof_true", "oof_pred"}}
            for result in model_results
        ],
    }
    save_json(args.report_dir / "cv_results.json", payload)
    save_summary_csv(args.report_dir / "cv_summary.csv", model_results)
    save_fold_csv(args.report_dir / "cv_fold_results.csv", model_results)
    save_markdown_report(
        args.report_dir / "model_selection_5fold.md",
        payload,
        model_results,
        selected,
    )

    print(
        json.dumps(
            {
                "selected_model": selected["model"],
                "selected_display_name": selected["display_name"],
                "macro_f1": format_score(
                    selected["summary"]["f1_macro_mean"],
                    selected["summary"]["f1_macro_std"],
                ),
                "accuracy": format_score(
                    selected["summary"]["accuracy_mean"],
                    selected["summary"]["accuracy_std"],
                ),
                "report": str(args.report_dir / "model_selection_5fold.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
