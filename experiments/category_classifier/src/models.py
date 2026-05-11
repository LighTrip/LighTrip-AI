from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from experiments.category_classifier.src.preprocess import load_stopwords, tokenize_ko

SUPPORTED_MODELS = ("nb", "logistic_regression", "linear_svm", "calibrated_linear_svm")
MODEL_DISPLAY_NAMES = {
    "nb": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "calibrated_linear_svm": "Calibrated Linear SVM",
}


def build_tfidf_vectorizer(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
) -> TfidfVectorizer:
    stopwords = load_stopwords(stopwords_path)
    tokenizer = partial(tokenize_ko, stopwords=stopwords)

    return TfidfVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def build_tfidf_nb_pipeline(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                build_tfidf_vectorizer(
                    stopwords_path=stopwords_path,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_max=ngram_max,
                ),
            ),
            ("classifier", MultinomialNB(alpha=alpha)),
        ]
    )


def build_tfidf_logistic_regression_pipeline(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
    c: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    class_weight: str | None = None,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                build_tfidf_vectorizer(
                    stopwords_path=stopwords_path,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_max=ngram_max,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    solver=solver,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def build_tfidf_linear_svm_pipeline(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
    c: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = None,
) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                build_tfidf_vectorizer(
                    stopwords_path=stopwords_path,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_max=ngram_max,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=c,
                    dual=True,
                    max_iter=max_iter,
                    class_weight=class_weight,
                ),
            ),
        ]
    )


def build_tfidf_calibrated_linear_svm_pipeline(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
    c: float = 1.0,
    max_iter: int = 1000,
    class_weight: str | None = None,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 5,
) -> Pipeline:
    base_estimator = LinearSVC(
        C=c,
        dual=True,
        max_iter=max_iter,
        class_weight=class_weight,
    )

    return Pipeline(
        [
            (
                "tfidf",
                build_tfidf_vectorizer(
                    stopwords_path=stopwords_path,
                    max_features=max_features,
                    min_df=min_df,
                    max_df=max_df,
                    ngram_max=ngram_max,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=base_estimator,
                    method=calibration_method,
                    cv=calibration_cv,
                ),
            ),
        ]
    )


def build_pipeline(model_name: str, **kwargs: Any) -> Pipeline:
    if model_name == "nb":
        return build_tfidf_nb_pipeline(
            stopwords_path=kwargs.get("stopwords_path"),
            max_features=kwargs.get("max_features", 20000),
            min_df=kwargs.get("min_df", 1),
            max_df=kwargs.get("max_df", 0.95),
            ngram_max=kwargs.get("ngram_max", 2),
            alpha=kwargs.get("alpha", 1.0),
        )
    if model_name == "logistic_regression":
        return build_tfidf_logistic_regression_pipeline(
            stopwords_path=kwargs.get("stopwords_path"),
            max_features=kwargs.get("max_features", 20000),
            min_df=kwargs.get("min_df", 1),
            max_df=kwargs.get("max_df", 0.95),
            ngram_max=kwargs.get("ngram_max", 2),
            c=kwargs.get("c", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            solver=kwargs.get("solver", "lbfgs"),
            class_weight=kwargs.get("class_weight"),
        )
    if model_name == "linear_svm":
        return build_tfidf_linear_svm_pipeline(
            stopwords_path=kwargs.get("stopwords_path"),
            max_features=kwargs.get("max_features", 20000),
            min_df=kwargs.get("min_df", 1),
            max_df=kwargs.get("max_df", 0.95),
            ngram_max=kwargs.get("ngram_max", 2),
            c=kwargs.get("c", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=kwargs.get("class_weight"),
        )
    if model_name == "calibrated_linear_svm":
        return build_tfidf_calibrated_linear_svm_pipeline(
            stopwords_path=kwargs.get("stopwords_path"),
            max_features=kwargs.get("max_features", 20000),
            min_df=kwargs.get("min_df", 1),
            max_df=kwargs.get("max_df", 0.95),
            ngram_max=kwargs.get("ngram_max", 2),
            c=kwargs.get("c", 1.0),
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=kwargs.get("class_weight"),
            calibration_method=kwargs.get("calibration_method", "sigmoid"),
            calibration_cv=kwargs.get("calibration_cv", 5),
        )
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")


def add_single_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default="linear_svm",
        choices=SUPPORTED_MODELS,
    )


def add_multi_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_MODELS),
        choices=SUPPORTED_MODELS,
    )


def add_model_hyperparameter_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Logistic Regression/Linear SVM regularization strength inverse.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Logistic Regression/Linear SVM 최대 반복 횟수입니다.",
    )
    parser.add_argument("--solver", default="lbfgs", help="Logistic Regression solver입니다.")
    parser.add_argument(
        "--class-weight",
        choices=["balanced"],
        help="Logistic Regression/Linear SVM class_weight 옵션입니다.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibrated Linear SVM의 probability calibration 방법입니다.",
    )
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=5,
        help="Calibrated Linear SVM 내부 교차 검증 fold 수입니다.",
    )


def model_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_df": args.max_df,
        "ngram_max": args.ngram_max,
        "alpha": args.alpha,
        "c": args.c,
        "max_iter": args.max_iter,
        "solver": args.solver,
        "class_weight": args.class_weight,
        "calibration_method": args.calibration_method,
        "calibration_cv": args.calibration_cv,
    }


def build_pipeline_from_args(args: argparse.Namespace, model_name: str) -> Pipeline:
    return build_pipeline(
        model_name,
        stopwords_path=args.stopwords,
        **model_params_from_args(args),
    )
