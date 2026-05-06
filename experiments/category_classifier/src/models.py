from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from experiments.category_classifier.src.preprocess import load_stopwords, tokenize_ko


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
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
