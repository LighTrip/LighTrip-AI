from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from experiments.category_classifier.src.preprocess import load_stopwords, tokenize_ko


def build_tfidf_nb_pipeline(
    *,
    stopwords_path: Path | None = None,
    max_features: int | None = 20000,
    min_df: int = 1,
    max_df: float = 0.95,
    ngram_max: int = 2,
    alpha: float = 1.0,
) -> Pipeline:
    stopwords = load_stopwords(stopwords_path)
    tokenizer = partial(tokenize_ko, stopwords=stopwords)

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenizer,
                    token_pattern=None,
                    lowercase=False,
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB(alpha=alpha)),
        ]
    )


def build_pipeline(model_name: str, **kwargs: Any) -> Pipeline:
    if model_name == "nb":
        return build_tfidf_nb_pipeline(**kwargs)
    raise ValueError(f"지원하지 않는 모델입니다: {model_name}")
