from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Optional, Union

import joblib


def required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f"필수 환경변수가 설정되지 않았습니다: {name}")
    return value.strip()


CATEGORY_ARTIFACT_PATH: Final[Path] = Path(required_env("CATEGORY_ARTIFACT_PATH"))
CATEGORY_UNKNOWN_LABEL: Final[str] = required_env("CATEGORY_UNKNOWN_LABEL")

_artifact: Optional[Dict[str, Any]] = None
_pipeline: Optional[Any] = None
_metadata: Dict[str, Any] = {}


@dataclass(frozen=True)
class CategoryPrediction:
    label: str
    raw_label: str
    confidence: Optional[float]
    score: Optional[float]
    scores: Optional[Dict[str, float]]
    model: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "raw_label": self.raw_label,
            "confidence": self.confidence,
            "score": self.score,
            "scores": self.scores,
            "model": self.model,
        }


def get_category_model() -> Optional[Any]:
    return _pipeline


def is_category_model_loaded() -> bool:
    return _pipeline is not None


def unload_category_model() -> None:
    global _artifact, _pipeline, _metadata

    _artifact = None
    _pipeline = None
    _metadata = {}


def load_category_model(artifact_path: Union[Path, str] = CATEGORY_ARTIFACT_PATH) -> None:
    global _artifact, _pipeline, _metadata

    if _pipeline is not None:
        return

    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"카테고리 분류 모델 artifact를 찾을 수 없습니다: {path}")

    loaded_artifact = joblib.load(path)
    if not isinstance(loaded_artifact, dict) or "pipeline" not in loaded_artifact:
        raise ValueError("카테고리 분류 artifact 형식이 올바르지 않습니다.")

    _artifact = loaded_artifact
    _pipeline = loaded_artifact["pipeline"]
    _metadata = loaded_artifact.get("metadata", {})


def _max_probability(pipeline: Any, text: str) -> Optional[float]:
    if not hasattr(pipeline, "predict_proba"):
        return None

    probabilities = pipeline.predict_proba([text])
    return float(probabilities[0].max())


def _decision_scores(pipeline: Any, text: str) -> Optional[Dict[str, float]]:
    if not hasattr(pipeline, "decision_function"):
        return None
    if not hasattr(pipeline, "classes_"):
        return None

    raw_scores = pipeline.decision_function([text])[0]
    classes = list(pipeline.classes_)

    if len(classes) == 2 and not hasattr(raw_scores, "__iter__"):
        negative_score = -float(raw_scores)
        positive_score = float(raw_scores)
        return {
            str(classes[0]): negative_score,
            str(classes[1]): positive_score,
        }

    return {
        str(label): float(score)
        for label, score in zip(classes, raw_scores)
    }


def _apply_unknown_threshold(
    raw_label: str,
    confidence: Optional[float],
    unknown_threshold: Optional[float],
) -> str:
    if unknown_threshold is None or confidence is None:
        return raw_label
    if confidence < unknown_threshold:
        return CATEGORY_UNKNOWN_LABEL
    return raw_label


def classify_text(
    text: str,
    unknown_threshold: Optional[float] = None,
) -> CategoryPrediction:
    pipeline = get_category_model()
    if pipeline is None:
        raise RuntimeError("카테고리 분류 모델이 아직 로드되지 않았습니다.")

    normalized_text = text.strip()
    if not normalized_text:
        raise ValueError("분류할 텍스트가 비어 있습니다.")

    raw_label = str(pipeline.predict([normalized_text])[0])
    confidence = _max_probability(pipeline, normalized_text)
    scores = _decision_scores(pipeline, normalized_text)
    score = max(scores.values()) if scores else None
    label = _apply_unknown_threshold(raw_label, confidence, unknown_threshold)

    return CategoryPrediction(
        label=label,
        raw_label=raw_label,
        confidence=confidence,
        score=score,
        scores=scores,
        model=_metadata.get("model"),
    )
