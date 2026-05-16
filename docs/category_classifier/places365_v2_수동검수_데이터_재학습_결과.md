# Places365 v2 Manual Full Linear SVM 재학습 결과

## 결론

최종 후보 artifact는 `class_weight=balanced`를 적용한 Linear SVM입니다.

- Artifact: `experiments/category_classifier/artifacts/places365_2_manual_full_balanced/linear_svm_tfidf.joblib`
- Test Accuracy: `0.8732`
- Test Macro F1: `0.8641`
- Test Weighted F1: `0.8726`

기본 Linear SVM 대비 Test Accuracy는 `0.8555 -> 0.8732`, Test Macro F1은 `0.8410 -> 0.8641`로 개선되었습니다.

## 데이터셋

- Source: `data/category_classifier/places365_v2/manual_review_full/accepted_drafts.jsonl`
- Total: `3,425`
- Split seed: `42`
- Train: `2,747`
- Validation: `339`
- Test: `339`
- Text field: `generated_text`
- Label field: `label`

라벨 분포는 `공원 879`, `쇼핑 623`, `카페 543`, `운동 492`, `문화 345`, `식당 335`, `술집 208`입니다. `술집`, `식당`, `문화`가 상대적으로 적기 때문에 Macro F1과 라벨별 recall을 함께 확인했습니다.

## 학습 설정

- Feature: TF-IDF unigram/bigram
- Model: Linear SVM (`LinearSVC`)
- `max_features`: `20000`
- `min_df`: `1`
- `max_df`: `0.95`
- `ngram_max`: `2`
- `C`: `1.0`
- `max_iter`: `2000`
- Selected class weight: `balanced`

## 모델 비교

| 설정 | Valid Accuracy | Valid Macro F1 | Test Accuracy | Test Macro F1 | Test Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 기본 Linear SVM | 0.8437 | 0.8411 | 0.8555 | 0.8410 | 0.8544 |
| Linear SVM balanced | 0.8407 | 0.8416 | 0.8732 | 0.8641 | 0.8726 |

Validation Accuracy는 기본 모델이 소폭 높지만, Test Accuracy와 Test Macro F1은 balanced 모델이 더 높습니다. 특히 샘플 수가 적은 `술집`의 Test F1이 `0.81 -> 0.90`으로 개선되어 카테고리별 안정성 관점에서 balanced 모델을 선택하는 것이 더 적합합니다.

## Test 라벨별 성능

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| 공원 | 0.8889 | 0.9195 | 0.9040 | 87 |
| 문화 | 0.8710 | 0.7941 | 0.8308 | 34 |
| 쇼핑 | 0.8906 | 0.9194 | 0.9048 | 62 |
| 술집 | 0.9000 | 0.9000 | 0.9000 | 20 |
| 식당 | 0.7429 | 0.7879 | 0.7647 | 33 |
| 운동 | 0.9200 | 0.9388 | 0.9293 | 49 |
| 카페 | 0.8571 | 0.7778 | 0.8155 | 54 |

## Confusion Matrix 분석

Balanced Linear SVM의 Test confusion matrix 기준 주요 혼동은 다음과 같습니다.

- `카페 -> 식당`: 6건
- `식당 -> 카페`: 4건
- `문화 -> 쇼핑`: 4건
- `공원 -> 운동`: 3건
- `공원 -> 문화`: 2건
- `카페 -> 공원`: 3건

가장 큰 잔여 리스크는 `카페`와 `식당`의 경계입니다. 두 카테고리는 생성 텍스트에서 "맛있는 것", "잠시 쉬어감", "기분 좋은 시간" 같은 표현을 공유하기 쉬워 혼동이 남아 있습니다. 다음 개선 단계에서는 카페/식당 경계 표현을 더 명확히 담은 수동 검수 데이터나 hard negative 샘플을 추가하는 것이 좋습니다.

## 생성 산출물

기본 Linear SVM:

- `experiments/category_classifier/artifacts/places365_2_manual_full/linear_svm_tfidf.joblib`
- `experiments/category_classifier/reports/places365_2_manual_full/linear_svm_metrics.json`
- `experiments/category_classifier/reports/places365_2_manual_full/linear_svm_valid_classification_report.txt`
- `experiments/category_classifier/reports/places365_2_manual_full/linear_svm_test_classification_report.txt`
- `experiments/category_classifier/reports/places365_2_manual_full/linear_svm_valid_confusion_matrix.csv`
- `experiments/category_classifier/reports/places365_2_manual_full/linear_svm_test_confusion_matrix.csv`

Balanced Linear SVM:

- `experiments/category_classifier/artifacts/places365_2_manual_full_balanced/linear_svm_tfidf.joblib`
- `experiments/category_classifier/reports/places365_2_manual_full_balanced/linear_svm_metrics.json`
- `experiments/category_classifier/reports/places365_2_manual_full_balanced/linear_svm_valid_classification_report.txt`
- `experiments/category_classifier/reports/places365_2_manual_full_balanced/linear_svm_test_classification_report.txt`
- `experiments/category_classifier/reports/places365_2_manual_full_balanced/linear_svm_valid_confusion_matrix.csv`
- `experiments/category_classifier/reports/places365_2_manual_full_balanced/linear_svm_test_confusion_matrix.csv`

## 적용 경로

API serving에 적용할 때는 다음 artifact를 지정합니다.

```bash
CATEGORY_ARTIFACT_PATH=experiments/category_classifier/artifacts/places365_2_manual_full_balanced/linear_svm_tfidf.joblib
CATEGORY_UNKNOWN_LABEL=기타
```

confidence threshold 기반 `기타` fallback을 운영하려면 `predict_proba`가 필요하므로 calibrated artifact를 사용합니다. 상세 threshold tuning 결과는 `docs/category_classifier/기타_폴백_임계값_튜닝_결과.md`를 참고합니다.

```bash
CATEGORY_ARTIFACT_PATH=experiments/category_classifier/artifacts/places365_2_manual_full_calibrated/calibrated_linear_svm_tfidf.joblib
CATEGORY_UNKNOWN_LABEL=기타
CATEGORY_UNKNOWN_THRESHOLD=0.49
```
