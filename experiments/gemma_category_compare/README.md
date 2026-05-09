# Gemma Direct Category vs Split Classifier Experiment

이 실험은 이미지 입력에서 Gemma가 블로그 초안과 카테고리를 한 번에 생성하는 구조와, 현재 서비스처럼 Gemma 초안 생성 뒤 TF-IDF + Linear SVM이 카테고리를 분류하는 구조를 비교한다.

## 비교 구조

| method | 의미 |
| --- | --- |
| `classifier_existing_draft` | `data_places365/processed/test.jsonl`의 저장된 Gemma 초안을 Linear SVM으로 분류한다. SVM 분류기 자체의 빠른 기준선이다. |
| `gemma_direct` | 기존 서비스 초안 생성 프롬프트의 작성 규칙을 반영한 프롬프트로, 이미지에서 Gemma가 `draft`와 `category`를 JSON으로 직접 생성한다. `--run-gemma-direct`를 켰을 때 실행된다. |
| `split_pipeline` | 이미지에서 Gemma 초안을 새로 생성하고, 그 초안을 Linear SVM으로 분류한다. 운영 파이프라인 end-to-end에 가장 가깝고 `--run-split-pipeline`을 켰을 때 실행된다. |

## 기본 실행

저장된 초안 기준으로 현재 SVM 성능만 빠르게 확인한다.

```bash
python3 experiments/gemma_category_compare/run_compare.py
```

Gemma 직접 분류를 30개 샘플에서 실행한다.

```bash
python3 experiments/gemma_category_compare/run_compare.py \
  --limit 30 \
  --run-gemma-direct \
  --repeat 3
```

현재 분리형 구조의 end-to-end 추론 시간까지 비교한다.

```bash
python3 experiments/gemma_category_compare/run_compare.py \
  --limit 30 \
  --run-gemma-direct \
  --run-split-pipeline \
  --repeat 3
```

## Label Set

현재 기본 Gemma 직접 분류 label은 Places365 classifier 데이터셋과 맞춰 `카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타`이다.

실험 목적상 멘토링 초안의 7개 label만 사용하려면 아래처럼 `공원`을 제외할 수 있다.

```bash
python3 experiments/gemma_category_compare/run_compare.py \
  --run-gemma-direct \
  --direct-labels 카페,식당,술집,문화,운동,쇼핑,기타
```

## 산출물

기본 저장 위치는 `experiments/gemma_category_compare/results/<timestamp>/`이다.

- `metrics.json`: Accuracy, Macro F1, 카테고리별 precision/recall/F1, 추론 시간, parse 성공률, 반복 일관성
- `summary.md`: 실험 결과 요약표
- `*_predictions.jsonl`: method별 raw 예측 결과
- `reports/*_classification_report.txt`: 사람이 읽기 쉬운 classification report
- `reports/*_confusion_matrix.csv`: confusion matrix

## Prompt

Gemma 직접 분류 프롬프트는 `prompt_json_strict.txt`를 사용한다. 이 프롬프트는 기존 서비스의 `configs/draft_prompt.txt` 작성 규칙을 바탕으로 하되, 비교 실험을 위해 `draft`와 `category`를 함께 담은 JSON 객체를 출력하도록 확장했다.

## Service-Prompt SVM Retraining

`split_pipeline` 성능이 낮게 나올 때는 SVM이 학습한 초안 스타일과 실제 서비스 초안 스타일이 다를 수 있다. 아래 runner는 실제 서비스 프롬프트(`configs/draft_prompt.txt`)로 Places365 초안을 다시 생성한 뒤, 그 데이터로 Linear SVM을 재학습한다.

먼저 1~2개만 명령 확인:

```bash
python3 experiments/gemma_category_compare/run_service_prompt_retrain.py \
  --limit-total 2 \
  --overwrite-drafts \
  --dry-run
```

작은 샘플로 실제 생성/학습 흐름 테스트:

```bash
python3 experiments/gemma_category_compare/run_service_prompt_retrain.py \
  --limit-total 70 \
  --overwrite-drafts \
  --draft-output data_places365/interim/places365_service_prompt_drafts_sample.jsonl \
  --processed-dir data_places365/processed_service_prompt_sample \
  --artifact-dir experiments/gemma_category_compare/artifacts/service_prompt_classifier_sample \
  --report-dir experiments/gemma_category_compare/reports/service_prompt_classifier_sample
```

전체 데이터 재생성 및 SVM 재학습:

```bash
python3 experiments/gemma_category_compare/run_service_prompt_retrain.py \
  --overwrite-drafts
```

재학습 후 Gemma direct와 split pipeline 비교까지 실행:

```bash
python3 experiments/gemma_category_compare/run_service_prompt_retrain.py \
  --skip-generate \
  --skip-split \
  --skip-train \
  --run-compare
```

GPU 메모리가 불안하면 `--safe-hybrid`를 붙인다. 이 경우 속도는 서비스 GPU 설정보다 느리고, 측정 시간은 CPU-GPU hybrid 기준이다.

## 환경변수

`classifier_existing_draft`만 실행할 때는 Gemma 환경변수가 필요 없다.

`gemma_direct` 또는 `split_pipeline` 실행 시에는 스크립트가 아래 로컬 파일이 존재하면 기본 환경변수를 자동으로 채운다.

```text
models/gemma-4-E2B-it-Q4_K_S.gguf
models/mmproj-F16.gguf
configs/draft_prompt.txt
```

다른 모델 파일을 쓰거나 운영 환경 설정과 맞추려면 기존 API와 동일하게 환경변수를 직접 지정하면 된다. 직접 지정한 값은 자동 기본값보다 우선한다.

```text
GEMMA_MODEL_PATH
GEMMA_MMPROJ_PATH
GEMMA_PROMPT_PATH
GEMMA_N_CTX
GEMMA_MAX_TOKENS
GEMMA_TEMPERATURE
GEMMA_TOP_P
GEMMA_TOP_K
GEMMA_REPEAT_PENALTY
GEMMA_STOP_TOKENS
GEMMA_N_GPU_LAYERS
GEMMA_MAIN_GPU
GEMMA_OFFLOAD_KQV
GEMMA_MMPROJ_USE_GPU
```

자동 기본값 주입을 끄려면 `--no-local-gemma-defaults`를 사용한다.
