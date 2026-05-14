# [KAU] LighTrip AI Repository

한국항공대학교 산학 프로젝트 **LighTrip AI** 레포지토리입니다.

LighTrip AI는 사용자 이미지로부터 한국어 블로그 스타일 초안과 서비스 카테고리를 생성하는 AI 기능을 제공합니다. 기본 경로는 Gemma Direct이며, 카테고리 출력이 비정상일 때 calibrated SVM fallback을 사용합니다.

## Developer

| <img src="https://avatars.githubusercontent.com/u/166575866?v=4" width="150" height="150"/> |
| :-: |
| Yoonsung Jung<br/>[@coouir](https://github.com/coouir) |

## Core Features

| Feature | Model / Method | Description |
| --- | --- | --- |
| Image -> Draft + Category | Gemma 4 E2B (GGUF) | 사용자 이미지와 선택 입력 텍스트를 기반으로 한국어 블로그 스타일 초안과 카테고리 생성 |
| Category Fallback | TF-IDF + calibrated Linear SVM | Gemma가 카테고리를 누락하거나, 비우거나, 허용되지 않은 카테고리를 출력한 경우 fallback 분류 |
| Places365 Draft Dataset Pipeline | Places365 + Gemma draft generation | Places365 이미지를 카테고리 분류 학습용 JSONL 데이터셋으로 변환 |

## API Serving

FastAPI 앱은 Gemma Direct 기반 통합 AI 파이프라인 API를 제공합니다.

### Install

```bash
pip install -r requirements-api.txt
```

### Run

모델 파일명, 경로, 추론 파라미터는 GitHub에 올리지 않고 실행 환경에서만 설정합니다.
아래 환경변수들은 로컬 `.env`, 서버 secret, 또는 shell export로 주입합니다.

Required environment variables:

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
CATEGORY_ARTIFACT_PATH
CATEGORY_UNKNOWN_LABEL
```

Optional environment variables:

```text
CATEGORY_UNKNOWN_THRESHOLD
```

Fallback 운영에는 calibrated SVM artifact를 사용합니다.

```bash
export CATEGORY_ARTIFACT_PATH=experiments/category_classifier/artifacts/places365_2_manual_full_calibrated/calibrated_linear_svm_tfidf.joblib
export CATEGORY_UNKNOWN_LABEL=기타
```

Gemma Direct는 기본 초안 프롬프트에 JSON 출력 규칙을 추가하므로, `draft_prompt_boundary_v2.txt` 사용 시 `GEMMA_N_CTX`는 최소 `2048` 이상을 권장합니다.

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | 서버 실행 상태 확인 |
| `GET` | `/health` | Gemma 모델과 카테고리 fallback 모델 로드 상태 확인 |
| `POST` | `/pipeline/generate` | Gemma Direct로 초안과 카테고리를 생성하고, 카테고리 이상 출력 시 calibrated Linear SVM fallback 적용 |

### Pipeline Request

`multipart/form-data` 형식으로 요청합니다.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `image` | File | Yes | `jpg`, `jpeg`, `png`, `webp` 이미지. OpenAPI/Swagger에서는 `string($binary)`로 표시될 수 있음 |
| `text` | string | No | 초안 생성에 반영할 사용자 요청 |

```bash
curl -X POST "http://localhost:8000/pipeline/generate" \
  -F "image=@sample.jpg" \
  -F "text=따뜻한 일상 기록 느낌으로 작성해줘"
```

SVM fallback threshold는 API 입력으로 받지 않고 `CATEGORY_UNKNOWN_THRESHOLD` 환경변수로 관리합니다. fallback 여부와 SVM 진단 정보는 서버 로그에만 기록됩니다.

### Pipeline Response

응답은 서비스 연동에 필요한 초안과 카테고리만 반환합니다.

```json
{
  "draft": "오늘은 커피 향이 유난히 좋았다.\n잠깐 쉬어가는 시간이 이렇게 반가울 줄 몰랐다.",
  "category": "카페"
}
```

운영 label set은 `카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타`입니다. Gemma가 `category`를 비우거나, `category` 필드를 누락하거나, 허용 label set 밖의 값을 출력하면 SVM fallback 결과를 최종 `category`로 반환합니다.

`calibrated_linear_svm` artifact는 fallback 발생 시 `predict_proba` 기반 confidence를 반환하며, confidence가 threshold보다 낮으면 최종 카테고리를 `기타`로 바꿉니다. 기본 `linear_svm` artifact는 `predict_proba`를 제공하지 않으므로 fallback 운영에는 calibrated artifact를 사용합니다.

## Project Structure

```text
LighTrip-AI/
├── app/
│   ├── api/
│   ├── config/
│   ├── prompts/
│   ├── services/
│   └── main.py
├── configs/
│   ├── dataset_categories.json
│   ├── places365_categories.json
│   └── places365_categories_v2.json
├── data/
│   ├── images/
│   ├── interim/
│   └── processed/
├── data_places365/
├── data_places365_2/
│   ├── <label>/
│   ├── final_filtered/
│   ├── interim/
│   ├── manual_review*/
│   ├── processed/
│   ├── quality/
│   ├── semantic_filter/
│   └── splits/
├── docs/
│   ├── category_classifier/
│   │   └── cv_5fold/
│   └── *.md
├── experiments/
│   ├── category_classifier/
│   ├── gemma/
│   └── gemma_category_compare/
├── models/
├── scripts/
│   └── dataset/
├── tests/
├── requirements-api.txt
├── requirements-classifier.txt
├── requirements-dataset.txt
├── run_api.local.sh
└── README.md
```

| Path | Description |
| --- | --- |
| `app/` | FastAPI 기반 AI serving 코드 |
| `app/api/` | Gemma 및 통합 pipeline API endpoint |
| `app/config/` | Gemma runtime/config 로딩 코드 |
| `app/prompts/` | Gemma 프롬프트 포맷터 및 프롬프트 헬퍼 |
| `app/services/` | 초안 생성, 카테고리 정책, fallback 분류 서비스 |
| `configs/` | 카테고리 매핑, Places365 설정, 초안 생성 프롬프트 |
| `data/` | 기본 이미지/중간 산출물/processed 데이터 |
| `data_places365/` | Places365 v1 및 service prompt 실험 데이터 |
| `data_places365_2/` | Places365 v2 이미지, 품질 검사, manual review, split/processed 데이터 |
| `docs/` | 데이터셋 검토, 모델 비교, Gemma/SVM 실험 결과 문서 |
| `docs/category_classifier/` | 카테고리 분류기 재학습 및 fallback threshold 문서 |
| `docs/category_classifier/cv_5fold/` | 5-fold 모델 비교 그래프, CSV, JSON, TXT 산출물 |
| `experiments/category_classifier/` | TF-IDF 카테고리 분류 학습, 추론, 교차검증 실험 코드 |
| `experiments/gemma/` | Gemma 초안 생성 실험 코드 |
| `experiments/gemma_category_compare/` | Gemma Direct와 Gemma + SVM pipeline 비교 실험 코드/결과 |
| `models/` | 로컬 Gemma GGUF, mmproj, SVM artifact |
| `scripts/dataset/` | 데이터 수집, 초안 생성, split, 검증 스크립트 |
| `tests/` | 서비스 정책 및 pipeline 단위 테스트 |

## Category Classification

### Fallback Model

- Fallback model: **TF-IDF + calibrated Linear SVM**
- Service labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타
- Training/evaluation labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원
- Model selection report: `docs/카테고리_분류_모델_5폴드_교차_검증_결과.md`
- Runtime artifact: `experiments/category_classifier/artifacts/places365_2_manual_full_calibrated/calibrated_linear_svm_tfidf.joblib`
- Training data: `data_places365_2/processed/train.jsonl` (`2747` rows)
- Validation data: `data_places365_2/processed/valid.jsonl` (`339` rows)
- Test data: `data_places365_2/processed/test.jsonl` (`339` rows)

### Model Selection Summary

Naive Bayes, Logistic Regression, Linear SVM을 동일 데이터셋 기준으로 비교했고, 5-fold Stratified 교차 검증 결과 **Linear SVM**을 fallback 모델 계열로 선정했습니다. 운영에서는 confidence 기반 `기타` 처리를 위해 calibrated artifact를 사용합니다.

| Metric | Calibrated Linear SVM |
| --- | --- |
| Test Accuracy | `0.8643` |
| Test Macro F1 | `0.8500` |
| Valid Accuracy | `0.8289` |
| Valid Macro F1 | `0.8292` |

선정 기준은 Macro F1 평균을 최우선으로 두고, Accuracy 평균, fold별 표준편차, 추론 속도와 학습 시간을 운영 관점의 보조 지표로 함께 고려했습니다.

## Places365 v2 Dataset Pipeline

### Goal

Places365 이미지를 LighTrip 서비스 카테고리에 매핑한 뒤, manual review와 filtering을 거쳐 Gemma 기반 한국어 블로그 초안을 생성하고 카테고리 fallback 학습용 JSONL 데이터셋을 구축합니다.

### Dataset Policy

- Data source: Places365 scene categories mapped to LighTrip service categories
- Dataset root: `data_places365_2/`
- Mapping config: `configs/places365_categories_v2.json`
- Dataset labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원
- Service inference labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타
- Processed split: `train=2747`, `valid=339`, `test=339`
- Draft prompt policy: `configs/draft_prompt_boundary_v2.txt` 기준의 카테고리 경계 규칙을 반영

서비스 API의 기본 프롬프트는 데이터셋 생성용 힌트와 분리해 유지합니다.

### Dataset Structure

```text
data_places365_2/
├── 카페/
│   ├── coffee_shop/
├── 식당/
│   ├── diner_outdoor/
│   ├── fastfood_restaurant/
│   ├── food_court/
│   ├── pizzeria/
│   ├── restaurant/
│   └── restaurant_patio/
├── 술집/
│   ├── bar/
│   ├── beer_garden/
│   ├── beer_hall/
│   └── pub_indoor/
├── 문화/
│   ├── amphitheater/
│   ├── art_gallery/
│   ├── library_indoor/
│   ├── movie_theater_indoor/
│   ├── museum_indoor/
│   ├── natural_history_museum/
│   └── science_museum/
├── 운동/
│   ├── athletic_field_outdoor/
│   ├── baseball_field/
│   ├── basketball_court_indoor/
│   ├── bowling_alley/
│   ├── boxing_ring/
│   ├── football_field/
│   ├── golf_course/
│   ├── gymnasium_indoor/
│   ├── ice_skating_rink_indoor/
│   ├── ice_skating_rink_outdoor/
│   ├── martial_arts_gym/
│   ├── ski_slope/
│   ├── soccer_field/
│   ├── swimming_pool_indoor/
│   ├── swimming_pool_outdoor/
│   └── volleyball_court_outdoor/
├── 쇼핑/
│   ├── bazaar_indoor/
│   ├── bazaar_outdoor/
│   ├── clothing_store/
│   ├── department_store/
│   ├── flea_market_indoor/
│   ├── general_store_indoor/
│   ├── gift_shop/
│   ├── jewelry_shop/
│   ├── market_indoor/
│   ├── market_outdoor/
│   ├── shoe_shop/
│   ├── shopping_mall_indoor/
│   ├── supermarket/
│   └── toyshop/
├── 공원/
│   ├── botanical_garden/
│   ├── formal_garden/
│   ├── japanese_garden/
│   ├── park/
│   ├── picnic_area/
│   ├── playground/
│   ├── topiary_garden/
│   └── zen_garden/
├── final_filtered/
├── manual_review_full/
├── splits/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── interim/
```

## Reports

| Report | Path |
| --- | --- |
| 5-fold model selection report | `docs/카테고리_분류_모델_5폴드_교차_검증_결과.md` |
| CV summary CSV | `docs/category_classifier/cv_5fold/모델별_성능_요약.csv` |
| Fold-level CV results | `docs/category_classifier/cv_5fold/폴드별_성능_결과.csv` |
| CV result JSON | `docs/category_classifier/cv_5fold/5폴드_전체_결과.json` |

## Tech Stack

| Area | Tools |
| --- | --- |
| Image/draft generation | GGUF Gemma, llama-cpp-python |
| Category classification | scikit-learn, joblib |
| Dataset collection/processing | Hugging Face Datasets, FiftyOne, Pillow |
| Evaluation/visualization | scikit-learn, matplotlib |
| Serving | FastAPI, Uvicorn |

## Development Workflow

### Git-flow Strategy

- `main`: 최종적으로 사용자에게 배포되는 가장 안정적인 버전 브랜치
- `develop`: 다음 출시 버전을 개발하는 중심 브랜치
- `feature/*`: 기능 개발용 브랜치

### Branch Rules

1. 모든 기능 개발은 `feature` 브랜치에서 시작합니다.
2. 작업 시작 전 최신 `develop` 내용을 반영합니다.
3. 작업 완료 후 `develop`으로 Pull Request를 생성합니다.
4. PR에 Reviewer를 지정한 뒤 리뷰를 거쳐 머지합니다.

브랜치 이름 형식:

```text
feature/이슈번호-기능명
```

예시:

```text
feature/1-login
```

### Commit Convention

- `type`은 소문자만 사용합니다.
- `subject`는 현재형 동사로 작성합니다.

| Type | Description |
| --- | --- |
| `start` | 새로운 프로젝트를 시작할 때 |
| `feat` | 새로운 기능을 추가할 때 |
| `fix` | 버그를 수정할 때 |
| `refactor` | 기능 변경 없이 코드를 리팩토링할 때 |
| `settings` | 설정 파일을 변경할 때 |
| `experiment` | 실험 코드나 실험 설정을 추가/변경할 때 |
| `comment` | 필요한 주석을 추가하거나 변경할 때 |
| `docs` | README.md 등 문서를 수정할 때 |
| `merge` | 브랜치를 병합할 때 |
| `rename` | 파일 혹은 폴더명을 수정하거나 옮길 때 |
| `remove` | 파일을 삭제하는 작업만 수행했을 때 |
| `revert` | 이전 버전으로 롤백할 때 |

예시:

```bash
feat: 로그인 기능 추가
fix: 로그인 버그 수정
refactor: 로그인 로직 리팩토링
```
