# [KAU] LighTrip AI Repository

한국항공대학교 산학 프로젝트 **LighTrip AI** 레포지토리입니다.

LighTrip AI는 사용자 이미지로부터 한국어 블로그 스타일 초안을 생성하고, 생성된 초안을 서비스 카테고리로 분류하는 AI 기능을 제공합니다.

## Developer

| <img src="https://avatars.githubusercontent.com/u/166575866?v=4" width="150" height="150"/> |
| :-: |
| Yoonsung Jung<br/>[@coouir](https://github.com/coouir) |

## Core Features

| Feature | Model / Method | Description |
| --- | --- | --- |
| Image -> Draft Generation | Gemma 4 E2B (GGUF) | 사용자 이미지와 프롬프트를 기반으로 한국어 블로그 스타일 초안(2-3줄) 생성 |
| Category Classification | TF-IDF + Linear SVM | 생성된 한국어 블로그 초안을 서비스 카테고리로 분류 |
| Places365 Draft Dataset Pipeline | Places365 + Gemma draft generation | Places365 이미지를 카테고리 분류 학습용 JSONL 데이터셋으로 변환 |

## Project Structure

```text
LighTrip-AI/
├── app/
│   ├── api/
│   │   └── gemma.py
│   ├── services/
│   │   └── gemma_service.py
│   └── main.py
├── configs/
│   ├── dataset_categories.json
│   └── places365_categories.json
├── data_places365/
│   ├── 카페/
│   ├── 식당/
│   ├── 술집/
│   ├── 문화/
│   ├── 운동/
│   ├── 쇼핑/
│   ├── 공원/
│   ├── interim/
│   └── processed/
├── docs/
│   └── category_classifier/
│       └── cv_5fold/
├── experiments/
│   ├── category_classifier/
│   │   ├── src/
│   │   ├── compare_cv.py
│   │   ├── infer.py
│   │   ├── train.py
│   │   └── stopwords_ko.txt
│   └── gemma/
│       └── v1_prompt.py
├── scripts/
│   └── dataset/
│       ├── collect_places365.py
│       ├── generate_places365_drafts.py
│       ├── split_places365_dataset.py
│       └── validate_drafts.py
├── requirements-classifier.txt
├── requirements-dataset.txt
└── README.md
```

| Path | Description |
| --- | --- |
| `app/` | FastAPI 기반 AI serving 코드 |
| `configs/` | 카테고리 매핑, Places365 설정, 초안 생성 프롬프트 |
| `data_places365/` | Places365 기반 이미지 원천, 중간 산출물, train/valid/test JSONL |
| `docs/category_classifier/cv_5fold/` | 5-fold 모델 비교 보고서, CSV, 그래프 산출물 |
| `experiments/category_classifier/` | TF-IDF 카테고리 분류 학습, 추론, 교차검증 실험 코드 |
| `experiments/gemma/` | Gemma 초안 생성 실험 코드 |
| `scripts/dataset/` | 데이터 수집, 초안 생성, split, 검증 스크립트 |

## Category Classification

### Final Model

- Selected model: **TF-IDF + Linear SVM**
- Service labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타
- Training/evaluation labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원
- Model selection report: `docs/category_classifier/cv_5fold/model_selection_5fold.md`

### Model Selection Summary

Naive Bayes, Logistic Regression, Linear SVM을 동일 데이터셋 기준으로 비교했고, 5-fold Stratified 교차 검증 결과 **Linear SVM**을 최종 모델로 선정했습니다.

| Metric | Linear SVM |
| --- | --- |
| Macro F1 | `0.9281 ± 0.0213` |
| Accuracy | `0.9282 ± 0.0213` |

선정 기준은 Macro F1 평균을 최우선으로 두고, Accuracy 평균, fold별 표준편차, 추론 속도와 학습 시간을 운영 관점의 보조 지표로 함께 고려했습니다.

## API Serving

FastAPI 앱은 기존 Gemma 초안 생성 API와 통합 AI 파이프라인 API를 함께 제공합니다.

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

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Gemma 모델과 카테고리 분류 모델 로드 상태 확인 |
| `POST` | `/gemma/generate` | 이미지 기반 블로그 초안 생성 |
| `POST` | `/pipeline/generate-and-classify` | 이미지 기반 블로그 초안 생성 후 TF-IDF + calibrated Linear SVM 카테고리 분류 |

### Pipeline Request

`multipart/form-data` 형식으로 요청합니다.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `image` | file | Yes | `jpg`, `png`, `webp` 이미지 |
| `prompt` | string | No | 초안 생성에 반영할 사용자 요청 |
| `unknown_threshold` | float | No | 낮은 confidence를 `기타`로 처리할 기준값. 생략하면 `CATEGORY_UNKNOWN_THRESHOLD`를 사용 |

```bash
curl -X POST "http://localhost:8000/pipeline/generate-and-classify" \
  -F "image=@sample.jpg" \
  -F "prompt=따뜻한 일상 기록 느낌으로 작성해줘"
```

디버그 정보가 필요할 때는 쿼리 파라미터로 `debug=true`를 추가합니다.

```bash
curl -X POST "http://localhost:8000/pipeline/generate-and-classify?debug=true" \
  -F "image=@sample.jpg" \
  -F "prompt=따뜻한 일상 기록 느낌으로 작성해줘"
```

### Pipeline Response

기본 응답은 서비스 연동에 필요한 초안과 카테고리만 반환합니다.

```json
{
  "success": true,
  "data": {
    "generated_text": "오늘은 커피 향이 유난히 좋았다.\n잠깐 쉬어가는 시간이 이렇게 반가울 줄 몰랐다.",
    "category": "카페"
  }
}
```

`debug=true`일 때만 분류 score, 모델명, 처리 시간 등의 진단 정보를 함께 반환합니다.

```json
{
  "success": true,
  "data": {
    "generated_text": "오늘은 커피 향이 유난히 좋았다.\n잠깐 쉬어가는 시간이 이렇게 반가울 줄 몰랐다.",
    "category": "카페"
  },
  "debug": {
    "category": {
      "label": "카페",
      "raw_label": "카페",
      "confidence": 0.7421,
      "score": 0.7421,
      "scores": {
        "카페": 0.7421,
        "식당": 0.1084
      },
      "model": "calibrated_linear_svm"
    },
    "filename": "sample.jpg",
    "prompt": "따뜻한 일상 기록 느낌으로 작성해줘",
    "elapsed_seconds": 3.42
  }
}
```

`calibrated_linear_svm` artifact는 `predict_proba` 기반 confidence를 반환하며, confidence가 threshold보다 낮으면 응답 카테고리를 `기타`로 바꿉니다. 기본 `linear_svm` artifact는 `predict_proba`를 제공하지 않으므로 fallback 운영에는 calibrated artifact를 사용합니다.

## Places365 Draft Dataset Pipeline

### Goal

Places365 이미지를 LighTrip 서비스 카테고리에 매핑한 뒤, Gemma 기반 한국어 블로그 초안을 생성해 카테고리 분류 학습용 JSONL 데이터셋을 구축합니다.

### Dataset Policy

- Data source: Places365 scene categories mapped to LighTrip service categories
- Dataset labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원
- Service inference labels: 카페, 식당, 술집, 문화, 운동, 쇼핑, 공원, 기타
- Recommended target: 서비스 카테고리별 180개 이상, 총 1,000개 이상
- Draft prompt policy: 카테고리 분류 학습 품질을 높이기 위해 각 라벨의 약한 힌트를 Gemma4 프롬프트에 자동 추가

서비스 API의 기본 프롬프트는 데이터셋 생성용 힌트와 분리해 유지합니다.

### Dataset Structure

```text
data_places365/
├── 카페/
│   ├── coffee_shop/
│   └── ice_cream_parlor/
├── 식당/
│   ├── cafeteria/
│   ├── fastfood_restaurant/
│   ├── food_court/
│   ├── pizzeria/
│   ├── restaurant/
│   └── sushi_bar/
├── 술집/
│   ├── bar/
│   ├── beer_garden/
│   ├── beer_hall/
│   ├── discotheque/
│   └── wet_bar/
├── 문화/
│   ├── amphitheater/
│   ├── art_gallery/
│   ├── art_school/
│   ├── art_studio/
│   ├── artists_loft/
│   ├── auditorium/
│   ├── music_studio/
│   ├── natural_history_museum/
│   └── science_museum/
├── 운동/
│   ├── baseball_field/
│   ├── bowling_alley/
│   ├── boxing_ring/
│   ├── football_field/
│   ├── golf_course/
│   ├── martial_arts_gym/
│   ├── racecourse/
│   ├── ski_resort/
│   ├── ski_slope/
│   └── soccer_field/
├── 쇼핑/
│   ├── bookstore/
│   ├── clothing_store/
│   ├── department_store/
│   ├── gift_shop/
│   └── shoe_shop/
├── 공원/
│   ├── botanical_garden/
│   ├── formal_garden/
│   ├── japanese_garden/
│   ├── lawn/
│   ├── park/
│   ├── picnic_area/
│   └── playground/
├── processed/
│   ├── train.jsonl
│   ├── valid.jsonl
│   └── test.jsonl
└── interim/
```

## Reports

| Report | Path |
| --- | --- |
| 5-fold model selection report | `docs/category_classifier/cv_5fold/model_selection_5fold.md` |
| CV summary CSV | `docs/category_classifier/cv_5fold/cv_summary.csv` |
| Fold-level CV results | `docs/category_classifier/cv_5fold/cv_fold_results.csv` |
| CV result JSON | `docs/category_classifier/cv_5fold/cv_results.json` |

## Tech Stack

| Area | Tools |
| --- | --- |
| Image/draft generation | PyTorch, HuggingFace Transformers, llama-cpp-python |
| Category classification | scikit-learn, NumPy, joblib |
| Evaluation/visualization | scikit-learn, matplotlib |
| Serving | FastAPI |

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
| `design` | CSS 등 사용자 UI 디자인을 변경할 때 |
| `refactor` | 기능 변경 없이 코드를 리팩토링할 때 |
| `settings` | 설정 파일을 변경할 때 |
| `comment` | 필요한 주석을 추가하거나 변경할 때 |
| `dependency/Plugin` | 의존성/플러그인을 추가할 때 |
| `docs` | README.md 등 문서를 수정할 때 |
| `merge` | 브랜치를 병합할 때 |
| `deploy` | 빌드 및 배포 관련 작업을 할 때 |
| `rename` | 파일 혹은 폴더명을 수정하거나 옮길 때 |
| `remove` | 파일을 삭제하는 작업만 수행했을 때 |
| `revert` | 이전 버전으로 롤백할 때 |

예시:

```bash
feat: 로그인 기능 추가
fix: 로그인 버그 수정
refactor: 로그인 로직 리팩토링
```
