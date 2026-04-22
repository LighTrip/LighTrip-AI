# [KAU] LighTrip AI Repository
> 한국항공대학교 산학 프로젝트 **LighTrip AI** 레포지토리입니다.

## 👨‍💻 Developer
|<img src="https://avatars.githubusercontent.com/u/166575866?v=4" width="150" height="150"/>|
|:-:|
|Yoonsung Jung<br/>[@coouir](https://github.com/coouir)|

---

## 🤖 AI Tech Stack

### 🧩 Core Features

#### 1. Image → Draft Generation
- Model: Gemma 4 E2B (GGUF)
- Description:  
  사용자 이미지와 프롬프트를 기반으로 한국어 블로그 스타일 초안(2–3줄) 생성

#### 2. Category Classification
- Model: TF-IDF + Naive Bayes (Baseline)  
- Description:  
  Gemma4 모델이 생성한 초안을 입력으로 받아  
  카테고리(카페, 식당, 술집, 공원, 운동 등) 자동 분류

### ⚙️ Framework & Libraries
- PyTorch  
- HuggingFace Transformers  
- llama-cpp-python (GGUF inference)  
- scikit-learn 
- NumPy  

### 🚀 Serving
- FastAPI (AI inference API)

---

### 🌐 Git-flow 전략 (Git-flow Strategy)

- **`main`**: 최종적으로 사용자에게 배포되는 가장 안정적인 버전 브랜치
- **`develop`**: 다음 출시 버전을 개발하는 중심 브랜치. 기능 개발 완료 후 `feature` 브랜치들이 병합
- **`feature`**: 기능 개발용 브랜치. `develop`에서 분기하여 작업

### 📌 브랜치 규칙 및 네이밍 (Branch Rules & Naming)

1. 모든 기능 개발은 **feature** 브랜치에서 시작
2. 작업 시작 전, 항상 최신 `develop` 내용 받아오기 (`git pull origin develop`)
3. 작업 완료 후, `develop`으로 Pull Request(PR) 생성
4. PR에 Reviewer(멘션) 지정 이후 머지

**브랜치 이름 형식:**  
feature/이슈번호-기능명

- 예시: `feature/1-login`

### 🎯 커밋 컨벤션 (Commit Convention)

- **주의 사항:**
- `type`은 소문자만 사용 (feat, fix, refactor, docs, style, test, chore)
- `subject`는 **모두 현재형 동사**

#### 📋 타입 목록

| type                | 설명                                  |
| :------------------ | :------------------------------------ |
| `start`             | 새로운 프로젝트를 시작할 때           |
| `feat`              | 새로운 기능을 추가할 때               |
| `fix`               | 버그를 수정할 때                      |
| `design`            | CSS 등 사용자 UI 디자인을 변경할 때   |
| `refactor`          | 기능 변경 없이 코드를 리팩토링할 때   |
| `settings`          | 설정 파일을 변경할 때                 |
| `comment`           | 필요한 주석을 추가하거나 변경할 때    |
| `dependency/Plugin` | 의존성/플러그인을 추가할 때           |
| `docs`              | README.md 등 문서를 수정할 때         |
| `merge`             | 브랜치를 병합할 때                    |
| `deploy`            | 빌드 및 배포 관련 작업을 할 때        |
| `rename`            | 파일 혹은 폴더명을 수정하거나 옮길 때 |
| `remove`            | 파일을 삭제하는 작업만 수행했을 때    |
| `revert`            | 이전 버전으로 롤백할 때               |

```bash
#### ✨ 예시
feat: 로그인 기능 추가
fix: 로그인 버그 수정
refactor: 로그인 로직 리팩토링
```
