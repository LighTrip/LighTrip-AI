# Gemma 카테고리 경계 프롬프트 적용 문서

## 목적

Gemma가 이미지 기반 블로그 초안을 생성할 때 이미지와 충돌하는 카테고리 단서를 줄이기 위해, 카테고리 경계 규칙을 반영한 서비스 프롬프트를 별도로 운영한다.

중점 개선 대상은 다음 경계다.

- 카페 / 술집
- 공원 / 운동
- 문화 / 쇼핑
- 식당 / 카페 / 술집처럼 테이블과 잔 단서가 겹치는 장면

## 보안 정책

서비스 프롬프트 원문은 보안 정책상 repository에 커밋하지 않는다.

프롬프트 파일은 배포 환경 또는 보안 관리 영역에만 저장하고, 애플리케이션은 `GEMMA_PROMPT_PATH` 환경변수로 해당 파일 경로를 참조한다.

기존 `configs/draft_prompt.txt`는 baseline 보존을 위해 변경하지 않는다.

## 적용 방식

배포 환경에서 보안 관리되는 프롬프트 파일을 준비한 뒤 `GEMMA_PROMPT_PATH`를 해당 경로로 지정한다.

예시:

```bash
export GEMMA_PROMPT_PATH=/secure/path/draft_prompt_boundary_v2.txt
```

로컬 검증 시에는 repository에 커밋되지 않는 ignored 파일을 사용할 수 있다.

```bash
export GEMMA_PROMPT_PATH=configs/draft_prompt_boundary_v2.txt
```

## 검증 요약

로컬에서 카테고리 경계 프롬프트를 작성한 뒤 다음 항목을 확인했다.

- `{user_prompt}` 치환 렌더링 정상 동작 확인
- 카테고리 경계 규칙 포함 여부 확인
- smoke 샘플 생성 결과 검토
- 카페/술집, 공원/운동, 문화/쇼핑 충돌 표현 감소 확인

프롬프트 원문과 생성 결과 파일은 보안 및 실험 산출물 관리 정책에 따라 PR에 포함하지 않는다.

## PR 범위

이 PR은 프롬프트 원문을 추가하지 않고, 보안 정책에 맞는 적용 방식과 검증 절차만 문서화한다.

실제 프롬프트 반영은 배포 환경의 보안 관리 파일 교체 또는 `GEMMA_PROMPT_PATH` 설정 변경으로 진행한다.
