# AI 서버 API Key 인증 운영 메모

## 환경변수

AI 서버는 `.env` 또는 운영 환경변수에서 `INTERNAL_API_KEY`를 읽습니다.

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

생성한 값은 코드에 하드코딩하지 말고 AI 서버와 백엔드 서버의 안전한 시크릿 저장소에만 보관합니다.

## 요청 인터페이스

백엔드는 AI 서버 호출 시 모든 보호 대상 API에 아래 헤더를 포함해야 합니다.

```http
X-API-Key: <shared-internal-api-key>
```

AI 서버는 누락되었거나 일치하지 않는 키를 `401 Unauthorized`로 거절하고 경고 로그를 남깁니다.

## 보호 정책

- 보호 대상: `/pipeline/generate`, `/get-embedding` 등 추론/임베딩 API
- 예외 대상: `/`, `/health`
- 운영 기본값: `/docs`, `/redoc`, `/openapi.json` 비활성화
- 로컬 디버깅에서 문서가 필요하면 `ENABLE_API_DOCS=true`를 임시로 설정합니다.

## HTTPS

공유 키는 평문 HTTP로 전송하면 탈취될 수 있으므로 운영에서는 반드시 HTTPS 뒤에서 AI 서버를 노출합니다.
권장 방식은 AWS 리버스 프록시, 로드밸런서, 또는 터널/VPN 구간에서 TLS를 종료하는 것입니다.

## Rotation 절차

1. 새 `INTERNAL_API_KEY`를 난수로 생성합니다.
2. 백엔드 시크릿 저장소에 새 키를 반영합니다.
3. AI 서버 환경변수 또는 `.env`에 새 키를 반영합니다.
4. 백엔드와 AI 서버를 순서대로 재시작하거나 재배포합니다.
5. 구 키로 호출한 요청이 `401`로 차단되는지 로그와 헬스체크로 확인합니다.
