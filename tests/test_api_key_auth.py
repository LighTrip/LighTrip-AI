from __future__ import annotations

import asyncio
import importlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple


API_KEY_HEADER = "X-API-Key"
PIPELINE_GENERATE_PATH = "/pipeline/generate"
TEST_API_KEY = "test-internal-api-key"
UNAUTHORIZED_RESPONSE = {"detail": "Unauthorized"}
REQUIRED_ENV = {
    "INTERNAL_API_KEY": TEST_API_KEY,
    "GEMMA_MODEL_PATH": "models/gemma-4-E2B-it-Q4_K_S.gguf",
    "GEMMA_MMPROJ_PATH": "models/mmproj-F16.gguf",
    "GEMMA_PROMPT_PATH": "configs/draft_prompt_boundary_v2.txt",
    "GEMMA_N_CTX": "1024",
    "GEMMA_MAX_TOKENS": "128",
    "GEMMA_TEMPERATURE": "0.2",
    "GEMMA_TOP_P": "0.9",
    "GEMMA_TOP_K": "40",
    "GEMMA_REPEAT_PENALTY": "1.1",
    "GEMMA_STOP_TOKENS": "<end_of_turn>",
    "GEMMA_N_GPU_LAYERS": "0",
    "GEMMA_MAIN_GPU": "0",
    "GEMMA_OFFLOAD_KQV": "false",
    "GEMMA_MMPROJ_USE_GPU": "false",
    "CATEGORY_ARTIFACT_PATH": (
        "experiments/category_classifier/artifacts/places365_2_manual_full_calibrated/"
        "calibrated_linear_svm_tfidf.joblib"
    ),
    "CATEGORY_UNKNOWN_LABEL": "기타",
}


def configure_required_env(monkeypatch) -> None:
    for name, value in REQUIRED_ENV.items():
        monkeypatch.setenv(name, value)


def load_test_app(monkeypatch):
    configure_required_env(monkeypatch)

    from app.config.settings import get_settings

    get_settings.cache_clear()
    import app.main as main_module

    return importlib.reload(main_module).app


async def asgi_request(
    app: Any,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, bytes]:
    request_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": request_headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "root_path": "",
    }
    events: List[Dict[str, Any]] = [
        {"type": "http.request", "body": b"", "more_body": False}
    ]
    sent: List[Dict[str, Any]] = []

    async def receive():
        if events:
            return events.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    status = next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, body


def request(
    app: Any,
    method: str,
    path: str,
    headers: Optional[Dict[str, str]] = None,
) -> Tuple[int, bytes]:
    return asyncio.run(asgi_request(app, method, path, headers))


def test_missing_api_key_is_rejected_and_logged(monkeypatch, caplog) -> None:
    app = load_test_app(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="app.security.api_key"):
        status_code, body = request(app, "POST", PIPELINE_GENERATE_PATH)

    assert status_code == 401
    assert json.loads(body) == UNAUTHORIZED_RESPONSE
    assert "Unauthorized AI API request blocked" in caplog.text


def test_wrong_api_key_is_rejected(monkeypatch) -> None:
    app = load_test_app(monkeypatch)

    status_code, body = request(
        app,
        "POST",
        PIPELINE_GENERATE_PATH,
        headers={API_KEY_HEADER: "wrong-key"},
    )

    assert status_code == 401
    assert json.loads(body) == UNAUTHORIZED_RESPONSE


def test_valid_api_key_reaches_protected_route(monkeypatch) -> None:
    app = load_test_app(monkeypatch)

    status_code, _ = request(
        app,
        "POST",
        PIPELINE_GENERATE_PATH,
        headers={API_KEY_HEADER: TEST_API_KEY},
    )

    assert status_code == 422


def test_health_is_public(monkeypatch) -> None:
    app = load_test_app(monkeypatch)

    status_code, body = request(app, "GET", "/health")

    assert status_code == 200
    assert json.loads(body)["status"] == "ok"


def test_fastapi_docs_are_disabled(monkeypatch) -> None:
    app = load_test_app(monkeypatch)
    headers = {API_KEY_HEADER: TEST_API_KEY}

    assert request(app, "GET", "/docs", headers=headers)[0] == 404
    assert request(app, "GET", "/redoc", headers=headers)[0] == 404
    assert request(app, "GET", "/openapi.json", headers=headers)[0] == 404
