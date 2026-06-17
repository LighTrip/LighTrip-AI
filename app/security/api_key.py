"""Shared API key verification for internal AI server requests."""

from __future__ import annotations

import logging
import secrets

from fastapi import Request

from app.config.settings import get_settings


API_KEY_HEADER = "X-API-Key"
AUTH_EXEMPT_PATHS = frozenset({"/", "/health"})
DOCS_AUTH_EXEMPT_PATHS = frozenset({
    "/docs",
    "/docs/oauth2-redirect",
    "/openapi.json",
    "/redoc",
})
logger = logging.getLogger(__name__)


def is_auth_exempt_path(path: str, docs_enabled: bool = False) -> bool:
    if path in AUTH_EXEMPT_PATHS:
        return True
    return docs_enabled and path in DOCS_AUTH_EXEMPT_PATHS


def verify_api_key(request: Request) -> bool:
    provided_key = request.headers.get(API_KEY_HEADER)
    expected_key = get_settings().internal_api_key

    if provided_key is None or not secrets.compare_digest(
        provided_key,
        expected_key,
    ):
        client_host = request.client.host if request.client else "unknown"
        logger.warning(
            "Unauthorized AI API request blocked: path=%s client=%s",
            request.url.path,
            client_host,
        )
        return False

    return True
