"""Bearer-token authentication for the API.

This application runs commands and writes files inside repositories. An
unauthenticated instance reachable by anything other than the person running it
is remote code execution, so authentication is on by default and there is no
configuration that silently turns it off -- `AUTH_ENABLED=false` has to be
written by hand, and it says what it does.

If no token is configured, one is generated on first start and persisted under
the data directory, the way Jupyter does it. That keeps a fresh local install
working without a setup step while making an unauthenticated deployment
something you cannot arrive at by accident.

Three ways to present the token, in precedence order:

    Authorization: Bearer <token>
    X-API-Key: <token>
    ?token=<token>

The query parameter exists because the browser's EventSource API cannot set
headers, and the run stream is an EventSource. It is deliberately last, and
worth knowing that query strings show up in server logs and Referer headers in
a way headers do not -- prefer a header anywhere you have the choice.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path

from fastapi import Depends, HTTPException, Request, status

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

TOKEN_FILENAME = "auth_token"


def resolve_auth_token(settings: Settings) -> str | None:
    """The token this instance expects, generating and persisting one if needed.

    Returns None only when authentication is switched off.
    """
    if not settings.auth_enabled:
        return None
    if settings.api_auth_token:
        return settings.api_auth_token

    token_path = Path(settings.data_dir) / TOKEN_FILENAME
    if token_path.exists():
        existing = token_path.read_text(encoding="utf-8").strip()
        if existing:
            return existing

    token = secrets.token_urlsafe(32)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token, encoding="utf-8")
    # Owner-only where the platform honours it. On Windows this is close to a
    # no-op, which is worth knowing rather than assuming otherwise.
    try:
        token_path.chmod(0o600)
    except OSError:  # pragma: no cover - platform dependent
        pass
    return token


def token_from_request(request: Request) -> str | None:
    """Pull the presented token out of a request, if there is one."""
    header = request.headers.get("authorization")
    if header:
        scheme, _, value = header.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value.strip()
    api_key = request.headers.get("x-api-key")
    if api_key:
        return api_key.strip()
    query_token = request.query_params.get("token")
    if query_token:
        return query_token.strip()
    return None


async def require_token(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Reject a request that does not carry the configured token."""
    if not settings.auth_enabled:
        return

    expected = resolve_auth_token(settings)
    if not expected:  # pragma: no cover - resolve_auth_token always returns one
        return

    presented = token_from_request(request)
    # compare_digest rather than ==: the comparison is on a secret, and the
    # early exit of a normal string compare leaks its length and prefix through
    # timing. Cheap to do correctly.
    if presented is None or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
