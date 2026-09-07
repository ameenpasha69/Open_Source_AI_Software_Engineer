"""Authentication behaviour.

The general `client` fixture runs with auth off, so these build their own with
auth on. That split is deliberate: everything else is about indexing, tools and
the agent, and this is about the token.
"""

import secrets

import pytest
from app.auth import resolve_auth_token, token_from_request
from app.config.settings import Settings, get_settings
from app.main import app
from httpx import ASGITransport, AsyncClient

TOKEN = "test-token-value"


@pytest.fixture
def authed_client(tmp_path):
    """A client against the real app with authentication switched on."""
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data",
        auth_enabled=True, api_auth_token=TOKEN,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    yield ASGITransport(app=app)
    app.dependency_overrides.clear()


async def test_request_without_a_token_is_refused(authed_client):
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get("/api/repositories")

    assert resp.status_code == 401
    # The scheme has to be advertised for a client to know how to retry.
    assert resp.headers["www-authenticate"] == "Bearer"


async def test_request_with_the_wrong_token_is_refused(authed_client):
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get(
            "/api/repositories", headers={"Authorization": "Bearer not-the-token"}
        )

    assert resp.status_code == 401


async def test_bearer_header_is_accepted(authed_client):
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get(
            "/api/repositories", headers={"Authorization": f"Bearer {TOKEN}"}
        )

    assert resp.status_code == 200


async def test_api_key_header_is_accepted(authed_client):
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get("/api/repositories", headers={"X-API-Key": TOKEN})

    assert resp.status_code == 200


async def test_query_parameter_is_accepted_for_eventsource(authed_client):
    """EventSource cannot set headers, so the stream needs the token in the URL."""
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get(f"/api/repositories?token={TOKEN}")

    assert resp.status_code == 200


async def test_health_is_also_protected(authed_client):
    """Health reports which models are loaded, so it is not public either."""
    async with AsyncClient(transport=authed_client, base_url="http://test") as http:
        resp = await http.get("/api/health")

    assert resp.status_code == 401


async def test_auth_can_be_disabled_explicitly(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", auth_enabled=False)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            resp = await http.get("/api/repositories")
        assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_token_is_generated_and_persisted_when_unset(tmp_path):
    """A fresh install gets a token without a setup step, and keeps it."""
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", auth_enabled=True)

    first = resolve_auth_token(settings)
    assert first
    assert (tmp_path / "data" / "auth_token").read_text(encoding="utf-8").strip() == first

    # Stable across restarts, or every restart would lock the frontend out.
    assert resolve_auth_token(settings) == first


def test_configured_token_takes_precedence_over_the_generated_one(tmp_path):
    settings = Settings(
        _env_file=None, data_dir=tmp_path / "data",
        auth_enabled=True, api_auth_token="explicit",
    )
    assert resolve_auth_token(settings) == "explicit"
    assert not (tmp_path / "data" / "auth_token").exists()


def test_disabled_auth_resolves_to_no_token(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data", auth_enabled=False)
    assert resolve_auth_token(settings) is None


@pytest.mark.parametrize(
    "headers,params,expected",
    [
        ({"Authorization": "Bearer abc"}, {}, "abc"),
        ({"Authorization": "bearer abc"}, {}, "abc"),          # scheme is case-insensitive
        ({"Authorization": "Basic abc"}, {}, None),            # wrong scheme is not a token
        ({"Authorization": "Bearer "}, {}, None),              # empty value is not a token
        ({"X-API-Key": "abc"}, {}, "abc"),
        ({}, {"token": "abc"}, "abc"),
        ({}, {}, None),
    ],
)
def test_token_extraction(headers, params, expected):
    from starlette.datastructures import Headers, QueryParams

    class _Request:
        def __init__(self):
            self.headers = Headers(headers)
            self.query_params = QueryParams(params)

    assert token_from_request(_Request()) == expected


def test_generated_tokens_are_not_guessable(tmp_path):
    """Distinct installs must not share a token."""
    tokens = {
        resolve_auth_token(
            Settings(_env_file=None, data_dir=tmp_path / f"d{i}", auth_enabled=True)
        )
        for i in range(5)
    }
    assert len(tokens) == 5
    assert all(len(t) >= 32 for t in tokens)


def test_comparison_is_constant_time():
    """A plain == on a secret leaks its prefix through timing."""
    import inspect

    from app import auth

    assert "compare_digest" in inspect.getsource(auth.require_token)
    assert secrets.compare_digest("a", "a")
