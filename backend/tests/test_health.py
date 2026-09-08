from app.config.settings import Settings, get_settings
from app.llm.factory import get_llm_provider
from app.main import app
from httpx import ASGITransport, AsyncClient


async def test_health_endpoint_reports_ok_when_llm_reachable(fake_llm_provider):
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm_provider
    # This test builds its own transport rather than using the client
    # fixture, so it has to opt out of auth the same way the fixture does.
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, auth_enabled=False
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm"]["reachable"] is True
    assert body["app_name"] == get_settings().app_name


async def test_health_endpoint_reports_degraded_when_llm_unreachable():
    from tests.conftest import FakeLLMProvider

    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(reachable=False)
    app.dependency_overrides[get_settings] = lambda: Settings(
        _env_file=None, auth_enabled=False
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"


def test_frontend_origins_splits_comma_separated_list():
    """Both localhost and the LAN address have to be allowed at once.

    Reaching the UI from a phone means the browser sends the LAN origin;
    keeping localhost working means the desktop's origin must survive too.
    A single-origin setting forces a choice between them.
    """
    settings = Settings(frontend_origin="http://localhost:3000,http://192.168.0.6:3000")
    assert settings.frontend_origins == [
        "http://localhost:3000",
        "http://192.168.0.6:3000",
    ]


def test_frontend_origins_tolerates_spacing_and_trailing_commas():
    settings = Settings(frontend_origin=" http://a:3000 , http://b:3000 ,")
    assert settings.frontend_origins == ["http://a:3000", "http://b:3000"]


def test_frontend_origins_single_value_is_unchanged():
    # Passed explicitly rather than relying on the class default: Settings
    # reads .env, so the default here is whatever this machine is configured
    # with, not the value in the source.
    settings = Settings(frontend_origin="http://localhost:3000")
    assert settings.frontend_origins == ["http://localhost:3000"]
