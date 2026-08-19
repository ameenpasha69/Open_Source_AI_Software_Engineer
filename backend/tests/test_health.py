from app.config.settings import get_settings
from app.llm.factory import get_llm_provider
from app.main import app
from httpx import ASGITransport, AsyncClient


async def test_health_endpoint_reports_ok_when_llm_reachable(fake_llm_provider):
    app.dependency_overrides[get_llm_provider] = lambda: fake_llm_provider

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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")

    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
