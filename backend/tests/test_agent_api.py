from app.llm.factory import get_llm_provider
from app.main import app
from httpx import AsyncClient

from tests.conftest import FakeLLMProvider

_PLAN_RESPONSE = '{"steps": ["Search the codebase"]}'
_FINISH_RESPONSE = '{"thought": "found it", "action": null, "finish": {"answer": "it is in main.py", "root_cause": "n/a"}}'


async def _index(client, repo_path) -> str:
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/repositories/index", json={"path": str(repo_path)})
    assert resp.status_code == 200
    return resp.json()["repository_id"]


def _use_scripted_llm(responses: list[str]) -> None:
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(responses=responses)


async def test_run_agent_returns_completed_run_summary(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/agent/run", json={"repository_id": repository_id, "task": "Where is entrypoint defined?"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["final_answer"] == "it is in main.py"
    assert body["plan"] == ["Search the codebase"]
    assert body["iteration_count"] == 1


async def test_run_agent_unknown_repository_returns_404(client):
    _use_scripted_llm([_PLAN_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/agent/run", json={"repository_id": "nope", "task": "task"})

    assert resp.status_code == 404


async def test_get_agent_run_returns_persisted_state(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        get_resp = await http.get(f"/api/agent/{run_id}")

    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == run_id
    assert get_resp.json()["status"] == "done"


async def test_get_unknown_agent_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist")
    assert resp.status_code == 404


async def test_get_agent_run_events_returns_ordered_timeline(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        events_resp = await http.get(f"/api/agent/{run_id}/events")

    assert events_resp.status_code == 200
    event_types = [e["event_type"] for e in events_resp.json()]
    assert event_types[0] == "plan_created"
    assert "finished" in event_types


async def test_get_events_for_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist/events")
    assert resp.status_code == 404
