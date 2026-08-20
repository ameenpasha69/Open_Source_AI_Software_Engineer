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


_APPLY_PATCH_ACTION = (
    '{"thought": "fix entrypoint", "action": {"tool": "apply_patch", '
    '"input": {"path": "main.py", "old_content": "def entrypoint():\\n    pass", '
    '"new_content": "def entrypoint():\\n    return 1"}}, "finish": null}'
)
_FINISH_AFTER_PATCH = (
    '{"thought": "done", "action": null, "finish": {"answer": "fixed it", "root_cause": "did nothing"}}'
)


async def test_run_agent_with_patch_reports_modified_files_and_verification_status(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["modified_files"] == ["main.py"]
    assert body["verification_status"] == "unverified"


async def test_get_agent_run_diff_returns_per_file_diffs(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        diff_resp = await http.get(f"/api/agent/{run_id}/diff")

    assert diff_resp.status_code == 200
    body = diff_resp.json()
    assert body["verification_status"] == "unverified"
    assert len(body["modified_files"]) == 1
    assert body["modified_files"][0]["path"] == "main.py"
    assert "+    return 1" in body["modified_files"][0]["diff"]


async def test_get_agent_run_diff_empty_when_no_modifications(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        diff_resp = await http.get(f"/api/agent/{run_id}/diff")

    assert diff_resp.status_code == 200
    body = diff_resp.json()
    assert body["verification_status"] == "not_applicable"
    assert body["modified_files"] == []


async def test_get_diff_for_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist/diff")
    assert resp.status_code == 404


_RUN_TESTS_ACTION = '{"thought": "check the fix", "action": {"tool": "run_tests", "input": {}}, "finish": null}'
_FINISH_VERIFIED = (
    '{"thought": "tests pass", "action": null, '
    '"finish": {"answer": "fixed and verified", "root_cause": "used - instead of +"}}'
)


async def test_run_agent_end_to_end_with_verified_fix(client, tmp_path):
    calc_repo = tmp_path / "calc_repo"
    calc_repo.mkdir()
    (calc_repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (calc_repo / "test_calc.py").write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    )
    repository_id = await _index(client, calc_repo)

    apply_patch_action = (
        '{"thought": "fix add", "action": {"tool": "apply_patch", '
        '"input": {"path": "calc.py", "old_content": "return a - b", "new_content": "return a + b"}}, '
        '"finish": null}'
    )
    _use_scripted_llm([_PLAN_RESPONSE, apply_patch_action, _RUN_TESTS_ACTION, _FINISH_VERIFIED])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/agent/run", json={"repository_id": repository_id, "task": "add(a, b) returns the wrong value"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "done"
    assert body["modified_files"] == ["calc.py"]
    assert body["verification_status"] == "verified"
    assert (calc_repo / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
