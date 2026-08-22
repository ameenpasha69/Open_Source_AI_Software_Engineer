import asyncio
import json

from app.llm.factory import get_llm_provider
from app.main import app
from httpx import AsyncClient
from sqlalchemy import select

from tests.conftest import FakeLLMProvider, wait_for_agent_run

_PLAN_RESPONSE = '{"steps": ["Search the codebase"]}'
_FINISH_RESPONSE = '{"thought": "found it", "action": null, "finish": {"answer": "it is in main.py", "root_cause": "n/a"}}'


async def _index(client, repo_path) -> str:
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/repositories/index", json={"path": str(repo_path)})
    assert resp.status_code == 200
    return resp.json()["repository_id"]


def _use_scripted_llm(responses: list[str]) -> None:
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(responses=responses)


async def _run_agent_and_wait(client, repository_id: str, task: str) -> dict:
    """POST /api/agent/run, wait for the background task to finish, and
    return the final GET /api/agent/{run_id} body."""
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": task})
        assert resp.status_code == 200
        run_id = resp.json()["id"]

        await wait_for_agent_run(run_id)

        final = await http.get(f"/api/agent/{run_id}")
    return final.json()


async def test_run_agent_returns_immediately_with_running_status(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/agent/run", json={"repository_id": repository_id, "task": "Where is entrypoint defined?"}
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    await wait_for_agent_run(resp.json()["id"])  # let it finish before the test ends


async def test_run_agent_completes_in_the_background(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    body = await _run_agent_and_wait(client, repository_id, "Where is entrypoint defined?")

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

    body = await _run_agent_and_wait(client, repository_id, "task")

    assert body["status"] == "done"


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
        await wait_for_agent_run(run_id)

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

    body = await _run_agent_and_wait(client, repository_id, "task")

    assert body["modified_files"] == ["main.py"]
    assert body["verification_status"] == "unverified"


async def test_get_agent_run_diff_returns_per_file_diffs(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]
        await wait_for_agent_run(run_id)

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
        await wait_for_agent_run(run_id)

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

    body = await _run_agent_and_wait(client, repository_id, "add(a, b) returns the wrong value")

    assert body["status"] == "done"
    assert body["modified_files"] == ["calc.py"]
    assert body["verification_status"] == "verified"
    assert (calc_repo / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


async def test_cancel_agent_run_stops_it_and_marks_cancelled(client, sample_repo):
    repository_id = await _index(client, sample_repo)

    class NeverRespondingLLM(FakeLLMProvider):
        async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
            await asyncio.Event().wait()  # hangs until the task is cancelled

    app.dependency_overrides[get_llm_provider] = lambda: NeverRespondingLLM()

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        cancel_resp = await http.post(f"/api/agent/{run_id}/cancel")

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"


async def test_cancel_already_finished_run_is_a_noop(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    body = await _run_agent_and_wait(client, repository_id, "task")
    assert body["status"] == "done"

    async with AsyncClient(transport=client, base_url="http://test") as http:
        cancel_resp = await http.post(f"/api/agent/{body['id']}/cancel")

    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "done"  # unchanged, not overwritten with "cancelled"


async def test_cancel_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/agent/does-not-exist/cancel")
    assert resp.status_code == 404


async def test_stream_agent_run_emits_events_and_closes_on_completion(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]

        async with http.stream("GET", f"/api/agent/{run_id}/stream") as stream_resp:
            assert stream_resp.status_code == 200
            assert "text/event-stream" in stream_resp.headers["content-type"]

            event_types = []
            closed = False
            async for line in stream_resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                if "event_type" in payload:
                    event_types.append(payload["event_type"])
                elif "status" in payload:  # the terminal run_completed marker
                    closed = True
                    break

    assert closed
    assert "plan_created" in event_types
    assert "finished" in event_types


async def test_stream_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist/stream")
    assert resp.status_code == 404


async def test_get_agent_run_tests_returns_structured_results(client, tmp_path):
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
        run_resp = await http.post(
            "/api/agent/run", json={"repository_id": repository_id, "task": "task"}
        )
        run_id = run_resp.json()["id"]
        await wait_for_agent_run(run_id)

        tests_resp = await http.get(f"/api/agent/{run_id}/tests")

    assert tests_resp.status_code == 200
    body = tests_resp.json()
    assert len(body) == 1
    assert body[0]["passed"] is True
    assert body[0]["scope"] == "full"
    assert body[0]["command"] == "python3 -m pytest"


async def test_get_agent_run_tests_for_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist/tests")
    assert resp.status_code == 404


async def test_get_agent_run_tool_calls_returns_full_detail(client, sample_repo):
    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "task"})
        run_id = run_resp.json()["id"]
        await wait_for_agent_run(run_id)

        calls_resp = await http.get(f"/api/agent/{run_id}/tool-calls")

    assert calls_resp.status_code == 200
    body = calls_resp.json()
    assert len(body) == 1
    assert body[0]["tool_name"] == "apply_patch"
    assert body[0]["success"] is True
    assert body[0]["output"]["path"] == "main.py"
    assert "+    return 1" in body[0]["output"]["diff"]


async def test_get_agent_run_tool_calls_for_unknown_run_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/agent/does-not-exist/tool-calls")
    assert resp.status_code == 404


# --- re-indexing after a write --------------------------------------------
#
# Regression coverage: a file the agent (or an earlier run) had just created
# was invisible to that same agent's own search_code — nothing re-indexed
# after a write. Observed live, repeatedly: the model correctly searched,
# got nothing back (because the file was never indexed, not because it was
# missing), and confidently reported the file "does not exist."

_CREATE_FILE_ACTION = (
    '{"thought": "add a helper", "action": {"tool": "create_file", '
    '"input": {"path": "helper.py", "content": "def helper():\\n    return 42\\n"}}, "finish": null}'
)


async def test_a_file_created_by_the_agent_is_indexed_without_a_manual_reindex(client, sample_repo):
    from app.database.models import IndexedFile
    from app.database.session import get_db_session

    repository_id = await _index(client, sample_repo)
    _use_scripted_llm([_PLAN_RESPONSE, _CREATE_FILE_ACTION, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "add a helper"})
        await wait_for_agent_run(run_resp.json()["id"])

    session = next(app.dependency_overrides[get_db_session]())
    indexed = session.scalars(
        select(IndexedFile).where(
            IndexedFile.repository_id == repository_id, IndexedFile.relative_path == "helper.py"
        )
    ).first()
    assert indexed is not None
    assert indexed.chunk_count > 0


async def test_a_file_deleted_by_the_agent_is_removed_from_the_index(client, sample_repo):
    from app.database.models import IndexedFile
    from app.database.session import get_db_session

    repository_id = await _index(client, sample_repo)
    delete_action = (
        '{"thought": "remove it", "action": {"tool": "delete_file", "input": {"path": "main.py"}}, "finish": null}'
    )
    _use_scripted_llm([_PLAN_RESPONSE, delete_action, _FINISH_RESPONSE])

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "remove main.py"})
        await wait_for_agent_run(run_resp.json()["id"])

    session = next(app.dependency_overrides[get_db_session]())
    indexed = session.scalars(
        select(IndexedFile).where(
            IndexedFile.repository_id == repository_id, IndexedFile.relative_path == "main.py"
        )
    ).first()
    assert indexed is None


async def test_search_code_finds_a_file_the_same_run_just_created(client, sample_repo):
    """The actual end-to-end point of re-indexing: a later tool call in the
    *same* run can find what an earlier tool call in that run just wrote."""
    find_after_create_action = (
        '{"thought": "confirm it is there", '
        '"action": {"tool": "search_code", "input": {"query": "def helper(): return 42"}}, "finish": null}'
    )
    _use_scripted_llm([_PLAN_RESPONSE, _CREATE_FILE_ACTION, find_after_create_action, _FINISH_RESPONSE])
    repository_id = await _index(client, sample_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        run_resp = await http.post("/api/agent/run", json={"repository_id": repository_id, "task": "add a helper"})
        run_id = run_resp.json()["id"]
        await wait_for_agent_run(run_id)
        calls_resp = await http.get(f"/api/agent/{run_id}/tool-calls")

    search_call = next(c for c in calls_resp.json() if c["tool_name"] == "search_code")
    assert search_call["success"] is True
    results = search_call["output"]["results"]
    assert any(r["file_path"] == "helper.py" for r in results)


async def test_reindex_failure_after_a_write_does_not_fail_the_run(client, sample_repo):
    """Best-effort: an embedding backend that fails mid-run must degrade
    search, not the write that already succeeded."""
    from app.embeddings.base import EmbeddingProvider
    from app.embeddings.factory import get_embedding_provider

    class BrokenEmbeddingProvider(EmbeddingProvider):
        @property
        def model(self) -> str:
            return "broken"

        async def embed_text(self, text: str) -> list[float]:
            raise ConnectionError("embedding backend is down")

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise ConnectionError("embedding backend is down")

        async def health_check(self) -> bool:
            return False

    repository_id = await _index(client, sample_repo)
    app.dependency_overrides[get_embedding_provider] = lambda: BrokenEmbeddingProvider()
    _use_scripted_llm([_PLAN_RESPONSE, _CREATE_FILE_ACTION, _FINISH_RESPONSE])

    result = await _run_agent_and_wait(client, repository_id, "add a helper")

    assert result["status"] == "done"
    assert result["modified_files"] == ["helper.py"]
