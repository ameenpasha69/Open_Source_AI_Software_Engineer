import pytest
from app.agents.background import get_background_agent_runner
from app.agents.runner import AgentRunner, SessionNotFoundError
from app.agents.titling import generate_session_title
from app.api.routes.sessions import _title_from, _title_task_key
from app.config.settings import get_settings
from app.database.models import AgentRun, ChatMessage, ChatSession
from app.database.session import get_db_session
from app.llm.base import LLMResponse, Message
from app.llm.factory import get_llm_provider, get_model_manager
from app.main import app
from app.retrieval.indexer import RepositoryIndexer
from app.tools.base import ToolRegistry
from app.tools.code_search_tools import FindSymbolTool
from app.tools.file_tools import ReadFileTool
from httpx import AsyncClient
from sqlalchemy import select

from tests.conftest import FakeLLMProvider
from tests.test_models_api import FakeModelManager

_PLAN = '{"steps": ["Look at the code"]}'
_FINISH = '{"thought": "done", "action": null, "finish": {"answer": "entrypoint lives in main.py", "root_cause": "typo"}}'
_AI_TITLE = "Entrypoint location lookup"


class TitleAwareLLM(FakeLLMProvider):
    """Answers the session-titling prompt with a fixed response, from its own
    slot rather than the scripted queue.

    The titling call runs in a background task concurrently with the run's
    own plan/loop calls — both would otherwise be popping from the same
    `_responses` list, and which one gets which reply would depend on
    asyncio's scheduling order rather than the test's intent.
    """

    def __init__(self, *args, title: str = _AI_TITLE, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title

    async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        if messages and messages[0].role == "system" and "short title" in messages[0].content:
            return LLMResponse(
                content=self.title, model="fake-model", prompt_tokens=8, completion_tokens=4, latency_ms=1.0
            )
        return await super().generate(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)


@pytest.fixture
def sessions_client(client, sample_repo):
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager()
    app.dependency_overrides[get_llm_provider] = lambda: TitleAwareLLM(responses=[_PLAN, _FINISH])
    yield client
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)


def db(client):
    return next(app.dependency_overrides[get_db_session]())


async def make_repo(http, sample_repo) -> str:
    resp = await http.post("/api/repositories/index", json={"path": str(sample_repo)})
    return resp.json()["repository_id"]


# --- session lifecycle -------------------------------------------------------


async def test_create_session_returns_it_with_its_repository(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        resp = await http.post("/api/sessions", json={"repository_id": repository_id})

    assert resp.status_code == 201
    body = resp.json()
    assert body["repository_id"] == repository_id
    assert body["repository_name"] == "sample_repo"
    assert body["title"] == "New session"
    assert body["message_count"] == 0


async def test_creating_a_session_for_an_unknown_repository_is_a_404(sessions_client):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        resp = await http.post("/api/sessions", json={"repository_id": "nope"})
    assert resp.status_code == 404


async def test_sessions_can_be_filtered_by_repository(sessions_client, sample_repo, tmp_path):
    other = tmp_path / "other_repo"
    other.mkdir()
    (other / "b.py").write_text("def b():\n    pass\n")

    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repo_a = await make_repo(http, sample_repo)
        repo_b = (await http.post("/api/repositories/index", json={"path": str(other)})).json()["repository_id"]
        await http.post("/api/sessions", json={"repository_id": repo_a})
        await http.post("/api/sessions", json={"repository_id": repo_b})

        listed = (await http.get("/api/sessions", params={"repository_id": repo_a})).json()
        all_sessions = (await http.get("/api/sessions")).json()

    assert len(listed) == 1
    assert listed[0]["repository_id"] == repo_a
    assert len(all_sessions) == 2


async def test_a_session_can_be_renamed(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        resp = await http.patch(f"/api/sessions/{session_id}", json={"title": "Palindrome work"})

    assert resp.status_code == 200
    assert resp.json()["title"] == "Palindrome work"


async def test_deleting_a_session_removes_the_transcript_but_keeps_the_runs(
    sessions_client, sample_repo
):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)

        resp = await http.delete(f"/api/sessions/{session_id}")
        missing = await http.get(f"/api/sessions/{session_id}")

    assert resp.status_code == 204
    assert missing.status_code == 404

    session = db(sessions_client)
    # The run is the audit trail of what the agent did to the repository —
    # deleting a conversation must not destroy it.
    run = session.get(AgentRun, run_id)
    assert run is not None
    assert run.session_id is None
    assert session.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)).all() == []


# --- messages ----------------------------------------------------------------


async def wait_for(run_id: str) -> None:
    task = get_background_agent_runner().get_task(run_id)
    if task is not None:
        await task


async def wait_for_title(session_id: str) -> None:
    task = get_background_agent_runner().get_task(_title_task_key(session_id))
    if task is not None:
        await task


async def test_posting_a_message_records_the_turn_and_starts_a_run(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        resp = await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})

    assert resp.status_code == 201
    body = resp.json()
    assert body["message"]["role"] == "user"
    assert body["message"]["content"] == "where is entrypoint?"
    assert body["run"]["status"] == "running"
    assert body["run"]["session_id"] == session_id


async def test_the_first_message_is_named_by_the_model(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        await wait_for_title(session_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()

    # Not a truncation of the message — the model's own summary.
    assert detail["title"] == _AI_TITLE


async def test_a_user_rename_survives_a_late_ai_title(sessions_client, sample_repo):
    # The title task only overwrites the placeholder it saw when it started —
    # if the user has already renamed the session by the time it lands, that
    # rename is the one that should stick.
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]

        await http.patch(f"/api/sessions/{session_id}", json={"title": "My own name"})
        await wait_for(run_id)
        await wait_for_title(session_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()

    assert detail["title"] == "My own name"


async def test_a_failed_ai_title_leaves_the_fallback_in_place(client, sample_repo):
    class NeverAnswersLLM(FakeLLMProvider):
        async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
            if messages and messages[0].role == "system" and "short title" in messages[0].content:
                raise RuntimeError("model backend is down")
            return await super().generate(messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode)

    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager()
    app.dependency_overrides[get_llm_provider] = lambda: NeverAnswersLLM(responses=[_PLAN, _FINISH])
    async with AsyncClient(transport=client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        await wait_for_title(session_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)

    assert detail["title"] == "where is entrypoint?"


async def test_only_the_first_message_ever_triggers_titling(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        await wait_for_title(session_id)

        await http.patch(f"/api/sessions/{session_id}", json={"title": "Kept as-is"})
        second_run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "now also check main.py"})
        ).json()["run"]["id"]
        await wait_for(second_run_id)

    assert get_background_agent_runner().get_task(_title_task_key(session_id)) is None
    session = db(sessions_client)
    session.expire_all()
    assert session.get(ChatSession, session_id).title == "Kept as-is"


def test_title_from_truncates_long_first_lines():
    long_message = "fix the thing that is broken in the module " * 5
    title = _title_from(long_message)
    assert len(title) <= 60
    assert title.endswith("…")


def test_title_from_keeps_a_short_first_line_untouched():
    assert _title_from("where is entrypoint?") == "where is entrypoint?"


def test_title_from_uses_only_the_first_line():
    assert _title_from("fix the bug\nit crashes on empty input") == "fix the bug"


async def test_generate_session_title_strips_quotes_and_trailing_punctuation():
    llm = FakeLLMProvider(response_text='"Palindrome check refactor."')
    title = await generate_session_title(llm, "add a palindrome check")
    assert title == "Palindrome check refactor"


async def test_generate_session_title_truncates_an_overlong_reply():
    llm = FakeLLMProvider(response_text="a" * 100)
    title = await generate_session_title(llm, "task")
    assert len(title) <= 60
    assert title.endswith("…")


async def test_generate_session_title_returns_none_when_the_model_is_unreachable():
    class BrokenLLM(FakeLLMProvider):
        async def generate(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    assert await generate_session_title(BrokenLLM(), "task") is None


async def test_generate_session_title_returns_none_for_an_empty_reply():
    assert await generate_session_title(FakeLLMProvider(response_text="   "), "task") is None


async def test_generate_session_title_sends_the_users_first_message():
    llm = FakeLLMProvider(response_text="A title")
    await generate_session_title(llm, "please fix the palindrome bug")
    sent = llm.received_messages[0]
    assert sent == [
        Message(role="system", content=sent[0].content),
        Message(role="user", content="please fix the palindrome bug"),
    ]


async def test_the_run_writes_its_answer_back_as_an_assistant_turn(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()

    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant"]
    assert detail["messages"][1]["content"] == "entrypoint lives in main.py"
    assert detail["messages"][1]["run_id"] == run_id


async def test_an_assistant_turn_carries_its_run_so_the_transcript_can_expand(
    sessions_client, sample_repo
):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()

    run = detail["messages"][1]["run"]
    assert run["id"] == run_id
    assert run["status"] == "done"
    assert run["root_cause"] == "typo"


async def test_a_run_that_fails_still_produces_an_assistant_turn(client, sample_repo):
    # A transcript that silently drops failed turns reads as though they never
    # happened, and the next turn's context would be wrong.
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager()
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(responses=[_PLAN, "not json", "still not json"])
    async with AsyncClient(transport=client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "do a thing"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)

    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["content"]


async def test_posting_to_an_unknown_session_is_a_404(sessions_client):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        resp = await http.post("/api/sessions/nope/messages", json={"content": "hi"})
    assert resp.status_code == 404


# --- chat history as agent context -------------------------------------------


@pytest.fixture
def indexed_repository_id(db_session, sample_repo):
    indexer = RepositoryIndexer(
        session=db_session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000
    )
    return indexer.index(sample_repo).repository_id


@pytest.fixture
def tool_registry(db_session):
    registry = ToolRegistry()
    registry.register(FindSymbolTool(db_session))
    registry.register(ReadFileTool(db_session, max_file_size_bytes=1_000_000))
    return registry


async def test_a_run_in_a_session_sees_the_earlier_turns(
    db_session, indexed_repository_id, tool_registry
):
    chat = ChatSession(repository_id=indexed_repository_id, title="s")
    db_session.add(chat)
    db_session.flush()
    db_session.add_all(
        [
            ChatMessage(session_id=chat.id, role="user", content="what does entrypoint do?"),
            ChatMessage(session_id=chat.id, role="assistant", content="It is a no-op stub."),
        ]
    )
    db_session.commit()

    llm = FakeLLMProvider(responses=[_PLAN, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)
    await runner.run(indexed_repository_id, "now make it raise instead", session_id=chat.id)

    # The loop's prompt (second generate call) has to carry the history.
    loop_prompt = llm.received_messages[1][1].content
    assert "Earlier in this conversation" in loop_prompt
    assert "what does entrypoint do?" in loop_prompt
    assert "It is a no-op stub." in loop_prompt


async def test_a_standalone_run_has_no_conversation_block(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)
    state = await runner.run(indexed_repository_id, "where is entrypoint?")

    assert state.conversation == []
    assert "Earlier in this conversation" not in llm.received_messages[1][1].content


async def test_the_current_turn_is_not_replayed_as_history(
    db_session, indexed_repository_id, tool_registry
):
    # The task text is already the instruction; echoing it back as an earlier
    # turn would have the model treating its own prompt as settled context.
    chat = ChatSession(repository_id=indexed_repository_id, title="s")
    db_session.add(chat)
    db_session.flush()
    db_session.commit()

    llm = FakeLLMProvider(responses=[_PLAN, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)
    state = await runner.run(indexed_repository_id, "where is entrypoint?", session_id=chat.id)

    assert state.conversation == []


async def test_creating_a_run_against_an_unknown_session_raises(
    db_session, indexed_repository_id, tool_registry
):
    runner = AgentRunner(db_session, FakeLLMProvider(), tool_registry, max_iterations=5)
    with pytest.raises(SessionNotFoundError):
        runner.create_run(indexed_repository_id, "task", session_id="nope")


# --- token usage and context window ------------------------------------------


async def test_a_run_records_the_tokens_it_spent(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)
    state = await runner.run(indexed_repository_id, "where is entrypoint?")

    # FakeLLMProvider reports 10 prompt / 5 completion per call; two calls.
    assert state.usage.call_count == 2
    assert state.usage.prompt_tokens == 20
    assert state.usage.completion_tokens == 10
    assert state.usage.total_tokens == 30
    assert state.usage.peak_prompt_tokens == 10

    run = db_session.get(AgentRun, state.run_id)
    assert run.prompt_tokens == 20
    assert run.llm_call_count == 2
    assert run.model == "fake-model"


async def test_session_usage_sums_every_run_in_it(sessions_client, sample_repo):
    async with AsyncClient(transport=sessions_client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()

    assert detail["usage"]["prompt_tokens"] == 20
    assert detail["usage"]["completion_tokens"] == 10
    assert detail["usage"]["total_tokens"] == 30
    assert detail["usage"]["llm_calls"] == 2


async def test_context_window_reports_the_peak_prompt_against_the_model_limit(client, sample_repo):
    class CataloguedModelLLM(FakeLLMProvider):
        @property
        def model(self) -> str:
            return "qwen2.5-coder:7b"

    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager()
    app.dependency_overrides[get_llm_provider] = lambda: CataloguedModelLLM(responses=[_PLAN, _FINISH])
    async with AsyncClient(transport=client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        run_id = (
            await http.post(f"/api/sessions/{session_id}/messages", json={"content": "where is entrypoint?"})
        ).json()["run"]["id"]
        await wait_for(run_id)
        detail = (await http.get(f"/api/sessions/{session_id}")).json()
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)

    window = detail["context_window"]
    # Peak, not total: one call's prompt, never the sum across calls.
    assert window["used_tokens"] == 10
    assert window["limit_tokens"] == 32768  # from the catalog entry for qwen2.5-coder:7b
    assert window["percent"] == 0.0


async def test_context_window_reports_no_percentage_for_an_unknown_model(client, sample_repo):
    class UnknownModelLLM(FakeLLMProvider):
        @property
        def model(self) -> str:
            return "someones-private-finetune:v3"

    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager()
    app.dependency_overrides[get_llm_provider] = lambda: UnknownModelLLM(responses=[_PLAN, _FINISH])
    async with AsyncClient(transport=client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        detail = (await http.get(f"/api/sessions/{session_id}")).json()
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)

    # Better an honest "unknown" than a percentage of a number nobody verified.
    assert detail["context_window"]["limit_tokens"] is None
    assert detail["context_window"]["percent"] is None


async def test_session_detail_still_answers_when_the_model_backend_is_down(client, sample_repo):
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager(unreachable=True)
    app.dependency_overrides[get_llm_provider] = lambda: FakeLLMProvider(responses=[_PLAN, _FINISH])
    async with AsyncClient(transport=client, base_url="http://test") as http:
        repository_id = await make_repo(http, sample_repo)
        session_id = (await http.post("/api/sessions", json={"repository_id": repository_id})).json()["id"]
        resp = await http.get(f"/api/sessions/{session_id}")
    app.dependency_overrides.pop(get_model_manager, None)
    app.dependency_overrides.pop(get_llm_provider, None)

    assert resp.status_code == 200
    assert resp.json()["context_window"]["model"] == "fake-model"


def test_settings_default_model_is_in_the_catalog():
    # The context-window fallback above depends on it.
    from app.llm.catalog import find_catalog_entry

    assert find_catalog_entry(get_settings().llm_model) is not None
