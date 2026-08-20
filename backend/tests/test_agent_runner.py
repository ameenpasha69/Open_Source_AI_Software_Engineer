import json

import pytest
from app.agents.runner import AgentRunner
from app.agents.state import AgentStatus
from app.database.models import AgentEvent, AgentRun, AgentToolCall, ModifiedFile
from app.llm.exceptions import LLMConnectionError
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from app.tools.base import ToolRegistry
from app.tools.code_search_tools import FindSymbolTool
from app.tools.file_tools import ReadFileTool
from app.tools.patch_tools import ApplyPatchTool
from sqlalchemy import select

from tests.conftest import FakeLLMProvider


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
    registry.register(ApplyPatchTool(db_session))
    return registry


_PLAN_RESPONSE = '{"steps": ["Find the entrypoint function"]}'
_FIND_SYMBOL_ACTION = (
    '{"thought": "search for entrypoint", '
    '"action": {"tool": "find_symbol", "input": {"symbol": "entrypoint"}}, "finish": null}'
)
_FINISH_RESPONSE = '{"thought": "found it", "action": null, "finish": {"answer": "entrypoint is in main.py", "root_cause": null}}'


async def test_agent_finds_symbol_and_finishes(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _FIND_SYMBOL_ACTION, _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "Where is entrypoint defined?")

    assert state.status == AgentStatus.DONE
    assert state.final_answer == "entrypoint is in main.py"
    assert state.iteration == 2
    assert state.plan == ["Find the entrypoint function"]
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].tool_name == "find_symbol"
    assert state.tool_calls[0].result.success is True


async def test_agent_stops_at_max_iterations_without_finishing(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE] + [_FIND_SYMBOL_ACTION] * 3)
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=3)

    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.MAX_ITERATIONS_REACHED
    assert state.iteration == 3
    assert len(state.tool_calls) == 3


async def test_agent_retries_once_on_invalid_json_within_same_iteration(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, "not valid json", _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.DONE
    assert state.iteration == 1  # the retry happened within the same iteration


async def test_agent_skips_iteration_when_decision_unparseable_after_retries(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, "garbage 1", "garbage 2", _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.DONE
    assert state.iteration == 2  # iteration 1 was skipped (unparseable), iteration 2 finished
    assert any("invalid" in obs.lower() for obs in state.observations)


async def test_agent_run_fails_when_llm_becomes_unreachable(db_session, indexed_repository_id, tool_registry):
    class FlakyLLM(FakeLLMProvider):
        def __init__(self):
            super().__init__(responses=[_PLAN_RESPONSE])
            self._calls = 0

        async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
            self._calls += 1
            if self._calls == 1:
                return await super().generate(messages, json_mode=json_mode)
            raise LLMConnectionError("Ollama went away")

    runner = AgentRunner(db_session, FlakyLLM(), tool_registry, max_iterations=5)
    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.FAILED
    assert "LLM error" in state.error


async def test_agent_run_raises_for_unknown_repository(db_session, tool_registry, fake_llm_provider):
    runner = AgentRunner(db_session, fake_llm_provider, tool_registry, max_iterations=5)
    with pytest.raises(RepositoryNotFoundError):
        await runner.run("does-not-exist", "task")


async def test_agent_run_persists_run_events_and_tool_calls(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _FIND_SYMBOL_ACTION, _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.status == "done"
    assert run_row.final_answer == "entrypoint is in main.py"
    assert json.loads(run_row.plan_json) == ["Find the entrypoint function"]
    assert run_row.finished_at is not None

    events = db_session.scalars(select(AgentEvent).where(AgentEvent.run_id == state.run_id)).all()
    event_types = {e.event_type for e in events}
    assert {"plan_created", "iteration_started", "thought", "tool_called", "tool_completed", "finished"} <= event_types

    tool_calls = db_session.scalars(select(AgentToolCall).where(AgentToolCall.run_id == state.run_id)).all()
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "find_symbol"
    assert tool_calls[0].success is True
    assert json.loads(tool_calls[0].input_json) == {"symbol": "entrypoint"}


_APPLY_PATCH_ACTION = (
    '{"thought": "fix entrypoint", "action": {"tool": "apply_patch", '
    '"input": {"path": "main.py", "old_content": "def entrypoint():\\n    pass", '
    '"new_content": "def entrypoint():\\n    return 1"}}, "finish": null}'
)
_FINISH_AFTER_PATCH = (
    '{"thought": "done", "action": null, '
    '"finish": {"answer": "fixed entrypoint to return 1", "root_cause": "it did nothing before"}}'
)


async def test_agent_applies_patch_and_tracks_modified_files(
    db_session, indexed_repository_id, tool_registry, sample_repo
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "entrypoint does nothing, it should return 1")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["main.py"]
    assert state.verification_status == "unverified"
    assert (sample_repo / "main.py").read_text() == "def entrypoint():\n    return 1\n"


async def test_agent_run_persists_modified_files(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _APPLY_PATCH_ACTION, _FINISH_AFTER_PATCH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.verification_status == "unverified"

    modified = db_session.scalars(select(ModifiedFile).where(ModifiedFile.run_id == state.run_id)).all()
    assert len(modified) == 1
    assert modified[0].relative_path == "main.py"
    assert modified[0].lines_added == 1
    assert modified[0].lines_removed == 1
    assert "+    return 1" in modified[0].diff


async def test_agent_run_without_modifications_has_not_applicable_verification(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _FIND_SYMBOL_ACTION, _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    assert state.modified_files == []
    assert state.verification_status == "not_applicable"
    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.verification_status == "not_applicable"


async def test_agent_run_persists_failed_status_on_unexpected_error(
    db_session, indexed_repository_id, tool_registry, monkeypatch
):
    def _boom(*args, **kwargs):
        raise RuntimeError("something genuinely unexpected")

    monkeypatch.setattr("app.agents.runner.format_observation", _boom)

    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _FIND_SYMBOL_ACTION])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.FAILED
    assert "Unexpected agent error" in state.error
    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.status == "failed"
