import json

import pytest
from app.agents.runner import AgentRunner
from app.agents.state import AgentStatus
from app.database.models import (
    AgentEvent,
    AgentRun,
    AgentToolCall,
    ModifiedFile,
    Repository,
    TestRun,
)
from app.llm.exceptions import LLMConnectionError
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from app.tools.base import ToolRegistry
from app.tools.code_search_tools import FindSymbolTool
from app.tools.execution_tools import RunCommandTool, RunTestsTool
from app.tools.file_tools import ReadFileTool
from app.tools.patch_tools import ApplyPatchTool, CreateFileTool, DeleteFileTool
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
    registry.register(CreateFileTool(db_session))
    registry.register(DeleteFileTool(db_session))
    registry.register(RunTestsTool(db_session, timeout_seconds=30.0))
    registry.register(RunCommandTool(db_session))
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
    # Distinct symbols per iteration on purpose: three *identical* calls would
    # now trip the redundancy guard and stop the run as NO_PROGRESS instead,
    # which is a different stop condition than the one under test here.
    searches = [
        f'{{"thought": "look", "action": {{"tool": "find_symbol", "input": {{"symbol": "{symbol}"}}}}, "finish": null}}'
        for symbol in ("entrypoint", "helper", "main")
    ]
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, *searches])
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
    # The specific parse error is persisted into observations (not just a
    # generic "invalid output" note) so a systematic mistake has a chance of
    # being visible to the model on the next iteration too.
    assert any("rejected" in obs.lower() and "Expecting value" in obs for obs in state.observations)


async def test_agent_runs_run_command_when_llm_sends_bare_argv_list(
    db_session, indexed_repository_id, tool_registry
):
    """Reproduces a real observed failure verbatim: qwen2.5-coder:7b sent
    exactly this JSON for run_command — "input" as the bare argv list
    instead of {"command": [...]} — and repeated the identical mistake
    across many iterations, burning the whole run without ever calling
    run_command successfully. AgentAction's coercion should make this
    resolve in a single iteration instead of ever hitting the parse-retry
    path at all."""
    bare_list_decision = (
        '{"thought": "clean up untracked files", '
        '"action": {"tool": "run_command", "input": ["git", "clean", "-fdx"]}, "finish": null}'
    )
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, bare_list_decision, _FINISH_RESPONSE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    assert state.status == AgentStatus.DONE
    assert state.iteration == 2  # no wasted "invalid_decision" iteration
    assert not any("rejected" in obs.lower() for obs in state.observations)
    [tool_call] = state.tool_calls
    assert tool_call.tool_name == "run_command"
    assert tool_call.result.success is True
    assert tool_call.input == {"command": ["git", "clean", "-fdx"]}


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


@pytest.fixture
def calc_repo(tmp_path):
    repo = tmp_path / "calc_repo"
    repo.mkdir()
    (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (repo / "test_calc.py").write_text("from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    return repo


@pytest.fixture
def calc_repository_id(db_session, calc_repo):
    indexer = RepositoryIndexer(
        session=db_session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000
    )
    result = indexer.index(calc_repo)
    repository = db_session.get(Repository, result.repository_id)
    repository.test_command_json = json.dumps(["python3", "-m", "pytest"])
    db_session.commit()
    return result.repository_id


def _apply_patch_action(old_content: str, new_content: str) -> str:
    payload = {
        "thought": "applying fix",
        "action": {
            "tool": "apply_patch",
            "input": {"path": "calc.py", "old_content": old_content, "new_content": new_content},
        },
        "finish": None,
    }
    return json.dumps(payload)


_RUN_TESTS_ACTION = json.dumps(
    {"thought": "check the fix", "action": {"tool": "run_tests", "input": {}}, "finish": None}
)
_FINISH_VERIFIED = json.dumps(
    {"thought": "tests pass", "action": None, "finish": {"answer": "fixed and verified", "root_cause": "used - instead of +"}}
)


async def test_agent_verifies_a_correct_fix_with_run_tests(db_session, calc_repository_id, tool_registry):
    llm = FakeLLMProvider(
        responses=[
            _PLAN_RESPONSE,
            _apply_patch_action("return a - b", "return a + b"),
            _RUN_TESTS_ACTION,
            _FINISH_VERIFIED,
        ]
    )
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=6)

    state = await runner.run(calc_repository_id, "add(a, b) returns the wrong value")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["calc.py"]
    assert len(state.test_results) == 1
    assert state.test_results[0].passed is True
    assert state.verification_status == "verified"


async def test_agent_self_corrects_after_a_failing_test_run(db_session, calc_repository_id, tool_registry):
    """The core self-correction behavior: a wrong first patch, a failing
    run_tests, a corrected second patch, a passing run_tests — driven
    entirely by the ordinary loop, no special-casing for "retry after
    failure" anywhere in AgentRunner."""
    llm = FakeLLMProvider(
        responses=[
            _PLAN_RESPONSE,
            _apply_patch_action("return a - b", "return a * b"),  # still wrong
            _RUN_TESTS_ACTION,  # fails
            _apply_patch_action("return a * b", "return a + b"),  # corrected
            _RUN_TESTS_ACTION,  # passes
            _FINISH_VERIFIED,
        ]
    )
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=8)

    state = await runner.run(calc_repository_id, "add(a, b) returns the wrong value")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["calc.py", "calc.py"]  # two apply_patch calls
    assert len(state.test_results) == 2
    assert state.test_results[0].passed is False
    assert state.test_results[0].failure_category == "test_failure"
    assert state.test_results[1].passed is True
    assert state.verification_status == "verified"  # derived from the *last* run only


async def test_agent_run_persists_test_runs(db_session, calc_repository_id, tool_registry):
    llm = FakeLLMProvider(
        responses=[
            _PLAN_RESPONSE,
            _apply_patch_action("return a - b", "return a + b"),
            _RUN_TESTS_ACTION,
            _FINISH_VERIFIED,
        ]
    )
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=6)

    state = await runner.run(calc_repository_id, "task")

    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.verification_status == "verified"

    test_runs = db_session.scalars(select(TestRun).where(TestRun.run_id == state.run_id)).all()
    assert len(test_runs) == 1
    assert test_runs[0].passed is True
    assert test_runs[0].repository_id == calc_repository_id


# --- create_file: the tool that plugs the "can't make a new file" gap -------
#
# Regression coverage for a run observed live: needing tests/test_x.py to
# exist, the model tried run_command(["touch", ...]) — correctly rejected by
# the allowlist — then repeated that exact rejected call until the loop guard
# stopped it, having never created anything. create_file is the fix; these
# confirm it plugs all the way through the same tracking apply_patch gets.

_CREATE_FILE_ACTION = (
    '{"thought": "add a test file", "action": {"tool": "create_file", '
    '"input": {"path": "tests/test_entrypoint.py", "content": "def test_it():\\n    assert True\\n"}}, '
    '"finish": null}'
)
_FINISH_AFTER_CREATE = (
    '{"thought": "done", "action": null, "finish": {"answer": "added a test file", "root_cause": null}}'
)


async def test_agent_creates_a_file_and_tracks_it_as_modified(
    db_session, indexed_repository_id, tool_registry, sample_repo
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _CREATE_FILE_ACTION, _FINISH_AFTER_CREATE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "add a test file for entrypoint")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["tests/test_entrypoint.py"]
    assert state.verification_status == "unverified"
    assert (sample_repo / "tests" / "test_entrypoint.py").read_text() == "def test_it():\n    assert True\n"


async def test_create_file_result_persists_as_a_modified_file_row(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _CREATE_FILE_ACTION, _FINISH_AFTER_CREATE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "task")

    modified = db_session.scalars(select(ModifiedFile).where(ModifiedFile.run_id == state.run_id)).all()
    assert len(modified) == 1
    assert modified[0].relative_path == "tests/test_entrypoint.py"
    assert modified[0].lines_added == 2
    assert modified[0].lines_removed == 0
    assert "+def test_it():" in modified[0].diff


# --- delete_file: the missing half of a rename ------------------------------
#
# Regression coverage for a run observed live: asked to rename calc.py to
# adv_calc.py, the model's own plan literally said "Rename calc.py to
# adv_calc.py" — a step with no corresponding tool, since apply_patch only
# modifies an existing path and create_file only adds a new one. It never
# attempted either; it re-searched for content that didn't exist until the
# redundancy guard stopped it. delete_file is the other half create_file
# needed to make a rename actually possible.

_DELETE_FILE_ACTION = (
    '{"thought": "remove the old file", "action": {"tool": "delete_file", '
    '"input": {"path": "main.py"}}, "finish": null}'
)
_FINISH_AFTER_DELETE = (
    '{"thought": "done", "action": null, "finish": {"answer": "removed main.py", "root_cause": null}}'
)


async def test_agent_deletes_a_file_and_tracks_it_as_modified(
    db_session, indexed_repository_id, tool_registry, sample_repo
):
    llm = FakeLLMProvider(responses=[_PLAN_RESPONSE, _DELETE_FILE_ACTION, _FINISH_AFTER_DELETE])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    state = await runner.run(indexed_repository_id, "remove main.py")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["main.py"]
    assert state.verification_status == "unverified"
    assert not (sample_repo / "main.py").exists()


async def test_a_rename_is_create_then_delete_in_one_run(
    db_session, indexed_repository_id, tool_registry, sample_repo
):
    create_new = (
        '{"thought": "write the renamed file", "action": {"tool": "create_file", '
        '"input": {"path": "entrypoint.py", "content": "def entrypoint():\\n    pass\\n"}}, "finish": null}'
    )
    delete_old = (
        '{"thought": "remove the old path", "action": {"tool": "delete_file", '
        '"input": {"path": "main.py"}}, "finish": null}'
    )
    llm = FakeLLMProvider(
        responses=[_PLAN_RESPONSE, create_new, delete_old, _FINISH_AFTER_DELETE]
    )
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=6)

    state = await runner.run(indexed_repository_id, "rename main.py to entrypoint.py")

    assert state.status == AgentStatus.DONE
    assert state.modified_files == ["entrypoint.py", "main.py"]
    assert not (sample_repo / "main.py").exists()
    assert (sample_repo / "entrypoint.py").is_file()
