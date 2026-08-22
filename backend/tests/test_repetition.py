"""Regression tests for the loop that ran 45 iterations (run d5ada39): the
model re-issued one identical search_code call 17 times in a row and nothing
in the agent stopped it."""

import pytest
from app.agents.repetition import describe_call, find_redundant_call
from app.agents.runner import AgentRunner, _decision_temperature
from app.agents.state import AgentState, AgentStatus, ToolCallRecord
from app.agents.termination import MAX_CONSECUTIVE_REDUNDANT_CALLS, check_termination
from app.database.models import AgentEvent, AgentToolCall
from app.retrieval.indexer import RepositoryIndexer
from app.tools.base import ToolRegistry, ToolResult
from app.tools.code_search_tools import FindSymbolTool
from app.tools.execution_tools import RunTestsTool
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
    registry.register(RunTestsTool(db_session, timeout_seconds=30.0))
    return registry


def _state(*records: ToolCallRecord) -> AgentState:
    return AgentState(run_id="r", task="t", repository_id="repo", tool_calls=list(records))


def _call(iteration: int, tool: str, input_data: dict, success: bool = True) -> ToolCallRecord:
    return ToolCallRecord(
        iteration=iteration,
        tool_name=tool,
        input=input_data,
        result=ToolResult(tool_name=tool, success=success, output={} if success else None, duration_seconds=0.0),
    )


# --- find_redundant_call -----------------------------------------------------


def test_a_first_time_call_is_not_redundant():
    state = _state()
    assert find_redundant_call(state, "search_code", {"query": "palindrome"}) is None


def test_an_identical_repeat_is_redundant_and_names_the_earlier_iteration():
    state = _state(_call(3, "search_code", {"query": "palindrome"}))
    assert find_redundant_call(state, "search_code", {"query": "palindrome"}) == 3


def test_the_same_tool_with_different_arguments_is_not_redundant():
    state = _state(_call(3, "search_code", {"query": "palindrome"}))
    assert find_redundant_call(state, "search_code", {"query": "reverse an int"}) is None


def test_a_different_tool_with_the_same_arguments_is_not_redundant():
    state = _state(_call(3, "search_code", {"query": "palindrome"}))
    assert find_redundant_call(state, "find_symbol", {"query": "palindrome"}) is None


def test_argument_order_does_not_affect_the_match():
    # The model emits key orders interchangeably across turns; a repeat is a
    # repeat regardless.
    state = _state(_call(2, "get_file_context", {"path": "p.py", "line": 1}))
    assert find_redundant_call(state, "get_file_context", {"line": 1, "path": "p.py"}) == 2


def test_the_most_recent_matching_iteration_is_reported():
    state = _state(
        _call(2, "search_code", {"query": "x"}),
        _call(5, "search_code", {"query": "x"}),
    )
    assert find_redundant_call(state, "search_code", {"query": "x"}) == 5


def test_a_successful_patch_makes_earlier_calls_repeatable_again():
    # Re-reading a file after changing it is the whole point of self-correction.
    state = _state(
        _call(1, "read_file", {"path": "calc.py"}),
        _call(2, "apply_patch", {"path": "calc.py"}),
    )
    assert find_redundant_call(state, "read_file", {"path": "calc.py"}) is None


def test_rerunning_tests_after_a_patch_is_allowed():
    state = _state(
        _call(1, "run_tests", {}),
        _call(2, "apply_patch", {"path": "calc.py"}),
    )
    assert find_redundant_call(state, "run_tests", {}) is None


def test_rerunning_tests_with_nothing_changed_in_between_is_redundant():
    state = _state(_call(1, "run_tests", {}))
    assert find_redundant_call(state, "run_tests", {}) == 1


def test_a_failed_patch_does_not_make_earlier_calls_repeatable():
    # A patch that didn't apply changed nothing, so the file still reads the same.
    state = _state(
        _call(1, "read_file", {"path": "calc.py"}),
        _call(2, "apply_patch", {"path": "calc.py"}, success=False),
    )
    assert find_redundant_call(state, "read_file", {"path": "calc.py"}) == 1


def test_a_repeat_made_after_a_patch_is_still_redundant():
    state = _state(
        _call(1, "apply_patch", {"path": "calc.py"}),
        _call(2, "read_file", {"path": "calc.py"}),
    )
    assert find_redundant_call(state, "read_file", {"path": "calc.py"}) == 2


def test_describe_call_renders_arguments_readably():
    assert describe_call("search_code", {"query": "abc"}) == "search_code(query='abc')"


# --- termination -------------------------------------------------------------


def test_consecutive_redundant_calls_terminate_the_run():
    state = AgentState(
        run_id="r", task="t", repository_id="repo", consecutive_redundant_calls=MAX_CONSECUTIVE_REDUNDANT_CALLS
    )
    assert check_termination(state, max_iterations=100) is AgentStatus.NO_PROGRESS


def test_a_run_below_the_threshold_keeps_going():
    state = AgentState(
        run_id="r", task="t", repository_id="repo", consecutive_redundant_calls=MAX_CONSECUTIVE_REDUNDANT_CALLS - 1
    )
    assert check_termination(state, max_iterations=100) is None


# --- end to end through the runner -------------------------------------------

_PLAN = '{"steps": ["Look at the code"]}'
_FIND_SYMBOL = (
    '{"thought": "look for entrypoint", '
    '"action": {"tool": "find_symbol", "input": {"symbol": "entrypoint"}}, "finish": null}'
)
_FINISH = '{"thought": "done", "action": null, "finish": {"answer": "found it", "root_cause": null}}'


async def test_a_repeated_call_is_not_executed_twice(db_session, indexed_repository_id, tool_registry):
    llm = FakeLLMProvider(responses=[_PLAN, _FIND_SYMBOL, _FIND_SYMBOL, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=10)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    assert state.status == AgentStatus.DONE
    # Three iterations, but the tool ran once — the repeat was answered from
    # what the agent already knew.
    assert len(state.tool_calls) == 1
    persisted = db_session.scalars(select(AgentToolCall).where(AgentToolCall.run_id == state.run_id)).all()
    assert len(persisted) == 1


async def test_a_repeated_call_tells_the_model_why_it_was_skipped(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, _FIND_SYMBOL, _FIND_SYMBOL, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=10)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    skipped = [o for o in state.observations if "SKIPPED" in o]
    assert len(skipped) == 1
    assert "iteration 1" in skipped[0]
    assert "cannot tell you anything new" in skipped[0]


async def test_a_repeated_call_is_visible_in_the_event_stream(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, _FIND_SYMBOL, _FIND_SYMBOL, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=10)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    events = db_session.scalars(
        select(AgentEvent).where(
            AgentEvent.run_id == state.run_id, AgentEvent.event_type == "redundant_call_skipped"
        )
    ).all()
    assert len(events) == 1


async def test_a_model_stuck_on_one_call_is_stopped_long_before_the_iteration_budget(
    db_session, indexed_repository_id, tool_registry
):
    # This is the observed failure, reproduced: the model asks the same
    # question forever. Before the fix this ran until max_iterations.
    llm = FakeLLMProvider(responses=[_PLAN, *[_FIND_SYMBOL] * 50])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=50)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    assert state.status == AgentStatus.NO_PROGRESS
    assert state.iteration == 1 + MAX_CONSECUTIVE_REDUNDANT_CALLS
    assert len(state.tool_calls) == 1
    assert "repeating itself" in state.error


async def test_the_no_progress_reason_is_persisted_on_the_run(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, *[_FIND_SYMBOL] * 50])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=50)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    from app.database.models import AgentRun

    run_row = db_session.get(AgentRun, state.run_id)
    assert run_row.status == "no_progress"
    assert "repeating itself" in run_row.error


async def test_an_intervening_different_call_resets_the_redundancy_streak(
    db_session, indexed_repository_id, tool_registry
):
    read_main = (
        '{"thought": "read it", "action": {"tool": "read_file", "input": {"path": "main.py"}}, "finish": null}'
    )
    # repeat, repeat, then something new — the streak resets, so the run is
    # not stopped and reaches its own conclusion.
    llm = FakeLLMProvider(
        responses=[_PLAN, _FIND_SYMBOL, _FIND_SYMBOL, _FIND_SYMBOL, read_main, _FIND_SYMBOL, _FINISH]
    )
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=20)

    state = await runner.run(indexed_repository_id, "Where is entrypoint?")

    assert state.status == AgentStatus.DONE
    assert state.consecutive_redundant_calls == 1


# --- escalating decoding temperature after a redundant call ------------------
#
# Regression coverage for a run observed live (qwen2.5-coder:7b, a
# rename+rewrite task): after a call was correctly skipped as redundant and
# the corrective observation was appended to the prompt exactly as designed,
# the model's very next response was — word for word — the same thought and
# the same tool call, three more times in a row. At the default decoding
# temperature (0.2, chosen for precision), a small textual change appended
# near the end of an otherwise-unchanged, fairly long prompt often isn't
# enough to move the argmax path. Once a repeat has actually happened,
# `_decision_temperature` asks for more sampling diversity on the next call.


def test_decision_temperature_defers_to_the_default_with_no_redundancy():
    state = AgentState(run_id="r", task="t", repository_id="repo", consecutive_redundant_calls=0)
    assert _decision_temperature(state) is None


def test_decision_temperature_escalates_with_each_consecutive_redundant_call():
    state = AgentState(run_id="r", task="t", repository_id="repo", consecutive_redundant_calls=1)
    first = _decision_temperature(state)
    state.consecutive_redundant_calls = 2
    second = _decision_temperature(state)
    assert 0 < first < second


def test_decision_temperature_is_capped():
    state = AgentState(run_id="r", task="t", repository_id="repo", consecutive_redundant_calls=50)
    assert _decision_temperature(state) <= 0.9


async def test_a_stuck_model_gets_higher_temperature_on_the_calls_right_after_a_skip(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, *[_FIND_SYMBOL] * 50])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=50)

    await runner.run(indexed_repository_id, "Where is entrypoint?")

    # Escalation is based on the streak *heading into* a call, so it can only
    # show up starting from the call made right after the first skip: [plan,
    # iter1 (first call ever, nothing redundant yet), iter2 (still nothing
    # redundant when this call was *made* — the skip happens after), iter3
    # (now escalated, since iter2's call was skipped), iter4 (escalated
    # further) — then the 3rd skip stops the run before a 5th call.
    assert llm.temperatures[0] is None  # planner call
    assert llm.temperatures[1] is None  # iteration 1
    assert llm.temperatures[2] is None  # iteration 2
    assert llm.temperatures[3] is not None  # iteration 3, after the first skip
    assert llm.temperatures[4] is not None  # iteration 4, after the second skip
    assert llm.temperatures[4] > llm.temperatures[3]
    assert len(llm.temperatures) == 5  # the third skip stops the run here


async def test_an_ordinary_iteration_does_not_escalate_temperature(
    db_session, indexed_repository_id, tool_registry
):
    llm = FakeLLMProvider(responses=[_PLAN, _FIND_SYMBOL, _FINISH])
    runner = AgentRunner(db_session, llm, tool_registry, max_iterations=5)

    await runner.run(indexed_repository_id, "Where is entrypoint?")

    assert all(t is None for t in llm.temperatures)
