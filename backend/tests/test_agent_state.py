import json

import pytest
from app.agents.state import AgentAction, AgentDecision, AgentFinish, AgentState, TestRunRecord
from pydantic import ValidationError


def test_decision_with_action_only_is_valid():
    decision = AgentDecision(thought="looking", action=AgentAction(tool="find_symbol", input={"symbol": "x"}))
    assert decision.action is not None
    assert decision.finish is None


def test_decision_with_finish_only_is_valid():
    decision = AgentDecision(thought="done", finish=AgentFinish(answer="the bug is here"))
    assert decision.finish is not None
    assert decision.action is None


def test_decision_with_both_action_and_finish_is_invalid():
    with pytest.raises(ValidationError, match="Exactly one"):
        AgentDecision(
            thought="confused",
            action=AgentAction(tool="find_symbol", input={}),
            finish=AgentFinish(answer="done"),
        )


def test_decision_with_neither_action_nor_finish_is_invalid():
    with pytest.raises(ValidationError, match="Exactly one"):
        AgentDecision(thought="stuck")


def test_action_coerces_null_input_to_empty_dict():
    """Regression: qwen2.5-coder:7b reliably emits "input": null for tools
    with no required arguments — found via live testing (run_tests kept
    failing to parse for several iterations before this fix)."""
    action = AgentAction(tool="run_tests", input=None)
    assert action.input == {}


def test_decision_parses_from_json_with_null_action_input():
    raw = '{"thought": "check", "action": {"tool": "run_tests", "input": null}, "finish": null}'
    decision = AgentDecision.model_validate(json.loads(raw))
    assert decision.action.tool == "run_tests"
    assert decision.action.input == {}


def _state(**overrides) -> AgentState:
    defaults = {"run_id": "r1", "task": "t", "repository_id": "repo1"}
    defaults.update(overrides)
    return AgentState(**defaults)


def _test_run(passed: bool, scope: str = "full") -> TestRunRecord:
    return TestRunRecord(command="pytest", scope=scope, passed=passed, failed_tests=[], failure_category="none")


def test_verification_status_not_applicable_when_nothing_modified():
    assert _state().verification_status == "not_applicable"


def test_verification_status_unverified_when_modified_but_no_tests_run():
    assert _state(modified_files=["a.py"]).verification_status == "unverified"


def test_verification_status_verified_when_full_suite_passed():
    state = _state(modified_files=["a.py"], test_results=[_test_run(passed=True, scope="full")])
    assert state.verification_status == "verified"


def test_verification_status_partially_verified_when_targeted_run_passed():
    state = _state(modified_files=["a.py"], test_results=[_test_run(passed=True, scope="targeted")])
    assert state.verification_status == "partially_verified"


def test_verification_status_failed_when_most_recent_run_failed():
    state = _state(modified_files=["a.py"], test_results=[_test_run(passed=False)])
    assert state.verification_status == "failed"


def test_verification_status_uses_most_recent_test_run_only():
    state = _state(
        modified_files=["a.py"],
        test_results=[_test_run(passed=False), _test_run(passed=True, scope="full")],
    )
    assert state.verification_status == "verified"
