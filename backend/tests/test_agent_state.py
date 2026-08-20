import pytest
from app.agents.state import AgentAction, AgentDecision, AgentFinish
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
