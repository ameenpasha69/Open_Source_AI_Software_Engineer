from app.agents.state import AgentState, AgentStatus
from app.agents.termination import check_termination


def _state(**overrides) -> AgentState:
    defaults = {"run_id": "r1", "task": "t", "repository_id": "repo1"}
    defaults.update(overrides)
    return AgentState(**defaults)


def test_no_termination_when_running_and_under_iteration_limit():
    assert check_termination(_state(iteration=2), max_iterations=8) is None


def test_terminates_at_max_iterations():
    assert check_termination(_state(iteration=8), max_iterations=8) == AgentStatus.MAX_ITERATIONS_REACHED


def test_terminates_past_max_iterations():
    assert check_termination(_state(iteration=9), max_iterations=8) == AgentStatus.MAX_ITERATIONS_REACHED


def test_already_terminal_status_is_preserved():
    assert check_termination(_state(status=AgentStatus.DONE, iteration=1), max_iterations=8) == AgentStatus.DONE
    assert check_termination(_state(status=AgentStatus.FAILED, iteration=1), max_iterations=8) == AgentStatus.FAILED
