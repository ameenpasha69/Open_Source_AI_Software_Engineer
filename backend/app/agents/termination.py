from app.agents.state import AgentState, AgentStatus


def check_termination(state: AgentState, max_iterations: int) -> AgentStatus | None:
    """Decides whether the loop should stop before running another
    iteration. Returns the terminal AgentStatus to apply, or None to keep
    going. Kept as its own inspectable function — not inline `if`s scattered
    through the loop — per the project's explicit "Termination Logic"
    architecture component; it's also the one place a future milestone
    (e.g. a cost/token budget, a cancellation flag) adds a new stop
    condition.
    """
    if state.status != AgentStatus.RUNNING:
        return state.status
    if state.iteration >= max_iterations:
        return AgentStatus.MAX_ITERATIONS_REACHED
    return None
