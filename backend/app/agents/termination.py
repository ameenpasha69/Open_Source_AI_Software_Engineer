from app.agents.state import AgentState, AgentStatus

# How many consecutive already-answered calls to tolerate before giving up.
# Two is a nudge the model can recover from (the observation tells it the
# call was skipped and why); three in a row is a loop, not a stumble.
MAX_CONSECUTIVE_REDUNDANT_CALLS = 3


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
    if state.consecutive_redundant_calls >= MAX_CONSECUTIVE_REDUNDANT_CALLS:
        # Without this the iteration budget is the only backstop, which on a
        # generous MAX_AGENT_ITERATIONS means dozens of identical LLM calls
        # before anyone notices (observed live: 17 in a row).
        return AgentStatus.NO_PROGRESS
    return None
