"""Detecting when the agent is asking a question it has already answered.

Observed live (run d5ada39, 45 iterations): the model issued the *identical*
`search_code(query="convert integer to string without str")` call 17 times in
a row, then concluded the task was impossible. Nothing in the loop noticed.
A tool call whose inputs are unchanged, against a repository that is also
unchanged, returns exactly what it returned before — so re-running it cannot
move the agent forward, it can only burn an iteration and an LLM call.

The check deliberately keys on "has anything changed since?" rather than
banning repeats outright. Re-running `run_tests` with the same arguments
after a patch is the core self-correction move and has to stay allowed; the
same call with *nothing* changed in between is the pathology.
"""

import json
from typing import Any

from app.agents.state import AgentState

# Tools that can change what any other tool would return. A successful call
# to one of these invalidates every earlier observation, so identical calls
# made before it are fair game to repeat.
MUTATING_TOOLS = frozenset({"apply_patch", "run_command", "run_formatter"})


def find_redundant_call(state: AgentState, tool_name: str, input_data: dict[str, Any]) -> int | None:
    """Return the iteration of an earlier identical call whose result must
    still hold, or None if this call could genuinely produce new information.
    """
    signature = _signature(tool_name, input_data)
    last_mutation = _last_mutation_iteration(state)

    for record in reversed(state.tool_calls):
        # Everything at or before the last repository change is stale — a
        # call made back then is allowed to be made again now.
        if record.iteration <= last_mutation:
            return None
        if _signature(record.tool_name, record.input) == signature:
            return record.iteration
    return None


def _last_mutation_iteration(state: AgentState) -> int:
    return max(
        (r.iteration for r in state.tool_calls if r.tool_name in MUTATING_TOOLS and r.result.success),
        default=0,
    )


def _signature(tool_name: str, input_data: dict[str, Any]) -> str:
    # sort_keys so {"path": p, "line": n} and {"line": n, "path": p} — which
    # the model emits interchangeably — compare equal. default=str keeps a
    # non-JSON-serializable value from raising instead of being compared.
    return f"{tool_name}({json.dumps(input_data, sort_keys=True, default=str)})"


def describe_call(tool_name: str, input_data: dict[str, Any]) -> str:
    """Compact, human/LLM-readable rendering of a call, matching the style
    the context manager uses for observations."""
    args = ", ".join(f"{k}={v!r}" for k, v in sorted(input_data.items()))
    return f"{tool_name}({args})"
