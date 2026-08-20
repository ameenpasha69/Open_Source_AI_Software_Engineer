from enum import StrEnum
from typing import Any

from pydantic import BaseModel, model_validator

from app.tools.base import ToolResult


class AgentStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"


class AgentAction(BaseModel):
    tool: str
    input: dict[str, Any] = {}


class AgentFinish(BaseModel):
    answer: str
    root_cause: str | None = None


class AgentDecision(BaseModel):
    """What the LLM returns each iteration, parsed from its JSON-mode
    response. Exactly one of `action` (call a tool) or `finish` (stop and
    answer) must be set — never both, never neither.
    """

    thought: str
    action: AgentAction | None = None
    finish: AgentFinish | None = None

    @model_validator(mode="after")
    def _exactly_one_of_action_or_finish(self) -> "AgentDecision":
        if (self.action is None) == (self.finish is None):
            raise ValueError("Exactly one of 'action' or 'finish' must be set")
        return self


class ToolCallRecord(BaseModel):
    iteration: int
    tool_name: str
    input: dict[str, Any]
    result: ToolResult


class AgentState(BaseModel):
    """Explicit, structured agent memory — not a raw conversation transcript.
    Every field here is something a caller (API response, DB row, the next
    prompt) can inspect independently, per the project's "structured state,
    not a giant message history" design goal.
    """

    run_id: str
    task: str
    repository_id: str
    plan: list[str] = []
    observations: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    modified_files: list[str] = []  # empty until Milestone 7 (apply_patch)
    test_results: list[Any] = []  # empty until Milestone 8 (run_tests)
    iteration: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None
    root_cause: str | None = None
    error: str | None = None
