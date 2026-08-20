from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator

from app.tools.base import ToolResult


class AgentStatus(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    CANCELLED = "cancelled"


class AgentAction(BaseModel):
    tool: str
    input: dict[str, Any] = {}

    @field_validator("input", mode="before")
    @classmethod
    def _coerce_null_input_to_empty_dict(cls, value: Any) -> Any:
        # Observed live: qwen2.5-coder:7b reliably emits `"input": null` for
        # tools that take no required arguments (e.g. run_tests with no
        # test_path) rather than `{}` — a reasonable reading of "no input
        # needed" that strict dict validation was rejecting outright, wasting
        # an iteration (and the one retry) every time it happened.
        return {} if value is None else value


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


class TestRunRecord(BaseModel):
    __test__ = False  # not a pytest test class — this name just mirrors the domain

    command: str
    scope: Literal["full", "targeted"]
    passed: bool
    failed_tests: list[str]
    failure_category: str


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
    modified_files: list[str] = []
    test_results: list[TestRunRecord] = []
    iteration: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: str | None = None
    root_cause: str | None = None
    error: str | None = None

    @property
    def verification_status(self) -> str:
        """Derived from the *most recent* test run, not a full correlation
        with which modification it was checking — a reasonable
        simplification: whatever the agent's last test run showed is the
        best available evidence of whether its change works.

        - "not_applicable": no code was ever modified.
        - "unverified": modified, but never ran tests to check.
        - "failed": the most recent test run failed.
        - "partially_verified": the most recent test run passed, but was
          scoped to specific tests rather than the full suite.
        - "verified": the most recent test run passed, running the full suite.
        """
        if not self.modified_files:
            return "not_applicable"
        if not self.test_results:
            return "unverified"
        last = self.test_results[-1]
        if not last.passed:
            return "failed"
        return "verified" if last.scope == "full" else "partially_verified"
