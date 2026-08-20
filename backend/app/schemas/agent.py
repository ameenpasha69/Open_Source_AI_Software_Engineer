import datetime
from typing import Any

from pydantic import BaseModel


class RunAgentRequest(BaseModel):
    repository_id: str
    task: str


class AgentRunSummary(BaseModel):
    id: str
    repository_id: str
    task: str
    status: str
    plan: list[str]
    final_answer: str | None
    root_cause: str | None
    iteration_count: int
    modified_files: list[str]
    verification_status: str
    started_at: datetime.datetime
    finished_at: datetime.datetime | None
    error: str | None


class AgentEventOut(BaseModel):
    iteration: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime.datetime


class ModifiedFileOut(BaseModel):
    path: str
    diff: str
    lines_added: int
    lines_removed: int


class AgentDiffResponse(BaseModel):
    run_id: str
    verification_status: str
    modified_files: list[ModifiedFileOut]


class TestRunOut(BaseModel):
    command: str
    scope: str
    passed: bool
    total_tests: int | None
    passed_tests: int | None
    failed_tests: list[str]
    failure_category: str
    duration_seconds: float
    stdout: str
    stderr: str
    created_at: datetime.datetime


class ToolCallOut(BaseModel):
    iteration: int
    tool_name: str
    input: dict[str, Any]
    success: bool
    output: dict[str, Any] | None
    error: str | None
    duration_seconds: float
    created_at: datetime.datetime
