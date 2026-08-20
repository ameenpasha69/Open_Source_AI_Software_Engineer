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
    started_at: datetime.datetime
    finished_at: datetime.datetime | None
    error: str | None


class AgentEventOut(BaseModel):
    iteration: int
    event_type: str
    payload: dict[str, Any]
    created_at: datetime.datetime
