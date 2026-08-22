import datetime

from pydantic import BaseModel, Field

from app.schemas.agent import AgentRunSummary


class CreateSessionRequest(BaseModel):
    repository_id: str
    title: str | None = None


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class PostMessageRequest(BaseModel):
    content: str = Field(min_length=1)


class TokenUsageOut(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0


class ContextWindowOut(BaseModel):
    """How close the largest single prompt came to the model's limit.

    Peak, not total: a session's cumulative tokens routinely exceed the
    window several times over without any individual request being near it,
    so a total-over-limit gauge would show 400% and mean nothing. This is the
    number that actually predicts a truncated prompt.
    """

    model: str
    # None when the backend doesn't report a window for this model and it
    # isn't in the catalog either — better to show "unknown" than a guess.
    limit_tokens: int | None = None
    used_tokens: int = 0
    percent: float | None = None


class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    run_id: str | None
    created_at: datetime.datetime
    # Assistant turns carry the run that produced them, so the transcript can
    # expand into the full timeline / diff / tests without a second lookup.
    run: AgentRunSummary | None = None
    usage: TokenUsageOut | None = None


class SessionSummary(BaseModel):
    id: str
    repository_id: str
    repository_name: str
    title: str
    message_count: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class SessionDetail(SessionSummary):
    messages: list[ChatMessageOut]
    usage: TokenUsageOut
    context_window: ContextWindowOut


class PostMessageResponse(BaseModel):
    message: ChatMessageOut
    run: AgentRunSummary
