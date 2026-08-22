"""Chat sessions: a persistent conversation scoped to one repository.

A session is what turns the agent from one-shot into conversational. Posting
a message launches an agent run that is handed the session's earlier turns as
context and the same repository-scoped retrieval tools, and writes its answer
back as the next turn — so the transcript, the run history, and what the model
sees next iteration are all the same object rather than three views that can
drift apart.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.background import BackgroundAgentRunner, get_background_agent_runner
from app.agents.runner import AgentRunner
from app.agents.titling import generate_session_title
from app.api.routes.agent import execute_in_background, to_summary
from app.config.settings import Settings, get_settings
from app.database.models import AgentRun, ChatMessage, ChatSession, Repository
from app.database.session import get_db_session, get_session_factory_dependency
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.llm.base import LLMProvider
from app.llm.catalog import find_catalog_entry, names_match
from app.llm.exceptions import LLMProviderError
from app.llm.factory import get_llm_provider, get_model_manager
from app.llm.management import ModelManager
from app.schemas.sessions import (
    ChatMessageOut,
    ContextWindowOut,
    CreateSessionRequest,
    PostMessageRequest,
    PostMessageResponse,
    RenameSessionRequest,
    SessionDetail,
    SessionSummary,
    TokenUsageOut,
)
from app.tools.registry_factory import build_tool_registry

router = APIRouter(tags=["sessions"])

# A session's name comes from its first message rather than asking for one up
# front — nobody knows what a conversation is about before they start it.
_TITLE_MAX_CHARS = 60


@router.post("/sessions", response_model=SessionSummary, status_code=201)
async def create_session(
    request: CreateSessionRequest, session: Session = Depends(get_db_session)
) -> SessionSummary:
    repository = session.get(Repository, request.repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail=f"Repository '{request.repository_id}' not found")

    row = ChatSession(repository_id=repository.id, title=request.title or "New session")
    session.add(row)
    session.commit()
    return _to_summary(session, row)


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions(
    repository_id: str | None = None, session: Session = Depends(get_db_session)
) -> list[SessionSummary]:
    """Most recently active first — a session list is a "pick up where I left
    off" list, not an archive."""
    query = select(ChatSession).order_by(ChatSession.updated_at.desc())
    if repository_id is not None:
        query = query.where(ChatSession.repository_id == repository_id)
    return [_to_summary(session, row) for row in session.scalars(query)]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(
    session_id: str,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
    manager: ModelManager = Depends(get_model_manager),
) -> SessionDetail:
    row = _require_session(session, session_id)
    runs = _runs_by_id(session, session_id)

    messages = [
        ChatMessageOut(
            id=message.id,
            role=message.role,
            content=message.content,
            run_id=message.run_id,
            created_at=message.created_at,
            run=to_summary(runs[message.run_id]) if message.run_id in runs else None,
            usage=_run_usage(runs[message.run_id]) if message.run_id in runs else None,
        )
        for message in row.messages
    ]

    return SessionDetail(
        **_to_summary(session, row).model_dump(),
        messages=messages,
        usage=_session_usage(runs.values()),
        context_window=await _context_window(llm.model, runs.values(), manager),
    )


@router.patch("/sessions/{session_id}", response_model=SessionSummary)
async def rename_session(
    session_id: str, request: RenameSessionRequest, session: Session = Depends(get_db_session)
) -> SessionSummary:
    row = _require_session(session, session_id)
    row.title = request.title.strip()
    session.commit()
    return _to_summary(session, row)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str, session: Session = Depends(get_db_session)) -> None:
    row = _require_session(session, session_id)
    # The runs themselves outlive the session — they're the audit trail, and
    # deleting a conversation shouldn't destroy the record of what the agent
    # did to the repository. Only the transcript goes.
    session.execute(
        AgentRun.__table__.update().where(AgentRun.session_id == session_id).values(session_id=None)
    )
    session.delete(row)
    session.commit()


@router.post("/sessions/{session_id}/messages", response_model=PostMessageResponse, status_code=201)
async def post_message(
    session_id: str,
    request: PostMessageRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    session_factory: sessionmaker = Depends(get_session_factory_dependency),
    background: BackgroundAgentRunner = Depends(get_background_agent_runner),
) -> PostMessageResponse:
    """Record the user's turn and start the run that answers it.

    Returns as soon as the run is created (status "running"), same as
    POST /api/agent/run — watch it via the run's SSE stream. The assistant's
    turn is written by the run itself when it reaches a terminal status.
    """
    row = _require_session(session, session_id)

    content = request.content.strip()
    # Read the count *before* adding — appending to the session then checking
    # `row.messages` sees the new row via autoflush and never names anything.
    is_first_message = not row.messages
    user_message = ChatMessage(session_id=row.id, role="user", content=content)
    session.add(user_message)
    fallback_title = None
    if is_first_message and row.title == "New session":
        # Instant, so the sidebar never sits on the literal string "New
        # session" — replaced with an AI-generated title in the background
        # below, which takes a real LLM round trip.
        fallback_title = _title_from(content)
        row.title = fallback_title
    session.commit()

    tool_registry = build_tool_registry(session, settings, embedding_provider)
    runner = AgentRunner(
        session, llm, tool_registry, settings.max_agent_iterations, settings, embedding_provider
    )
    run_row = runner.create_run(row.repository_id, content, session_id=row.id)

    background.launch(
        run_row.id,
        execute_in_background(
            run_row.id, row.repository_id, content, session_factory, settings, llm, embedding_provider
        ),
    )
    if fallback_title is not None:
        background.launch(
            _title_task_key(row.id),
            _generate_and_apply_title(row.id, content, fallback_title, session_factory, llm),
        )

    return PostMessageResponse(
        message=ChatMessageOut(
            id=user_message.id,
            role=user_message.role,
            content=user_message.content,
            run_id=None,
            created_at=user_message.created_at,
        ),
        run=to_summary(run_row),
    )


def _title_task_key(session_id: str) -> str:
    # A distinct namespace from agent-run ids (uuid4 hex, no colon) — the
    # background runner keys everything off one dict, and titling has to
    # never collide with an in-flight run's own key.
    return f"title:{session_id}"


async def _generate_and_apply_title(
    session_id: str, first_message: str, fallback_title: str, session_factory: sessionmaker, llm: LLMProvider
) -> None:
    """Runs in its own asyncio.Task with its own DB session, same reasoning
    as `execute_in_background`: the triggering request's session is closed by
    the time this finishes.
    """
    title = await generate_session_title(llm, first_message)
    if title is None:
        return
    db_session = session_factory()
    try:
        row = db_session.get(ChatSession, session_id)
        if row is None:
            return
        # Only overwrite the placeholder this same turn set. If the user (or
        # an earlier duplicate task) already renamed it, leave it alone —
        # naming a session is not this task's business once someone else has.
        if row.title == fallback_title:
            row.title = title
            db_session.commit()
    finally:
        db_session.close()


def _require_session(session: Session, session_id: str) -> ChatSession:
    row = session.get(ChatSession, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return row


def _runs_by_id(session: Session, session_id: str) -> dict[str, AgentRun]:
    runs = session.scalars(select(AgentRun).where(AgentRun.session_id == session_id)).all()
    return {run.id: run for run in runs}


def _title_from(content: str) -> str:
    first_line = content.strip().splitlines()[0].strip()
    if len(first_line) <= _TITLE_MAX_CHARS:
        return first_line
    return first_line[: _TITLE_MAX_CHARS - 1].rstrip() + "…"


def _run_usage(run: AgentRun) -> TokenUsageOut:
    return TokenUsageOut(
        prompt_tokens=run.prompt_tokens,
        completion_tokens=run.completion_tokens,
        total_tokens=run.prompt_tokens + run.completion_tokens,
        llm_calls=run.llm_call_count,
    )


def _session_usage(runs) -> TokenUsageOut:
    total = TokenUsageOut()
    for run in runs:
        total.prompt_tokens += run.prompt_tokens
        total.completion_tokens += run.completion_tokens
        total.llm_calls += run.llm_call_count
    total.total_tokens = total.prompt_tokens + total.completion_tokens
    return total


async def _context_window(model: str, runs, manager: ModelManager) -> ContextWindowOut:
    """Peak prompt size in this session against the active model's window.

    The limit is read from the backend first (it knows the real number for
    the exact tag installed) and falls back to the catalog. If neither has
    it, `limit_tokens` stays None and the UI shows the raw count rather than
    a percentage of a number nobody verified.
    """
    used = max((run.peak_prompt_tokens for run in runs), default=0)
    limit = await _context_limit(model, manager)
    percent = round(min(used / limit, 1.0) * 100, 1) if limit else None
    return ContextWindowOut(model=model, limit_tokens=limit, used_tokens=used, percent=percent)


async def _context_limit(model: str, manager: ModelManager) -> int | None:
    try:
        for installed in await manager.list_installed():
            if names_match(installed.name, model) and installed.context_length:
                return installed.context_length
    except LLMProviderError:
        # Backend unreachable — the catalog is still a usable answer, and a
        # session detail request shouldn't fail because Ollama is down.
        pass
    entry = find_catalog_entry(model)
    return entry.context_window if entry else None


def _to_summary(session: Session, row: ChatSession) -> SessionSummary:
    message_count = session.scalar(
        select(func.count(ChatMessage.id)).where(ChatMessage.session_id == row.id)
    )
    return SessionSummary(
        id=row.id,
        repository_id=row.repository_id,
        repository_name=row.repository.name,
        title=row.title,
        message_count=message_count or 0,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
