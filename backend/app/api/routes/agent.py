import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.agents.background import BackgroundAgentRunner, get_background_agent_runner
from app.agents.runner import AgentRunner
from app.config.settings import Settings, get_settings
from app.database.models import AgentEvent, AgentRun, AgentToolCall, TestRun
from app.database.session import get_db_session, get_session_factory_dependency
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.retrieval.indexer import RepositoryNotFoundError
from app.schemas.agent import (
    AgentDiffResponse,
    AgentEventOut,
    AgentRunSummary,
    ModifiedFileOut,
    RunAgentRequest,
    TestRunOut,
    ToolCallOut,
)
from app.tools.registry_factory import build_tool_registry

router = APIRouter(tags=["agent"])

_STREAM_POLL_INTERVAL_SECONDS = 0.3


@router.post("/agent/run", response_model=AgentRunSummary)
async def run_agent(
    request: RunAgentRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
    session_factory: sessionmaker = Depends(get_session_factory_dependency),
    background: BackgroundAgentRunner = Depends(get_background_agent_runner),
) -> AgentRunSummary:
    """Creates the run and hands it off to a background asyncio.Task,
    returning immediately with status "running". A real investigation can
    take tens of seconds (several LLM calls plus tool executions) — this
    endpoint no longer blocks for that; watch progress via
    GET /api/agent/{run_id}/stream or poll GET /api/agent/{run_id}.
    """
    tool_registry = build_tool_registry(session, settings, embedding_provider)
    runner = AgentRunner(session, llm, tool_registry, settings.max_agent_iterations)
    try:
        run_row = runner.create_run(request.repository_id, request.task)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    background.launch(
        run_row.id,
        _execute_in_background(
            run_row.id, request.repository_id, request.task, session_factory, settings, llm, embedding_provider
        ),
    )
    return _to_summary(run_row)


async def _execute_in_background(
    run_id: str,
    repository_id: str,
    task: str,
    session_factory: sessionmaker,
    settings: Settings,
    llm: LLMProvider,
    embedding_provider: EmbeddingProvider,
) -> None:
    """Runs in its own asyncio.Task with its own DB session — a session from
    the triggering request can't be reused here, since it's closed the
    moment that request returns."""
    session = session_factory()
    try:
        tool_registry = build_tool_registry(session, settings, embedding_provider)
        runner = AgentRunner(session, llm, tool_registry, settings.max_agent_iterations)
        await runner.execute(run_id, repository_id, task)
    finally:
        session.close()


@router.post("/agent/{run_id}/cancel", response_model=AgentRunSummary)
async def cancel_agent_run(
    run_id: str,
    session: Session = Depends(get_db_session),
    background: BackgroundAgentRunner = Depends(get_background_agent_runner),
) -> AgentRunSummary:
    run_row = session.get(AgentRun, run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")

    if run_row.status == "running":
        cancelled = await background.cancel(run_id)
        if not cancelled:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Run '{run_id}' is marked running but isn't tracked in this process "
                    "(the server may have restarted since it started) and can't be cancelled."
                ),
            )
        session.refresh(run_row)

    return _to_summary(run_row)


@router.get("/agent/{run_id}", response_model=AgentRunSummary)
async def get_agent_run(run_id: str, session: Session = Depends(get_db_session)) -> AgentRunSummary:
    run_row = session.get(AgentRun, run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return _to_summary(run_row)


@router.get("/agent/{run_id}/events", response_model=list[AgentEventOut])
async def get_agent_run_events(run_id: str, session: Session = Depends(get_db_session)) -> list[AgentEventOut]:
    if session.get(AgentRun, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    events = session.scalars(
        select(AgentEvent).where(AgentEvent.run_id == run_id).order_by(AgentEvent.created_at)
    ).all()
    return [_to_event_out(e) for e in events]


@router.get("/agent/{run_id}/stream")
async def stream_agent_run(
    run_id: str,
    request: Request,
    session: Session = Depends(get_db_session),
    session_factory: sessionmaker = Depends(get_session_factory_dependency),
) -> StreamingResponse:
    """Server-Sent Events: replays every event recorded so far, then keeps
    the connection open, polling the DB for new AgentEvent rows and pushing
    each one as it's committed, until the run reaches a terminal status.

    DB-polling rather than an in-process pub/sub queue: the writer (the
    background task) and this stream are different asyncio.Tasks with their
    own DB sessions, and SQLAlchemy's identity map means only a *fresh*
    session reliably sees another session's commits — simplest correct fix
    is a new short-lived session each poll, which also means this survives
    the browser reconnecting mid-run without losing anything.
    """
    if session.get(AgentRun, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")

    return StreamingResponse(
        _event_stream(run_id, request, session_factory), media_type="text/event-stream"
    )


async def _event_stream(run_id: str, request: Request, session_factory: sessionmaker) -> AsyncIterator[str]:
    seen_event_ids: set[str] = set()
    while True:
        if await request.is_disconnected():
            return

        poll_session = session_factory()
        try:
            run_row = poll_session.get(AgentRun, run_id)
            events = poll_session.scalars(
                select(AgentEvent).where(AgentEvent.run_id == run_id).order_by(AgentEvent.created_at)
            ).all()
            new_events = [e for e in events if e.id not in seen_event_ids]
            status = run_row.status
        finally:
            poll_session.close()

        for event in new_events:
            seen_event_ids.add(event.id)
            # Deliberately unnamed (no "event:" line): event_type already
            # travels inside the JSON payload, and a plain default message is
            # what EventSource.onmessage picks up without the client having
            # to register a listener per possible event_type ahead of time.
            yield f"data: {_to_event_out(event).model_dump_json()}\n\n"

        if status != "running":
            yield f"event: run_completed\ndata: {json.dumps({'status': status})}\n\n"
            return

        await asyncio.sleep(_STREAM_POLL_INTERVAL_SECONDS)


@router.get("/agent/{run_id}/diff", response_model=AgentDiffResponse)
async def get_agent_run_diff(run_id: str, session: Session = Depends(get_db_session)) -> AgentDiffResponse:
    run_row = session.get(AgentRun, run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    return AgentDiffResponse(
        run_id=run_id,
        verification_status=run_row.verification_status,
        modified_files=[
            ModifiedFileOut(
                path=m.relative_path, diff=m.diff, lines_added=m.lines_added, lines_removed=m.lines_removed
            )
            for m in run_row.modified_files
        ],
    )


@router.get("/agent/{run_id}/tests", response_model=list[TestRunOut])
async def get_agent_run_tests(run_id: str, session: Session = Depends(get_db_session)) -> list[TestRunOut]:
    if session.get(AgentRun, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    rows = session.scalars(select(TestRun).where(TestRun.run_id == run_id).order_by(TestRun.created_at)).all()
    return [
        TestRunOut(
            command=r.command,
            scope=r.scope,
            passed=r.passed,
            total_tests=r.total_tests,
            passed_tests=r.passed_tests,
            failed_tests=json.loads(r.failed_tests_json),
            failure_category=r.failure_category,
            duration_seconds=r.duration_seconds,
            stdout=r.stdout,
            stderr=r.stderr,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/agent/{run_id}/tool-calls", response_model=list[ToolCallOut])
async def get_agent_run_tool_calls(run_id: str, session: Session = Depends(get_db_session)) -> list[ToolCallOut]:
    """Full per-call detail (arguments, output) — the source for the UI's
    Code panel (search/read results the agent retrieved) and a richer audit
    trail than /events, whose payloads are deliberately compact."""
    if session.get(AgentRun, run_id) is None:
        raise HTTPException(status_code=404, detail=f"Agent run '{run_id}' not found")
    rows = session.scalars(
        select(AgentToolCall).where(AgentToolCall.run_id == run_id).order_by(AgentToolCall.created_at)
    ).all()
    return [
        ToolCallOut(
            iteration=r.iteration,
            tool_name=r.tool_name,
            input=json.loads(r.input_json),
            success=r.success,
            output=json.loads(r.output_json) if r.output_json else None,
            error=r.error,
            duration_seconds=r.duration_seconds,
            created_at=r.created_at,
        )
        for r in rows
    ]


def _to_event_out(event: AgentEvent) -> AgentEventOut:
    return AgentEventOut(
        iteration=event.iteration,
        event_type=event.event_type,
        payload=json.loads(event.payload_json),
        created_at=event.created_at,
    )


def _to_summary(run_row: AgentRun) -> AgentRunSummary:
    return AgentRunSummary(
        id=run_row.id,
        repository_id=run_row.repository_id,
        task=run_row.task,
        status=run_row.status,
        plan=json.loads(run_row.plan_json),
        final_answer=run_row.final_answer,
        root_cause=run_row.root_cause,
        iteration_count=run_row.iteration_count,
        modified_files=[m.relative_path for m in run_row.modified_files],
        verification_status=run_row.verification_status,
        started_at=run_row.started_at,
        finished_at=run_row.finished_at,
        error=run_row.error,
    )
