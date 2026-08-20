import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.runner import AgentRunner
from app.config.settings import Settings, get_settings
from app.database.models import AgentEvent, AgentRun
from app.database.session import get_db_session
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
)
from app.tools.registry_factory import build_tool_registry

router = APIRouter(tags=["agent"])


@router.post("/agent/run", response_model=AgentRunSummary)
async def run_agent(
    request: RunAgentRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> AgentRunSummary:
    """Runs the agent loop to completion and returns the final state.

    This blocks for the duration of the run (a real investigation can take
    tens of seconds — several LLM calls plus tool executions). It doesn't
    block *other* API requests (the loop is `async`/non-blocking under the
    hood), but there's no background execution, polling, or cancellation
    yet — that arrives with Milestone 9's streaming work, where a run needs
    to be observable and cancellable while still in progress.
    """
    tool_registry = build_tool_registry(session, settings, embedding_provider)
    runner = AgentRunner(session, llm, tool_registry, settings.max_agent_iterations)
    try:
        state = await runner.run(request.repository_id, request.task)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_row = session.get(AgentRun, state.run_id)
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
    return [
        AgentEventOut(
            iteration=e.iteration,
            event_type=e.event_type,
            payload=json.loads(e.payload_json),
            created_at=e.created_at,
        )
        for e in events
    ]


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
