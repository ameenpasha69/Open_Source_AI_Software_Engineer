import datetime
import json
import logging

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.agents.context_manager import ContextManager, format_observation
from app.agents.planner import Planner
from app.agents.state import AgentAction, AgentDecision, AgentState, AgentStatus, ToolCallRecord
from app.agents.termination import check_termination
from app.database.models import AgentEvent, AgentRun, AgentToolCall, Repository
from app.llm.base import LLMProvider, Message
from app.llm.exceptions import LLMProviderError
from app.retrieval.indexer import RepositoryNotFoundError
from app.tools.base import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)

_MAX_DECISION_PARSE_ATTEMPTS = 2


class AgentRunner:
    """The basic agent loop: TASK -> PLAN -> (OBSERVE -> SELECT TOOL ->
    EXECUTE -> UPDATE STATE)* -> DONE, persisted incrementally to SQLite so
    a run survives a process restart and is inspectable mid-flight via the
    API. This milestone's agent investigates and diagnoses — it can't modify
    code or run tests yet (Milestones 7/8 add those tools), so "DONE" here
    means "produced a cited diagnosis," not "fixed the bug."
    """

    def __init__(self, session: Session, llm: LLMProvider, tool_registry: ToolRegistry, max_iterations: int):
        self._session = session
        self._llm = llm
        self._tool_registry = tool_registry
        self._tool_executor = ToolExecutor(tool_registry)
        self._context_manager = ContextManager()
        self._planner = Planner(llm)
        self._max_iterations = max_iterations

    async def run(self, repository_id: str, task: str) -> AgentState:
        repository = self._session.get(Repository, repository_id)
        if repository is None:
            raise RepositoryNotFoundError(f"Repository '{repository_id}' not found")

        run_row = AgentRun(repository_id=repository_id, task=task, status=AgentStatus.RUNNING.value)
        self._session.add(run_row)
        self._session.flush()
        state = AgentState(run_id=run_row.id, task=task, repository_id=repository_id)

        try:
            state.plan = await self._planner.create_plan(task, repository.name)
            self._emit_event(run_row.id, 0, "plan_created", {"plan": state.plan})
            self._session.commit()

            while True:
                termination = check_termination(state, self._max_iterations)
                if termination is not None:
                    state.status = termination
                    break
                await self._run_iteration(run_row, state)
                self._session.commit()
        except LLMProviderError as exc:
            state.status = AgentStatus.FAILED
            state.error = f"LLM error: {exc}"
        except Exception as exc:
            logger.exception("Unexpected error in agent run %s", run_row.id)
            state.status = AgentStatus.FAILED
            state.error = f"Unexpected agent error: {exc}"

        self._persist_final(run_row, state)
        return state

    async def _run_iteration(self, run_row: AgentRun, state: AgentState) -> None:
        state.iteration += 1
        self._emit_event(run_row.id, state.iteration, "iteration_started", {})

        messages = self._context_manager.build_messages(state, self._tool_registry.list_specs(), self._max_iterations)
        decision = await self._get_decision(messages)

        if decision is None:
            observation = f"[iter {state.iteration}] LLM produced invalid/unparseable output; skipping iteration"
            state.observations.append(observation)
            self._emit_event(run_row.id, state.iteration, "invalid_decision", {})
            return

        self._emit_event(run_row.id, state.iteration, "thought", {"thought": decision.thought})

        if decision.finish is not None:
            state.status = AgentStatus.DONE
            state.final_answer = decision.finish.answer
            state.root_cause = decision.finish.root_cause
            self._emit_event(run_row.id, state.iteration, "finished", {"answer": decision.finish.answer})
            return

        await self._execute_action(run_row, state, decision.action)

    async def _execute_action(self, run_row: AgentRun, state: AgentState, action: AgentAction) -> None:
        self._emit_event(run_row.id, state.iteration, "tool_called", {"tool": action.tool, "input": action.input})

        # repository_id is always the run's own repository, never something
        # the LLM should have to (or be trusted to) supply — this also means
        # the agent can never be tricked into acting on a different repo.
        tool_input = {**action.input, "repository_id": state.repository_id}
        result = await self._tool_executor.execute(action.tool, tool_input)
        record = ToolCallRecord(iteration=state.iteration, tool_name=action.tool, input=action.input, result=result)
        state.tool_calls.append(record)
        state.observations.append(format_observation(record))
        self._persist_tool_call(run_row.id, record)

        self._emit_event(
            run_row.id, state.iteration, "tool_completed", {"tool": action.tool, "success": result.success}
        )

    async def _get_decision(self, messages: list[Message]) -> AgentDecision | None:
        for _ in range(_MAX_DECISION_PARSE_ATTEMPTS):
            response = await self._llm.generate(messages, json_mode=True)
            try:
                return AgentDecision.model_validate(json.loads(response.content))
            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning("Agent decision failed to parse, retrying: %s", exc)
                messages = [
                    *messages,
                    Message(role="assistant", content=response.content),
                    Message(
                        role="user",
                        content="That was not valid JSON matching the required schema. "
                        "Respond again with ONLY the JSON object.",
                    ),
                ]
        return None

    def _emit_event(self, run_id: str, iteration: int, event_type: str, payload: dict) -> None:
        self._session.add(
            AgentEvent(run_id=run_id, iteration=iteration, event_type=event_type, payload_json=json.dumps(payload))
        )

    def _persist_tool_call(self, run_id: str, record: ToolCallRecord) -> None:
        self._session.add(
            AgentToolCall(
                run_id=run_id,
                iteration=record.iteration,
                tool_name=record.tool_name,
                input_json=json.dumps(record.input),
                success=record.result.success,
                output_json=json.dumps(record.result.output) if record.result.output is not None else None,
                error=record.result.error,
                duration_seconds=record.result.duration_seconds,
            )
        )

    def _persist_final(self, run_row: AgentRun, state: AgentState) -> None:
        run_row.status = state.status.value
        run_row.plan_json = json.dumps(state.plan)
        run_row.final_answer = state.final_answer
        run_row.root_cause = state.root_cause
        run_row.iteration_count = state.iteration
        run_row.error = state.error
        run_row.finished_at = datetime.datetime.now(datetime.UTC)
        self._session.add(run_row)
        self._session.commit()
