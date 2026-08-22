import asyncio
import datetime
import json
import logging
import time
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.context_manager import ContextManager, format_observation
from app.agents.planner import Planner
from app.agents.repetition import describe_call, find_redundant_call
from app.agents.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AgentStatus,
    ConversationTurn,
    TestRunRecord,
    ToolCallRecord,
)
from app.agents.termination import check_termination
from app.config.settings import Settings
from app.database.models import (
    AgentEvent,
    AgentRun,
    AgentToolCall,
    ChatMessage,
    ChatSession,
    ModifiedFile,
    Repository,
    TestRun,
)
from app.embeddings.base import EmbeddingProvider
from app.llm.base import LLMProvider, Message
from app.llm.exceptions import LLMProviderError
from app.llm.usage import UsageTrackingLLMProvider
from app.observability.logging import run_id_var
from app.retrieval.indexer import RepositoryNotFoundError
from app.retrieval.reindex import reindex_repository
from app.tools.base import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)

_MAX_DECISION_PARSE_ATTEMPTS = 2

_NO_ANSWER_FALLBACK = {
    AgentStatus.MAX_ITERATIONS_REACHED: "I ran out of iterations before reaching a conclusion.",
    AgentStatus.NO_PROGRESS: (
        "I stopped because I was repeating myself instead of making progress. "
        "If code search kept coming back empty, the repository may need re-indexing."
    ),
    AgentStatus.CANCELLED: "This run was cancelled.",
    AgentStatus.FAILED: "This run failed before I could reach a conclusion.",
    AgentStatus.RUNNING: "This run did not complete.",
    AgentStatus.DONE: "This run finished without an answer.",
}


class SessionNotFoundError(Exception):
    pass


# How much to raise decoding temperature per consecutive redundant call, and
# the ceiling on how high that climbs.
_REDUNDANCY_TEMPERATURE_STEP = 0.25
_MAX_REDUNDANCY_TEMPERATURE = 0.9


def _decision_temperature(state: AgentState) -> float | None:
    """None defers to the provider's configured default (kept low — this
    task rewards precision, not creativity) for an ordinary iteration. But
    low temperature is *why* a stuck model stays stuck: observed live
    (qwen2.5-coder:7b, a rename+rewrite task) — after a call got skipped as
    redundant, the corrective observation was appended to the prompt exactly
    as designed, and the model's next response was still, word for word, the
    same thought and the same tool call. At temperature 0.2 with grammar-
    constrained JSON decoding, a few new lines appended near the end of an
    otherwise-unchanged, fairly long prompt often isn't enough to move the
    argmax path — the corrective text was present and correct, and still
    couldn't win. Raising temperature specifically once a repeat has already
    happened gives the model an actual chance to sample a different
    continuation, without touching the default the rest of the time.
    """
    if state.consecutive_redundant_calls == 0:
        return None
    return min(
        _REDUNDANCY_TEMPERATURE_STEP * (state.consecutive_redundant_calls + 1), _MAX_REDUNDANCY_TEMPERATURE
    )


class AgentRunner:
    """The basic agent loop: TASK -> PLAN -> (OBSERVE -> SELECT TOOL ->
    EXECUTE -> UPDATE STATE)* -> DONE, persisted incrementally to SQLite so
    a run survives a process restart and is inspectable mid-flight via the
    API. The agent can modify code (apply_patch) and run tests (run_tests)
    to check its own work — self-correction (iterating after a test
    failure) isn't special-cased anywhere: it emerges from the same loop,
    since run_tests is just another tool the model can choose to call again
    after seeing a failure.
    """

    def __init__(
        self,
        session: Session,
        llm: LLMProvider,
        tool_registry: ToolRegistry,
        max_iterations: int,
        settings: Settings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self._session = session
        # Wrapped once here so the planner and the loop feed the same counter
        # without either of them knowing they're metered.
        self._llm = UsageTrackingLLMProvider(llm)
        self._tool_registry = tool_registry
        self._tool_executor = ToolExecutor(tool_registry)
        self._context_manager = ContextManager()
        self._planner = Planner(self._llm)
        self._max_iterations = max_iterations
        # Both optional, and both required together, to re-index after a
        # write (see _reindex_after_write) — every real caller has them, but
        # tests that only exercise read-only tools shouldn't need to supply
        # an embedding provider just to construct a runner.
        self._settings = settings
        self._embedding_provider = embedding_provider

    def create_run(self, repository_id: str, task: str, session_id: str | None = None) -> AgentRun:
        """Creates and commits the AgentRun row synchronously, before any LLM
        call — so a caller (the API route) has a real run_id to return to the
        client immediately, whether or not `execute()` then runs inline or is
        handed off to a background task."""
        repository = self._session.get(Repository, repository_id)
        if repository is None:
            raise RepositoryNotFoundError(f"Repository '{repository_id}' not found")
        if session_id is not None and self._session.get(ChatSession, session_id) is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found")
        run_row = AgentRun(
            repository_id=repository_id,
            session_id=session_id,
            task=task,
            status=AgentStatus.RUNNING.value,
            model=self._llm.model,
        )
        self._session.add(run_row)
        self._session.commit()
        return run_row

    async def run(self, repository_id: str, task: str, session_id: str | None = None) -> AgentState:
        """Create the run and execute it inline, end to end. Used directly by
        tests and by any caller that wants to simply await the whole thing."""
        run_row = self.create_run(repository_id, task, session_id)
        return await self.execute(run_row.id, repository_id, task)

    async def execute(self, run_id: str, repository_id: str, task: str) -> AgentState:
        """Runs the loop for an already-created AgentRun row. Cancellation
        (asyncio.CancelledError, raised into this coroutine by the caller
        cancelling the asyncio.Task it's running in) is caught so the run's
        final status is persisted as "cancelled" instead of leaving the row
        stuck at "running" forever — then re-raised, since the task really
        is being cancelled and callers awaiting it need to see that.
        """
        run_row = self._session.get(AgentRun, run_id)
        repository = self._session.get(Repository, repository_id)
        state = AgentState(
            run_id=run_id,
            task=task,
            repository_id=repository_id,
            conversation=self._load_conversation(run_row),
            model=self._llm.model,
        )

        token = run_id_var.set(run_id)
        started = time.monotonic()
        try:
            logger.info("agent run started", extra={"repository_id": repository_id, "task": task})
            try:
                state.plan = await self._planner.create_plan(task, repository.name)
                self._emit_event(run_row.id, 0, "plan_created", {"plan": state.plan})
                self._session.commit()

                while True:
                    termination = check_termination(state, self._max_iterations)
                    if termination is not None:
                        state.status = termination
                        if termination is AgentStatus.NO_PROGRESS:
                            state.error = (
                                f"Stopped after {state.consecutive_redundant_calls} consecutive tool calls "
                                "that had already been made against an unchanged repository — the agent was "
                                "repeating itself rather than converging."
                            )
                        break
                    await self._run_iteration(run_row, state)
                    self._session.commit()
            except asyncio.CancelledError:
                state.status = AgentStatus.CANCELLED
                self._persist_final(run_row, state)
                logger.info("agent run cancelled", extra={"duration_seconds": round(time.monotonic() - started, 3)})
                raise
            except LLMProviderError as exc:
                state.status = AgentStatus.FAILED
                state.error = f"LLM error: {exc}"
            except Exception as exc:
                logger.exception("Unexpected error in agent run %s", run_row.id)
                state.status = AgentStatus.FAILED
                state.error = f"Unexpected agent error: {exc}"

            logger.info(
                "agent run finished",
                extra={
                    "status": state.status.value,
                    "iterations": state.iteration,
                    "tool_calls": len(state.tool_calls),
                    "duration_seconds": round(time.monotonic() - started, 3),
                },
            )
            self._persist_final(run_row, state)
            return state
        finally:
            run_id_var.reset(token)

    def _load_conversation(self, run_row: AgentRun) -> list[ConversationTurn]:
        """Earlier turns of this run's session, if it has one.

        Deliberately excludes the user message that *started* this run — that
        text is already the task, and repeating it as conversation history
        would have the model treating its own current instruction as
        something previously agreed.
        """
        if run_row.session_id is None:
            return []
        messages = self._session.scalars(
            select(ChatMessage)
            .where(ChatMessage.session_id == run_row.session_id)
            .order_by(ChatMessage.created_at)
        ).all()
        return [
            ConversationTurn(role=m.role, content=m.content)
            for m in messages
            if m.run_id != run_row.id and m.content.strip()
        ]

    async def _run_iteration(self, run_row: AgentRun, state: AgentState) -> None:
        state.iteration += 1
        self._emit_event(run_row.id, state.iteration, "iteration_started", {})

        messages = self._context_manager.build_messages(state, self._tool_registry.list_specs(), self._max_iterations)
        decision, last_error = await self._get_decision(messages, temperature=_decision_temperature(state))

        if decision is None:
            # Carries the actual validation error into next iteration's
            # observations (not just "invalid output, skipping") — otherwise
            # a systematic mistake (e.g. sending a bare list where the
            # schema wants {"command": [...]}) has no persisted memory
            # across iterations and can repeat indefinitely until the run
            # burns its whole iteration budget on the same wrong shape.
            observation = f"[iter {state.iteration}] Response rejected, iteration skipped: {last_error}"
            state.observations.append(observation)
            self._emit_event(run_row.id, state.iteration, "invalid_decision", {"error": last_error})
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

        repeat_of = find_redundant_call(state, action.tool, action.input)
        if repeat_of is not None:
            self._skip_redundant_call(run_row, state, action, repeat_of)
            return
        state.consecutive_redundant_calls = 0

        # repository_id is always the run's own repository, never something
        # the LLM should have to (or be trusted to) supply — this also means
        # the agent can never be tricked into acting on a different repo.
        tool_input = {**action.input, "repository_id": state.repository_id}
        result = await self._tool_executor.execute(action.tool, tool_input)
        record = ToolCallRecord(iteration=state.iteration, tool_name=action.tool, input=action.input, result=result)
        state.tool_calls.append(record)
        state.observations.append(format_observation(record))
        self._persist_tool_call(run_row.id, record)

        if action.tool in ("apply_patch", "create_file", "delete_file") and result.success:
            self._record_modified_file(run_row.id, state, result.output)
            await self._reindex_after_write(state)
        elif action.tool == "run_tests" and result.success:
            self._record_test_run(run_row.id, state, result.output)

        self._emit_event(
            run_row.id, state.iteration, "tool_completed", {"tool": action.tool, "success": result.success}
        )

    async def _reindex_after_write(self, state: AgentState) -> None:
        """Keeps the agent's own search_code/find_symbol in sync with what it
        just wrote — otherwise a file the agent itself just created or
        changed is invisible to its own tools until someone re-indexes by
        hand. Observed live, repeatedly: a model asked to act on a file it
        (or an earlier run) had just written concluded the file "does not
        exist" purely because search_code came back empty, and never
        reached for list_files/read_file to check directly. Re-indexing
        after every write doesn't fix that reasoning mistake on its own —
        see the system prompt for that half — but it does mean search stops
        being reliably wrong immediately after the one thing most likely to
        make it wrong.

        Best-effort: a re-index failure (e.g. the embedding backend is
        unreachable) must never fail the run over it. The write itself
        already succeeded and is what's persisted; search staying stale is a
        degraded, recoverable state, not a reason to abort.
        """
        if self._settings is None or self._embedding_provider is None:
            return
        repository = self._session.get(Repository, state.repository_id)
        if repository is None:
            return
        try:
            await reindex_repository(self._session, self._settings, self._embedding_provider, Path(repository.path))
        except Exception:
            logger.warning("re-index after write failed", exc_info=True)

    def _skip_redundant_call(
        self, run_row: AgentRun, state: AgentState, action: AgentAction, repeat_of: int
    ) -> None:
        """Answer a call the agent has already made, without spending it.

        The point is the observation: telling the model *which* iteration
        already answered this, and that nothing has changed since, is what
        breaks the loop — silently re-running the tool just feeds it the same
        result and it asks again. Not executing also means no ToolCallRecord,
        so the original call stays the single record of that result.
        """
        state.consecutive_redundant_calls += 1
        observation = (
            f"[iter {state.iteration}] {describe_call(action.tool, action.input)} -> SKIPPED: "
            f"identical to the call you already made at iteration {repeat_of}, and nothing in the "
            f"repository has changed since, so the result would be identical. Repeating it cannot "
            f"tell you anything new. Do something different: apply a fix with apply_patch, look at "
            f"a different file, or finish with what you already know."
        )
        state.observations.append(observation)
        self._emit_event(
            run_row.id,
            state.iteration,
            "redundant_call_skipped",
            {"tool": action.tool, "input": action.input, "first_called_at_iteration": repeat_of},
        )
        logger.warning(
            "skipped redundant tool call",
            extra={
                "tool": action.tool,
                "iteration": state.iteration,
                "first_called_at_iteration": repeat_of,
                "consecutive_redundant_calls": state.consecutive_redundant_calls,
            },
        )

    def _record_modified_file(self, run_id: str, state: AgentState, output: dict) -> None:
        state.modified_files.append(output["path"])
        self._session.add(
            ModifiedFile(
                run_id=run_id,
                relative_path=output["path"],
                diff=output["diff"],
                lines_added=output["lines_added"],
                lines_removed=output["lines_removed"],
            )
        )

    def _record_test_run(self, run_id: str, state: AgentState, output: dict) -> None:
        state.test_results.append(
            TestRunRecord(
                command=output["command"],
                scope=output["scope"],
                passed=output["passed"],
                failed_tests=output["failed_tests"],
                failure_category=output["failure_category"],
            )
        )
        self._session.add(
            TestRun(
                repository_id=state.repository_id,
                run_id=run_id,
                command=output["command"],
                scope=output["scope"],
                exit_code=output["exit_code"],
                passed=output["passed"],
                total_tests=output["total_tests"],
                passed_tests=output["passed_tests"],
                failed_tests_json=json.dumps(output["failed_tests"]),
                failure_category=output["failure_category"],
                duration_seconds=output["duration_seconds"],
                stdout=output["stdout"],
                stderr=output["stderr"],
            )
        )

    async def _get_decision(
        self, messages: list[Message], temperature: float | None = None
    ) -> tuple[AgentDecision | None, str | None]:
        last_error: str | None = None
        for _ in range(_MAX_DECISION_PARSE_ATTEMPTS):
            response = await self._llm.generate(messages, temperature=temperature, json_mode=True)
            try:
                return AgentDecision.model_validate(json.loads(response.content)), None
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
                logger.warning("Agent decision failed to parse, retrying: %s", exc)
                messages = [
                    *messages,
                    Message(role="assistant", content=response.content),
                    Message(
                        role="user",
                        # Includes the actual validation error (not just "invalid
                        # JSON") — a wrong-shape mistake like sending a bare list
                        # for "input" needs the specific complaint to have any
                        # chance of being corrected on retry, not a generic nudge.
                        content=f"That response was rejected: {exc}\n\n"
                        "Respond again with ONLY a corrected JSON object matching the schema exactly.",
                    ),
                ]
        return None, last_error

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
        state.usage = self._llm.usage
        run_row.status = state.status.value
        run_row.prompt_tokens = state.usage.prompt_tokens
        run_row.completion_tokens = state.usage.completion_tokens
        run_row.llm_call_count = state.usage.call_count
        run_row.peak_prompt_tokens = state.usage.peak_prompt_tokens
        run_row.model = state.model
        run_row.plan_json = json.dumps(state.plan)
        run_row.final_answer = state.final_answer
        run_row.root_cause = state.root_cause
        run_row.iteration_count = state.iteration
        run_row.verification_status = state.verification_status
        run_row.error = state.error
        run_row.finished_at = datetime.datetime.now(datetime.UTC)
        self._session.add(run_row)
        self._record_assistant_turn(run_row, state)
        self._session.commit()

    def _record_assistant_turn(self, run_row: AgentRun, state: AgentState) -> None:
        """Write this run's outcome back into its session's transcript.

        Every terminal status gets a turn, not just a successful one — a
        session where the failed turns are missing reads as though they never
        happened, and the next turn's context would be quietly wrong.
        """
        if run_row.session_id is None:
            return
        existing = self._session.scalars(
            select(ChatMessage).where(ChatMessage.run_id == run_row.id)
        ).first()
        content = state.final_answer or state.error or _NO_ANSWER_FALLBACK[state.status]
        if existing is not None:
            existing.content = content
            return
        self._session.add(
            ChatMessage(
                session_id=run_row.session_id, role="assistant", content=content, run_id=run_row.id
            )
        )
