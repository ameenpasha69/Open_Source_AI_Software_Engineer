import asyncio
import contextlib
from collections.abc import Coroutine
from typing import Any


class BackgroundAgentRunner:
    """Tracks in-flight agent-run asyncio.Tasks by run_id, so they can be
    awaited (tests; a clean answer to "is this actually still running") or
    cancelled (POST /api/agent/{run_id}/cancel) without a separate worker
    process or queue — reasonable for this single-process local app. A
    consequence worth being upfront about: a run's cancellability doesn't
    survive a server restart, since the tracking is in-memory only.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def launch(self, run_id: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))
        return task

    def get_task(self, run_id: str) -> asyncio.Task | None:
        return self._tasks.get(run_id)

    async def cancel(self, run_id: str) -> bool:
        """Requests cancellation and waits for the task to actually stop
        (and persist a final status) before returning, so callers don't
        observe a "cancelled" response while the run is still writing to
        the DB. Returns False if there's nothing to cancel — already
        finished, or not tracked in this process."""
        task = self._tasks.get(run_id)
        if task is None or task.done():
            return False
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return True


_background_runner = BackgroundAgentRunner()


def get_background_agent_runner() -> BackgroundAgentRunner:
    return _background_runner
