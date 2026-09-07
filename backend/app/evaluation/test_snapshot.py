import re
from pathlib import Path

from app.evaluation.models import TestSnapshot
from app.execution.subprocess_runner import PYTHON_BINARY, run_command

_VERBOSE_RESULT_LINE = re.compile(r"^(\S+::\S+)\s+(PASSED|FAILED|ERROR)\b")

_SNAPSHOT_TIMEOUT_SECONDS = 30.0


async def take_test_snapshot(repo_path: Path) -> TestSnapshot:
    """The evaluator's own ground-truth pytest run — `-v` so every test's
    pass/fail is explicit in the output (pytest's default output only lists
    *failures* by name; a plain "." per passing test isn't enough to build
    the passed_tests list a regression check needs). Independent of
    whatever the agent's own run_tests calls did or didn't do.
    """
    result = await run_command(
        [PYTHON_BINARY, "-m", "pytest", "-v"], cwd=repo_path, timeout_seconds=_SNAPSHOT_TIMEOUT_SECONDS
    )
    passed, failed = [], []
    for line in result.stdout.splitlines():
        match = _VERBOSE_RESULT_LINE.match(line)
        if not match:
            continue
        node_id, outcome = match.groups()
        (passed if outcome == "PASSED" else failed).append(node_id)

    return TestSnapshot(exit_code=result.exit_code, passed_tests=passed, failed_tests=failed)
