import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.execution.subprocess_runner import (
    CommandExecutionError,
    CommandNotAllowedError,
    run_command,
)
from app.tools.base import Tool, ToolError
from app.tools.repo_utils import get_repository
from app.tools.test_result_parser import classify_failure, parse_pytest_output


class RunTestsInput(BaseModel):
    repository_id: str
    test_path: str | None = None


class RunTestsOutput(BaseModel):
    command: str
    scope: Literal["full", "targeted"]
    exit_code: int
    passed: bool
    total_tests: int | None
    passed_tests: int | None
    failed_tests: list[str]
    failure_category: str
    duration_seconds: float
    stdout: str
    stderr: str


class RunTestsTool(Tool):
    """Runs the repository's configured test command and returns a
    structured result, not a raw terminal dump. `passed`/`failure_category`
    distinguish "the code under test is wrong" (test_failure) from "the run
    itself couldn't happen" (syntax_error, dependency_error,
    environment_error, timeout) — the agent should react very differently
    to those (fix the code vs. give up and report an environment problem).
    """

    name = "run_tests"
    description = (
        "Run the repository's test suite, or a specific test_path within it, and report structured results "
        "(pass/fail counts, failed test ids, and why a run failed if it wasn't a real test failure)."
    )
    input_schema = RunTestsInput
    output_schema = RunTestsOutput
    timeout_seconds = 120.0

    def __init__(self, session: Session, timeout_seconds: float | None = None):
        self._session = session
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds

    async def run(self, input_data: RunTestsInput) -> RunTestsOutput:
        repository = get_repository(self._session, input_data.repository_id)
        if not repository.test_command_json:
            raise ToolError(
                "No test command configured for this repository. Set one via "
                "IndexRepositoryRequest.test_command when indexing, or re-index a "
                "Python-majority repository to get an automatic default."
            )

        base_command = json.loads(repository.test_command_json)
        command = [*base_command, input_data.test_path] if input_data.test_path else base_command
        scope: Literal["full", "targeted"] = "targeted" if input_data.test_path else "full"

        try:
            result = await run_command(
                command, cwd=Path(repository.path), timeout_seconds=self.timeout_seconds, enforce_allowlist=True
            )
        except (CommandExecutionError, CommandNotAllowedError) as exc:
            raise ToolError(str(exc)) from exc

        total, passed_count, failed_tests = parse_pytest_output(result.stdout)
        return RunTestsOutput(
            command=" ".join(command),
            scope=scope,
            exit_code=result.exit_code,
            passed=result.exit_code == 0 and not result.timed_out,
            total_tests=total,
            passed_tests=passed_count,
            failed_tests=failed_tests,
            failure_category=classify_failure(result.exit_code, result.timed_out, result.stdout, result.stderr),
            duration_seconds=result.duration_seconds,
            stdout=result.stdout,
            stderr=result.stderr,
        )


class RunCommandInput(BaseModel):
    repository_id: str
    command: list[str]


class RunCommandOutput(BaseModel):
    command: str
    exit_code: int
    passed: bool
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool


class RunCommandTool(Tool):
    name = "run_command"
    description = "Run an allowlisted command (e.g. python, pytest, ruff, npm) in the repository."
    input_schema = RunCommandInput
    output_schema = RunCommandOutput
    timeout_seconds = 60.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: RunCommandInput) -> RunCommandOutput:
        repository = get_repository(self._session, input_data.repository_id)
        try:
            result = await run_command(
                input_data.command,
                cwd=Path(repository.path),
                timeout_seconds=self.timeout_seconds,
                enforce_allowlist=True,
            )
        except (CommandExecutionError, CommandNotAllowedError) as exc:
            raise ToolError(str(exc)) from exc

        return RunCommandOutput(
            command=" ".join(input_data.command),
            exit_code=result.exit_code,
            passed=result.exit_code == 0 and not result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
            timed_out=result.timed_out,
        )


class RunConfiguredCommandInput(BaseModel):
    repository_id: str


class RunConfiguredCommandOutput(BaseModel):
    command: str
    exit_code: int
    passed: bool
    stdout: str
    stderr: str
    duration_seconds: float


class _ConfiguredCommandTool(Tool):
    """Shared implementation for run_linter/run_formatter: both just run a
    repository-configured command and report a plain pass/fail result, no
    output parsing needed the way run_tests needs pytest-aware parsing."""

    input_schema = RunConfiguredCommandInput
    output_schema = RunConfiguredCommandOutput
    timeout_seconds = 60.0
    _config_field: str
    _missing_config_message: str

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: RunConfiguredCommandInput) -> RunConfiguredCommandOutput:
        repository = get_repository(self._session, input_data.repository_id)
        command_json = getattr(repository, self._config_field)
        if not command_json:
            raise ToolError(self._missing_config_message)
        command = json.loads(command_json)

        try:
            result = await run_command(
                command, cwd=Path(repository.path), timeout_seconds=self.timeout_seconds, enforce_allowlist=True
            )
        except (CommandExecutionError, CommandNotAllowedError) as exc:
            raise ToolError(str(exc)) from exc

        return RunConfiguredCommandOutput(
            command=" ".join(command),
            exit_code=result.exit_code,
            passed=result.exit_code == 0 and not result.timed_out,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_seconds=result.duration_seconds,
        )


class RunLinterTool(_ConfiguredCommandTool):
    name = "run_linter"
    description = "Run the repository's configured linter."
    _config_field = "lint_command_json"
    _missing_config_message = (
        "No lint command configured for this repository. Set one via "
        "IndexRepositoryRequest.lint_command when indexing."
    )


class RunFormatterTool(_ConfiguredCommandTool):
    name = "run_formatter"
    description = "Run the repository's configured code formatter (rewrites files in place)."
    _config_field = "format_command_json"
    _missing_config_message = (
        "No format command configured for this repository. Set one via "
        "IndexRepositoryRequest.format_command when indexing."
    )
