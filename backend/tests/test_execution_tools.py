import json
import shutil

import pytest
from app.database.models import Repository
from app.execution.subprocess_runner import PYTHON_BINARY, SandboxSettings
from app.tools.base import ToolError
from app.tools.execution_tools import (
    RunCommandInput,
    RunCommandTool,
    RunConfiguredCommandInput,
    RunFormatterTool,
    RunLinterTool,
    RunTestsInput,
    RunTestsTool,
)

requires_docker = pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is not installed on this machine")


@pytest.fixture
def test_repo(tmp_path):
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "test_math.py").write_text(
        "def add(a, b):\n    return a + b\n\n\n"
        "def test_add_passes():\n    assert add(1, 2) == 3\n\n\n"
        "def test_add_fails():\n    assert add(1, 2) == 4\n"
    )
    return repo


@pytest.fixture
def repository_with_test_command(db_session, test_repo):
    repository = Repository(
        name="test_repo",
        path=str(test_repo),
        test_command_json=json.dumps([PYTHON_BINARY, "-m", "pytest"]),
        lint_command_json=json.dumps([PYTHON_BINARY, "-c", "print('lint ok')"]),
        format_command_json=json.dumps([PYTHON_BINARY, "-c", "print('format ok')"]),
    )
    db_session.add(repository)
    db_session.flush()
    return repository


@pytest.fixture
def repository_without_test_command(db_session, test_repo):
    repository = Repository(name="test_repo", path=str(test_repo))
    db_session.add(repository)
    db_session.flush()
    return repository


async def test_run_tests_reports_structured_failure(db_session, repository_with_test_command):
    tool = RunTestsTool(db_session, timeout_seconds=30.0)
    result = await tool.run(RunTestsInput(repository_id=repository_with_test_command.id))

    assert result.scope == "full"
    assert result.passed is False
    assert result.failure_category == "test_failure"
    assert result.total_tests == 2
    assert result.passed_tests == 1
    assert result.failed_tests == ["test_math.py::test_add_fails"]


async def test_run_tests_targeted_scope_runs_only_named_test(db_session, repository_with_test_command):
    tool = RunTestsTool(db_session, timeout_seconds=30.0)
    result = await tool.run(
        RunTestsInput(repository_id=repository_with_test_command.id, test_path="test_math.py::test_add_passes")
    )

    assert result.scope == "targeted"
    assert result.passed is True
    assert result.total_tests == 1


async def test_run_tests_raises_when_no_command_configured(db_session, repository_without_test_command):
    tool = RunTestsTool(db_session, timeout_seconds=30.0)
    with pytest.raises(ToolError, match="No test command configured"):
        await tool.run(RunTestsInput(repository_id=repository_without_test_command.id))


async def test_run_tests_passes_after_fixing_the_bug(db_session, repository_with_test_command, test_repo):
    (test_repo / "test_math.py").write_text(
        "def add(a, b):\n    return a + b\n\n\ndef test_add_passes():\n    assert add(1, 2) == 3\n"
    )
    tool = RunTestsTool(db_session, timeout_seconds=30.0)
    result = await tool.run(RunTestsInput(repository_id=repository_with_test_command.id))

    assert result.passed is True
    assert result.failure_category == "none"
    assert result.failed_tests == []


async def test_run_command_rejects_disallowed_binary(db_session, repository_with_test_command):
    tool = RunCommandTool(db_session)
    with pytest.raises(ToolError, match="not an allowed command"):
        await tool.run(RunCommandInput(repository_id=repository_with_test_command.id, command=["rm", "-rf", "."]))


async def test_run_command_runs_allowed_binary(db_session, repository_with_test_command):
    tool = RunCommandTool(db_session)
    result = await tool.run(
        RunCommandInput(repository_id=repository_with_test_command.id, command=[PYTHON_BINARY, "-c", "print('hi')"])
    )
    assert result.passed is True
    assert "hi" in result.stdout


async def test_run_linter_uses_configured_command(db_session, repository_with_test_command):
    tool = RunLinterTool(db_session)
    result = await tool.run(RunConfiguredCommandInput(repository_id=repository_with_test_command.id))
    assert result.passed is True
    assert "lint ok" in result.stdout


async def test_run_linter_raises_when_not_configured(db_session, repository_without_test_command):
    tool = RunLinterTool(db_session)
    with pytest.raises(ToolError, match="No lint command configured"):
        await tool.run(RunConfiguredCommandInput(repository_id=repository_without_test_command.id))


async def test_run_formatter_uses_configured_command(db_session, repository_with_test_command):
    tool = RunFormatterTool(db_session)
    result = await tool.run(RunConfiguredCommandInput(repository_id=repository_with_test_command.id))
    assert result.passed is True
    assert "format ok" in result.stdout


@requires_docker
async def test_run_tests_via_docker_sandbox_reports_the_same_structured_result(
    db_session, repository_with_test_command
):
    docker_sandbox = SandboxSettings(backend="docker", docker_image="local-ai-softeng-sandbox:latest")
    tool = RunTestsTool(db_session, timeout_seconds=30.0, sandbox=docker_sandbox)

    result = await tool.run(RunTestsInput(repository_id=repository_with_test_command.id))

    assert result.passed is False
    assert result.failure_category == "test_failure"
    assert result.failed_tests == ["test_math.py::test_add_fails"]


@requires_docker
async def test_run_command_via_docker_sandbox_cannot_reach_the_network(db_session, repository_with_test_command):
    docker_sandbox = SandboxSettings(backend="docker", docker_image="local-ai-softeng-sandbox:latest")
    tool = RunCommandTool(db_session, sandbox=docker_sandbox)

    result = await tool.run(
        RunCommandInput(
            repository_id=repository_with_test_command.id,
            command=[PYTHON_BINARY, "-c", "import socket; socket.create_connection(('8.8.8.8', 53), timeout=2)"],
        )
    )

    assert result.passed is False
    assert "Network is unreachable" in result.stderr
