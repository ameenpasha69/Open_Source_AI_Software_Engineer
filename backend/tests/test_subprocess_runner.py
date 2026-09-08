import shutil

import pytest
from app.execution.subprocess_runner import (
    ALLOWED_COMMANDS,
    PYTHON_BINARY,
    CommandNotAllowedError,
    SandboxUnavailableError,
    build_sandbox_env,
    check_command_allowed,
    run_command,
)

_DOCKER_AVAILABLE = shutil.which("docker") is not None
_SANDBOX_IMAGE = "local-ai-softeng-sandbox:latest"
requires_docker = pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker is not installed on this machine")


def test_check_command_allowed_accepts_allowlisted_binary():
    check_command_allowed([PYTHON_BINARY, "-c", "pass"])  # must not raise


def test_check_command_allowed_rejects_disallowed_binary():
    with pytest.raises(CommandNotAllowedError, match="not an allowed command"):
        check_command_allowed(["rm", "-rf", "/"])


def test_check_command_allowed_checks_by_basename_not_full_path():
    check_command_allowed(["/usr/bin/python3", "-c", "pass"])  # must not raise


def test_check_command_allowed_rejects_empty_command():
    with pytest.raises(CommandNotAllowedError, match="Empty command"):
        check_command_allowed([])


def test_allowed_commands_excludes_destructive_binaries():
    for dangerous in ("rm", "sudo", "curl", "wget", "dd", "mkfs", "shutdown"):
        assert dangerous not in ALLOWED_COMMANDS


def test_build_sandbox_env_only_includes_passthrough_keys(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SOME_SECRET_API_KEY", "sk-super-secret")

    env = build_sandbox_env()

    assert env.get("PATH") == "/usr/bin"
    assert "SOME_SECRET_API_KEY" not in env


async def test_run_command_enforces_allowlist_when_requested(tmp_path):
    with pytest.raises(CommandNotAllowedError):
        await run_command(["rm", "-rf", "."], cwd=tmp_path, timeout_seconds=5.0, enforce_allowlist=True)


async def test_run_command_without_allowlist_flag_permits_any_binary(tmp_path):
    result = await run_command(["echo", "hi"], cwd=tmp_path, timeout_seconds=5.0)
    assert "hi" in result.stdout


async def test_run_command_does_not_leak_host_secret_into_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_SECRET_API_KEY", "sk-super-secret")
    result = await run_command(
        [PYTHON_BINARY, "-c", "import os; print(os.environ.get('SOME_SECRET_API_KEY', 'NOT_SET'))"],
        cwd=tmp_path,
        timeout_seconds=10.0,
    )
    assert "NOT_SET" in result.stdout
    assert "sk-super-secret" not in result.stdout


async def test_run_command_docker_backend_raises_when_docker_binary_missing(tmp_path, monkeypatch):
    async def _boom(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr("asyncio.create_subprocess_exec", _boom)

    with pytest.raises(SandboxUnavailableError, match="Docker is not installed"):
        await run_command(["echo", "hi"], cwd=tmp_path, timeout_seconds=5.0, sandbox_backend="docker")


@requires_docker
async def test_run_command_docker_backend_runs_the_command_in_a_container(tmp_path):
    result = await run_command(
        [PYTHON_BINARY, "-c", "print('hello from container')"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
    )

    assert result.exit_code == 0
    assert "hello from container" in result.stdout


@requires_docker
async def test_run_command_docker_backend_can_see_and_modify_the_mounted_directory(tmp_path):
    (tmp_path / "input.txt").write_text("42")

    result = await run_command(
        [PYTHON_BINARY, "-c", "open('output.txt', 'w').write(open('input.txt').read() + '!')"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
    )

    assert result.exit_code == 0
    assert (tmp_path / "output.txt").read_text() == "42!"


@requires_docker
async def test_run_command_docker_backend_blocks_network_access(tmp_path):
    result = await run_command(
        [PYTHON_BINARY, "-c", "import socket; socket.create_connection(('8.8.8.8', 53), timeout=2)"],
        cwd=tmp_path,
        timeout_seconds=30.0,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
    )

    assert result.exit_code != 0
    assert "Network is unreachable" in result.stderr


@requires_docker
async def test_run_command_docker_backend_raises_for_a_nonexistent_image(tmp_path):
    with pytest.raises(SandboxUnavailableError):
        await run_command(
            ["echo", "hi"],
            cwd=tmp_path,
            timeout_seconds=30.0,
            sandbox_backend="docker",
            docker_image="this-image-does-not-exist-xyz:latest",
        )


@requires_docker
async def test_run_command_docker_backend_kills_container_on_timeout(tmp_path):
    result = await run_command(
        [PYTHON_BINARY, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        timeout_seconds=1.0,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
    )

    assert result.timed_out is True


@requires_docker
async def test_memory_limit_is_a_hard_ceiling(tmp_path):
    """The configured memory limit must not be silently doubled by swap.

    Docker defaults --memory-swap to twice --memory when it is not set, so a
    "512m" container could hold 512m of RAM plus 512m of swap. A 900 MB
    allocation used to succeed under a 512m limit; it must now be OOM-killed
    (exit 137) instead.
    """
    result = await run_command(
        [PYTHON_BINARY, "-c", "x = bytearray(900 * 1024 * 1024)"],
        cwd=tmp_path,
        timeout_seconds=120,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
        docker_memory_limit="512m",
    )

    assert result.exit_code == 137, "allocation above the limit should be OOM-killed"


@requires_docker
async def test_allocation_under_the_memory_limit_still_runs(tmp_path):
    """The ceiling must not be so tight that ordinary work is killed."""
    result = await run_command(
        [PYTHON_BINARY, "-c", "x = bytearray(200 * 1024 * 1024); print('ok')"],
        cwd=tmp_path,
        timeout_seconds=120,
        sandbox_backend="docker",
        docker_image=_SANDBOX_IMAGE,
        docker_memory_limit="512m",
    )

    assert result.exit_code == 0
    assert "ok" in result.stdout
