import pytest
from app.execution.subprocess_runner import (
    ALLOWED_COMMANDS,
    CommandNotAllowedError,
    build_sandbox_env,
    check_command_allowed,
    run_command,
)


def test_check_command_allowed_accepts_allowlisted_binary():
    check_command_allowed(["python3", "-c", "pass"])  # must not raise


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
        ["python3", "-c", "import os; print(os.environ.get('SOME_SECRET_API_KEY', 'NOT_SET'))"],
        cwd=tmp_path,
        timeout_seconds=10.0,
    )
    assert "NOT_SET" in result.stdout
    assert "sk-super-secret" not in result.stdout
