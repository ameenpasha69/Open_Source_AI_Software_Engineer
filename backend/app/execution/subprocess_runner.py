import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path

# Environment variables passed through to sandboxed subprocesses. Everything
# else from the host process's environment is dropped by default — a stray
# API key or cloud credential in the parent shell must never leak into a
# command the agent runs inside a repository it doesn't control the code of.
# PATH/HOME/LANG make a normal toolchain (python, node, git, ...) work;
# VIRTUAL_ENV/CONDA_PREFIX/PYTHONPATH respect an already-active environment
# for the target repository rather than silently ignoring it.
_ENV_PASSTHROUGH_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
)

# Binaries the sandbox will execute at all. Not a guarantee those binaries
# are themselves safe to run unsupervised — it's a floor, not a ceiling:
# anything not in this list (rm, sudo, curl, dd, mkfs, ...) is refused
# outright, regardless of arguments. Extend deliberately, not reflexively.
ALLOWED_COMMANDS = frozenset(
    {
        "python",
        "python3",
        "pip",
        "pip3",
        "pytest",
        "ruff",
        "black",
        "flake8",
        "mypy",
        "npm",
        "npx",
        "yarn",
        "pnpm",
        "node",
        "go",
        "cargo",
        "mvn",
        "gradle",
        "git",
    }
)


class CommandExecutionError(Exception):
    """The command could not even be started (e.g. the executable doesn't exist)."""


class CommandNotAllowedError(Exception):
    """The requested command's binary isn't in ALLOWED_COMMANDS."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool


def build_sandbox_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_PASSTHROUGH_KEYS if key in os.environ}
    # CPython's default .pyc cache invalidation compares source mtime at
    # *second* granularity. The agent can plausibly patch a file and re-run
    # tests within the same wall-clock second — without this, that re-run
    # can silently execute stale bytecode from before the fix, breaking
    # self-correction. Found by actually re-running the loop, not by
    # inspection: a first "fixed" run still failed with the pre-patch result.
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def check_command_allowed(command: list[str]) -> None:
    if not command:
        raise CommandNotAllowedError("Empty command")
    binary = Path(command[0]).name  # reject by binary name even if given as a path
    if binary not in ALLOWED_COMMANDS:
        raise CommandNotAllowedError(
            f"'{binary}' is not an allowed command. Allowed: {sorted(ALLOWED_COMMANDS)}"
        )


async def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int = 1_000_000,
    enforce_allowlist: bool = False,
) -> CommandResult:
    """Run a fixed argv command with a hard timeout, output truncation, an
    isolated environment, and cwd pinned to a specific directory.

    This is the one place in the codebase allowed to spawn a subprocess —
    callers pass an explicit argument list (never a shell string), so there's
    no shell-injection surface regardless of what ends up in individual
    arguments. `enforce_allowlist=True` additionally requires the binary be
    in ALLOWED_COMMANDS — used for tools that run agent/config-supplied
    commands (run_command, run_tests, run_linter, run_formatter), not for
    the fixed, hardcoded git subcommands the read-only git tools use.

    Deliberately local-process sandboxing, not container-based — a Docker
    backend is scoped separately (Milestone 11) rather than bolted on here.
    """
    if enforce_allowlist:
        check_command_allowed(command)

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
            env=build_sandbox_env(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CommandExecutionError(f"Command not found: {command[0]}") from exc
    except OSError as exc:
        raise CommandExecutionError(f"Could not start command {command}: {exc}") from exc

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        proc.kill()
        await proc.wait()
        stdout_bytes, stderr_bytes = b"", b""

    return CommandResult(
        command=command,
        exit_code=-1 if timed_out else (proc.returncode or 0),
        stdout=_truncate(stdout_bytes.decode("utf-8", errors="replace"), max_output_bytes),
        stderr=_truncate(stderr_bytes.decode("utf-8", errors="replace"), max_output_bytes),
        duration_seconds=round(time.monotonic() - started, 3),
        timed_out=timed_out,
    )


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n... [output truncated]"
