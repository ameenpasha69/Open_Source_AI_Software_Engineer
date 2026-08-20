import asyncio
import time
from dataclasses import dataclass
from pathlib import Path


class CommandExecutionError(Exception):
    """The command could not even be started (e.g. the executable doesn't exist)."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool


async def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int = 1_000_000,
) -> CommandResult:
    """Run a fixed argv command with a hard timeout, output truncation, and
    cwd pinned to a specific directory.

    This is the one place in the codebase allowed to spawn a subprocess —
    callers pass an explicit argument list (never a shell string), so there's
    no shell-injection surface regardless of what ends up in individual
    arguments. It's deliberately minimal today (used only for read-only git
    commands); the sandboxing milestone extends this with an allowlist,
    environment isolation, and an optional Docker backend for the
    higher-risk tools (run_command, run_tests) that don't exist yet.
    """
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd),
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
