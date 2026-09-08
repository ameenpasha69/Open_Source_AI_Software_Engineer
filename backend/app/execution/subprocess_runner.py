import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

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


class SandboxUnavailableError(Exception):
    """sandbox_backend="docker" was requested but Docker isn't usable. Never
    silently falls back to the subprocess backend — a caller that asked for
    container isolation and got plain-subprocess isolation instead without
    being told would have a false sense of the security boundary in place.
    """


@dataclass(frozen=True)
class SandboxSettings:
    """The subset of Settings that picks which run_command backend a tool
    uses and how it's configured — bundled so tools take one object instead
    of four individually-threaded constructor args.
    """

    backend: str = "subprocess"
    docker_image: str = "local-ai-softeng-sandbox:latest"
    docker_memory_limit: str = "512m"
    docker_cpu_limit: str = "1.0"

    def as_kwargs(self) -> dict[str, str]:
        return {
            "sandbox_backend": self.backend,
            "docker_image": self.docker_image,
            "docker_memory_limit": self.docker_memory_limit,
            "docker_cpu_limit": self.docker_cpu_limit,
        }


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
    sandbox_backend: str = "subprocess",
    docker_image: str = "local-ai-softeng-sandbox:latest",
    docker_memory_limit: str = "512m",
    docker_cpu_limit: str = "1.0",
) -> CommandResult:
    """Run a fixed argv command with a hard timeout, output truncation, an
    isolated environment, and cwd pinned to a specific directory.

    This is the one place in the codebase allowed to spawn a subprocess (or,
    for the docker backend, the one place allowed to shell out to the
    `docker` CLI) — callers pass an explicit argument list (never a shell
    string), so there's no shell-injection surface regardless of what ends
    up in individual arguments. `enforce_allowlist=True` additionally
    requires the binary be in ALLOWED_COMMANDS — used for tools that run
    agent/config-supplied commands (run_command, run_tests, run_linter,
    run_formatter), not for the fixed, hardcoded git subcommands the
    read-only git tools use.

    `sandbox_backend="docker"` runs the same command inside a throwaway,
    network-isolated container instead of a bare subprocess — see
    `_run_docker` for what that buys over the default.
    """
    if enforce_allowlist:
        check_command_allowed(command)

    if sandbox_backend == "docker":
        result = await _run_docker(
            command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            image=docker_image,
            memory_limit=docker_memory_limit,
            cpu_limit=docker_cpu_limit,
        )
    else:
        result = await _run_subprocess(command, cwd=cwd, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)

    logger.info(
        "sandboxed command executed",
        extra={
            "command": " ".join(command),
            "sandbox_backend": sandbox_backend,
            "cwd": str(cwd),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
        },
    )
    return result


async def _run_subprocess(
    command: list[str], *, cwd: Path, timeout_seconds: float, max_output_bytes: int
) -> CommandResult:
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


async def _run_docker(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    max_output_bytes: int,
    image: str,
    memory_limit: str,
    cpu_limit: str,
) -> CommandResult:
    """Runs `command` inside a `--rm`, `--network none` container with the
    target directory bind-mounted read-write at /workspace (read-write
    because the whole point — apply_patch, run_formatter — is editing files
    there; isolation is about the network and the rest of the host
    filesystem, not about protecting these particular files from writes).

    Networking is fully disabled, which means the image must already have
    every tool an allowlisted command might need baked in (see
    docker/sandbox.Dockerfile) — nothing can `pip install` at run time.

    Killing the local `docker run` client process on timeout does NOT stop
    the container; dockerd keeps it running server-side independently of
    its CLI. Each run gets a unique --name so a timeout can reliably
    `docker kill` the actual container, not just orphan it to keep burning
    CPU/memory until it finishes on its own.
    """
    container_name = f"aisofteng-sandbox-{uuid.uuid4().hex[:12]}"
    docker_command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        f"--memory={memory_limit}",
        # Without --memory-swap, Docker defaults it to twice --memory, so a
        # "512m" container can actually hold 512m of RAM plus 512m of swap.
        # Verified before this line existed: a 900 MB allocation succeeded
        # under a 512m limit. Setting the two equal disables swap and makes
        # the configured number the real ceiling -- a runaway process gets
        # OOM-killed rather than quietly using double what was asked for.
        f"--memory-swap={memory_limit}",
        f"--cpus={cpu_limit}",
        "-e",
        "PYTHONDONTWRITEBYTECODE=1",
        "-v",
        f"{cwd}:/workspace",
        "-w",
        "/workspace",
        image,
        *command,
    ]

    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
    except FileNotFoundError as exc:
        raise SandboxUnavailableError("Docker is not installed or not on PATH") from exc
    except OSError as exc:
        raise SandboxUnavailableError(f"Could not start docker: {exc}") from exc

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        timed_out = True
        await _force_kill_container(container_name)
        proc.kill()
        await proc.wait()
        stdout_bytes, stderr_bytes = b"", b""

    stdout = _truncate(stdout_bytes.decode("utf-8", errors="replace"), max_output_bytes)
    stderr = _truncate(stderr_bytes.decode("utf-8", errors="replace"), max_output_bytes)
    exit_code = -1 if timed_out else (proc.returncode or 0)

    if not timed_out and (proc.returncode == 125 or "Cannot connect to the Docker daemon" in stderr):
        # 125 is `docker run`'s own exit code for "the container never
        # started" (bad image, ...) — distinct from the containerized
        # command's own exit code, which docker forwards unchanged for
        # every other value. The daemon-unreachable case is checked
        # separately since the CLI reports that as exit 1, indistinguishable
        # by code alone from a real failing exit-1 test run. Both are
        # setup problems, not something the agent's command chose to
        # return, and reporting either as "the tests failed" would be a
        # false positive worth avoiding.
        raise SandboxUnavailableError(f"Docker sandbox could not start: {stderr.strip() or stdout.strip()}")

    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=round(time.monotonic() - started, 3),
        timed_out=timed_out,
    )


async def _force_kill_container(container_name: str) -> None:
    kill_proc = await asyncio.create_subprocess_exec(
        "docker", "kill", container_name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await kill_proc.wait()


def _truncate(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "\n... [output truncated]"
