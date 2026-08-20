import re
from pathlib import Path

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.execution.subprocess_runner import CommandExecutionError, run_command
from app.tools.base import Tool, ToolError
from app.tools.repo_utils import get_repository_root

_GIT_TIMEOUT_SECONDS = 15.0


async def _run_git(repo_root: Path, args: list[str]) -> str:
    try:
        result = await run_command(["git", *args], cwd=repo_root, timeout_seconds=_GIT_TIMEOUT_SECONDS)
    except CommandExecutionError as exc:
        raise ToolError(f"git is not available: {exc}") from exc
    if result.timed_out:
        raise ToolError(f"git {' '.join(args)} timed out after {_GIT_TIMEOUT_SECONDS}s")
    if result.exit_code != 0:
        raise ToolError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


class GetGitStatusInput(BaseModel):
    repository_id: str


class GetGitStatusOutput(BaseModel):
    branch: str
    is_clean: bool
    staged: list[str]
    unstaged: list[str]
    untracked: list[str]


class GetGitStatusTool(Tool):
    name = "get_git_status"
    description = "Show the current git branch and working tree status (staged/unstaged/untracked files)."
    input_schema = GetGitStatusInput
    output_schema = GetGitStatusOutput
    timeout_seconds = _GIT_TIMEOUT_SECONDS + 5

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: GetGitStatusInput) -> GetGitStatusOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        output = await _run_git(repo_root, ["status", "--porcelain=v1", "--branch"])
        lines = output.splitlines()

        branch = "unknown"
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []

        for line in lines:
            if line.startswith("## "):
                branch = re.split(r"\.\.\.| \[", line[3:], maxsplit=1)[0]
                continue
            if line.startswith("??"):
                untracked.append(line[3:])
                continue
            index_status, worktree_status, path = line[0], line[1], line[3:]
            if index_status != " ":
                staged.append(path)
            if worktree_status != " ":
                unstaged.append(path)

        return GetGitStatusOutput(
            branch=branch,
            is_clean=not (staged or unstaged or untracked),
            staged=staged,
            unstaged=unstaged,
            untracked=untracked,
        )


class GetGitDiffInput(BaseModel):
    repository_id: str
    staged: bool = False
    path: str | None = None


class GetGitDiffOutput(BaseModel):
    diff: str


class GetGitDiffTool(Tool):
    name = "get_git_diff"
    description = "Show the git diff of working tree or staged changes, optionally scoped to one path."
    input_schema = GetGitDiffInput
    output_schema = GetGitDiffOutput
    timeout_seconds = _GIT_TIMEOUT_SECONDS + 5

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: GetGitDiffInput) -> GetGitDiffOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        args = ["diff"]
        if input_data.staged:
            args.append("--staged")
        if input_data.path:
            args.extend(["--", input_data.path])
        diff = await _run_git(repo_root, args)
        return GetGitDiffOutput(diff=diff)


class CommitInfo(BaseModel):
    commit_hash: str
    author: str
    date: str
    message: str


class GetGitLogInput(BaseModel):
    repository_id: str
    max_count: int = 10
    path: str | None = None


class GetGitLogOutput(BaseModel):
    commits: list[CommitInfo]


class GetGitLogTool(Tool):
    name = "get_git_log"
    description = "Show recent commit history, optionally scoped to one path."
    input_schema = GetGitLogInput
    output_schema = GetGitLogOutput
    timeout_seconds = _GIT_TIMEOUT_SECONDS + 5

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: GetGitLogInput) -> GetGitLogOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        args = [
            "log",
            f"-n{max(input_data.max_count, 1)}",
            "--pretty=format:%H%x1f%an%x1f%aI%x1f%s%x1e",
        ]
        if input_data.path:
            args.extend(["--", input_data.path])
        output = await _run_git(repo_root, args)

        commits = []
        for record in filter(None, output.split("\x1e")):
            commit_hash, author, date, message = record.strip("\n").split("\x1f")
            commits.append(CommitInfo(commit_hash=commit_hash, author=author, date=date, message=message))
        return GetGitLogOutput(commits=commits)
