from pathlib import Path

from app.tools.base import ToolError


def resolve_safe_path(repo_root: Path, relative_path: str) -> Path:
    """Resolve `relative_path` against `repo_root` and refuse anything that
    escapes it (`../../etc/passwd`, an absolute path elsewhere, a symlink
    pointing outside the repo). This is the one check every tool that takes
    a file path must run before touching the filesystem — the agent must
    never be able to read or write outside the repository it was given.
    """
    repo_root = repo_root.resolve()
    candidate = (repo_root / relative_path).resolve()
    try:
        candidate.relative_to(repo_root)
    except ValueError as exc:
        raise ToolError(f"Path '{relative_path}' escapes the repository root") from exc
    return candidate
