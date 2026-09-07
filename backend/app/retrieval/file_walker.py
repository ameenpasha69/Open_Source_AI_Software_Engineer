from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Directories that are never worth indexing: VCS internals, dependency trees,
# build output, caches. Matched by directory *name*, at any depth.
IGNORED_DIR_NAMES = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        ".next",
        "target",
        ".idea",
        ".vscode",
        "coverage",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".tox",
        "vendor",
        ".egg-info",
        "site-packages",
    }
)

# Extension -> language name. Python gets AST-based chunking; everything else
# falls back to generic sliding-window chunking (see retrieval/chunker.py).
EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".kt": "kotlin",
    ".swift": "swift",
}


@dataclass(frozen=True)
class CandidateFile:
    absolute_path: Path
    relative_path: str
    language: str
    size_bytes: int


def walk_repository(root: Path, max_file_size_bytes: int) -> Iterator[CandidateFile]:
    """Yield indexable source files under `root`, skipping ignored directories,
    unsupported extensions, and files over the size limit."""
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES or part.endswith(".egg-info") for part in path.relative_to(root).parts[:-1]):
            continue
        language = EXTENSION_LANGUAGE_MAP.get(path.suffix)
        if language is None:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > max_file_size_bytes:
            continue
        yield CandidateFile(
            absolute_path=path,
            # as_posix(), not str(): on Windows str() yields 'app\main.py'.
            # These paths are stored in the index, matched against what the
            # agent passes to its file tools, and shown in search results, so
            # a backslash here makes an index built on Windows disagree with
            # one built anywhere else -- and with the model, which writes
            # forward slashes regardless of host.
            relative_path=path.relative_to(root).as_posix(),
            language=language,
            size_bytes=size,
        )
