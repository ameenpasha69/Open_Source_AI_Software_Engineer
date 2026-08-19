from pathlib import Path

from app.retrieval.file_walker import walk_repository


def _write(root: Path, relative: str, content: str = "x = 1\n") -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_walk_repository_finds_supported_source_files(tmp_path):
    _write(tmp_path, "app/main.py")
    _write(tmp_path, "app/utils.js")
    _write(tmp_path, "README.md")  # unsupported extension for this milestone

    found = {c.relative_path for c in walk_repository(tmp_path, max_file_size_bytes=10_000)}

    assert found == {"app/main.py", "app/utils.js"}


def test_walk_repository_skips_ignored_directories(tmp_path):
    _write(tmp_path, "src/real.py")
    _write(tmp_path, "node_modules/pkg/index.py")
    _write(tmp_path, ".git/hooks/pre-commit.py")
    _write(tmp_path, ".venv/lib/site.py")

    found = {c.relative_path for c in walk_repository(tmp_path, max_file_size_bytes=10_000)}

    assert found == {"src/real.py"}


def test_walk_repository_skips_files_over_size_limit(tmp_path):
    _write(tmp_path, "small.py", "x = 1\n")
    _write(tmp_path, "big.py", "x = 1\n" * 1000)

    found = {c.relative_path for c in walk_repository(tmp_path, max_file_size_bytes=100)}

    assert found == {"small.py"}


def test_walk_repository_assigns_language_by_extension(tmp_path):
    _write(tmp_path, "a.py")
    _write(tmp_path, "b.ts")

    by_path = {c.relative_path: c.language for c in walk_repository(tmp_path, max_file_size_bytes=10_000)}

    assert by_path == {"a.py": "python", "b.ts": "typescript"}
