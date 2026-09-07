import pytest
from app.retrieval.command_detection import detect_default_commands
from app.execution.subprocess_runner import PYTHON_BINARY
from app.retrieval.indexer import RepositoryIndexer


@pytest.fixture
def python_repo(tmp_path):
    repo = tmp_path / "python_repo"
    repo.mkdir()
    (repo / "a.py").write_text("def f():\n    pass\n")
    (repo / "b.py").write_text("def g():\n    pass\n")
    (repo / "c.js").write_text("function h() {}\n")
    return repo


@pytest.fixture
def js_repo(tmp_path):
    repo = tmp_path / "js_repo"
    repo.mkdir()
    (repo / "a.js").write_text("function f() {}\n")
    (repo / "b.js").write_text("function g() {}\n")
    (repo / "c.py").write_text("def h():\n    pass\n")
    return repo


def _index(session, repo_path):
    indexer = RepositoryIndexer(session=session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000)
    return indexer.index(repo_path)


def test_detects_python_defaults_for_python_majority_repo(db_session, python_repo):
    result = _index(db_session, python_repo)
    defaults = detect_default_commands(db_session, result.repository_id)

    assert defaults["test_command"] == [PYTHON_BINARY, "-m", "pytest"]
    assert defaults["lint_command"] == ["ruff", "check", "."]
    assert defaults["format_command"] == ["ruff", "format", "."]


def test_no_defaults_for_non_python_majority_repo(db_session, js_repo):
    result = _index(db_session, js_repo)
    defaults = detect_default_commands(db_session, result.repository_id)
    assert defaults == {}


def test_no_defaults_for_unindexed_repository(db_session):
    assert detect_default_commands(db_session, "does-not-exist") == {}
