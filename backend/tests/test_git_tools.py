import subprocess

import pytest
from app.database.models import Repository
from app.tools.base import ToolError
from app.tools.git_tools import (
    GetGitDiffInput,
    GetGitDiffTool,
    GetGitLogInput,
    GetGitLogTool,
    GetGitStatusInput,
    GetGitStatusTool,
)


def _git(repo_path, *args) -> None:
    subprocess.run(["git", "-C", str(repo_path), *args], check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "git_repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "app.py").write_text("def main():\n    pass\n")
    _git(repo, "add", "app.py")
    _git(repo, "commit", "-q", "-m", "initial commit")
    return repo


@pytest.fixture
def git_repository(db_session, git_repo):
    repository = Repository(name="git_repo", path=str(git_repo))
    db_session.add(repository)
    db_session.flush()
    return repository


async def test_git_status_reports_clean_tree(db_session, git_repository):
    tool = GetGitStatusTool(db_session)
    result = await tool.run(GetGitStatusInput(repository_id=git_repository.id))
    assert result.branch == "main"
    assert result.is_clean is True
    assert result.staged == result.unstaged == result.untracked == []


async def test_git_status_reports_untracked_and_modified_files(db_session, git_repository, git_repo):
    (git_repo / "app.py").write_text("def main():\n    return 1\n")
    (git_repo / "new_file.py").write_text("x = 1\n")

    tool = GetGitStatusTool(db_session)
    result = await tool.run(GetGitStatusInput(repository_id=git_repository.id))

    assert result.is_clean is False
    assert "app.py" in result.unstaged
    assert "new_file.py" in result.untracked


async def test_git_status_reports_staged_files(db_session, git_repository, git_repo):
    (git_repo / "app.py").write_text("def main():\n    return 2\n")
    _git(git_repo, "add", "app.py")

    tool = GetGitStatusTool(db_session)
    result = await tool.run(GetGitStatusInput(repository_id=git_repository.id))

    assert "app.py" in result.staged
    assert "app.py" not in result.unstaged


async def test_git_diff_shows_unstaged_changes(db_session, git_repository, git_repo):
    (git_repo / "app.py").write_text("def main():\n    return 42\n")

    tool = GetGitDiffTool(db_session)
    result = await tool.run(GetGitDiffInput(repository_id=git_repository.id))

    assert "return 42" in result.diff


async def test_git_diff_staged_only_shows_staged_changes(db_session, git_repository, git_repo):
    (git_repo / "app.py").write_text("def main():\n    return 1\n")  # unstaged
    _git(git_repo, "add", "-A")  # now staged

    unstaged_diff = await GetGitDiffTool(db_session).run(
        GetGitDiffInput(repository_id=git_repository.id, staged=False)
    )
    staged_diff = await GetGitDiffTool(db_session).run(
        GetGitDiffInput(repository_id=git_repository.id, staged=True)
    )

    assert unstaged_diff.diff == ""
    assert "return 1" in staged_diff.diff


async def test_git_log_returns_commits_newest_first(db_session, git_repository, git_repo):
    (git_repo / "app.py").write_text("def main():\n    return 1\n")
    _git(git_repo, "commit", "-q", "-am", "second commit")

    tool = GetGitLogTool(db_session)
    result = await tool.run(GetGitLogInput(repository_id=git_repository.id, max_count=10))

    messages = [c.message for c in result.commits]
    assert messages == ["second commit", "initial commit"]
    assert result.commits[0].author == "Test User"


async def test_git_log_respects_max_count(db_session, git_repository, git_repo):
    for i in range(3):
        (git_repo / "app.py").write_text(f"x = {i}\n")
        _git(git_repo, "commit", "-q", "-am", f"commit {i}")

    tool = GetGitLogTool(db_session)
    result = await tool.run(GetGitLogInput(repository_id=git_repository.id, max_count=2))

    assert len(result.commits) == 2


async def test_git_status_on_non_git_directory_raises(db_session, sample_repo, bare_repository):
    tool = GetGitStatusTool(db_session)
    with pytest.raises(ToolError, match="git"):
        await tool.run(GetGitStatusInput(repository_id=bare_repository.id))
