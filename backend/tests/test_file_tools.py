import pytest
from app.database.models import Repository
from app.tools.base import ToolError
from app.tools.file_tools import (
    GetFileContextInput,
    GetFileContextTool,
    ListFilesInput,
    ListFilesTool,
    ReadFileInput,
    ReadFileTool,
)


@pytest.fixture
def nested_repo(tmp_path):
    repo = tmp_path / "nested_repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "main.py").write_text("\n".join(f"line {i}" for i in range(1, 51)) + "\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "junk.js").write_text("ignored\n")
    (repo / "README.md").write_text("# hello\n")
    return repo


@pytest.fixture
def nested_repository(db_session, nested_repo):
    repository = Repository(name="nested_repo", path=str(nested_repo))
    db_session.add(repository)
    db_session.flush()
    return repository


async def test_list_files_at_root_excludes_ignored_dirs(db_session, nested_repository):
    tool = ListFilesTool(db_session)
    result = await tool.run(ListFilesInput(repository_id=nested_repository.id, path=""))
    paths = {e.path for e in result.entries}
    assert "src" in paths
    assert "README.md" in paths
    assert "node_modules" not in paths


async def test_list_files_recursive_finds_nested_file(db_session, nested_repository):
    tool = ListFilesTool(db_session)
    result = await tool.run(ListFilesInput(repository_id=nested_repository.id, path="", recursive=True))
    paths = {e.path for e in result.entries}
    assert "src/main.py" in paths


async def test_list_files_rejects_path_outside_repo(db_session, nested_repository):
    tool = ListFilesTool(db_session)
    with pytest.raises(ToolError, match="escapes"):
        await tool.run(ListFilesInput(repository_id=nested_repository.id, path="../../etc"))


async def test_list_files_on_a_file_raises(db_session, nested_repository):
    tool = ListFilesTool(db_session)
    with pytest.raises(ToolError, match="not a directory"):
        await tool.run(ListFilesInput(repository_id=nested_repository.id, path="README.md"))


async def test_read_file_returns_full_content_by_default(db_session, nested_repository):
    tool = ReadFileTool(db_session, max_file_size_bytes=1_000_000)
    result = await tool.run(ReadFileInput(repository_id=nested_repository.id, path="src/main.py"))
    assert result.total_lines == 50
    assert result.start_line == 1
    assert result.end_line == 50
    assert result.content.splitlines()[0] == "line 1"


async def test_read_file_respects_line_range(db_session, nested_repository):
    tool = ReadFileTool(db_session, max_file_size_bytes=1_000_000)
    result = await tool.run(
        ReadFileInput(repository_id=nested_repository.id, path="src/main.py", start_line=10, end_line=12)
    )
    assert result.content.splitlines() == ["line 10", "line 11", "line 12"]


async def test_read_file_rejects_start_after_end(db_session, nested_repository):
    tool = ReadFileTool(db_session, max_file_size_bytes=1_000_000)
    with pytest.raises(ToolError, match="start_line"):
        await tool.run(
            ReadFileInput(repository_id=nested_repository.id, path="src/main.py", start_line=20, end_line=5)
        )


async def test_read_file_enforces_size_limit(db_session, nested_repository):
    tool = ReadFileTool(db_session, max_file_size_bytes=5)
    with pytest.raises(ToolError, match="exceeds"):
        await tool.run(ReadFileInput(repository_id=nested_repository.id, path="src/main.py"))


async def test_read_file_missing_file_raises(db_session, nested_repository):
    tool = ReadFileTool(db_session, max_file_size_bytes=1_000_000)
    with pytest.raises(ToolError, match="not a file"):
        await tool.run(ReadFileInput(repository_id=nested_repository.id, path="does_not_exist.py"))


async def test_get_file_context_centers_window_on_line(db_session, nested_repository):
    tool = GetFileContextTool(db_session, max_file_size_bytes=1_000_000)
    result = await tool.run(
        GetFileContextInput(repository_id=nested_repository.id, path="src/main.py", line=25, context_lines=3)
    )
    assert result.start_line == 22
    assert result.end_line == 28
    assert "line 25" in result.content


async def test_get_file_context_clamps_at_start_of_file(db_session, nested_repository):
    tool = GetFileContextTool(db_session, max_file_size_bytes=1_000_000)
    result = await tool.run(
        GetFileContextInput(repository_id=nested_repository.id, path="src/main.py", line=2, context_lines=10)
    )
    assert result.start_line == 1
