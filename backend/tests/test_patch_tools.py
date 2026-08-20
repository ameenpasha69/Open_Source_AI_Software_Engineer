import pytest
from app.database.models import Repository
from app.tools.base import ToolError
from app.tools.patch_tools import ApplyPatchInput, ApplyPatchTool


@pytest.fixture
def patchable_repo(tmp_path):
    repo = tmp_path / "patchable_repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n\n\ndef calculate_total_v2(items):\n"
        "    return sum(items)\n"
    )
    (repo / ".env").write_text("SECRET=xyz\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "pkg.js").write_text("module.exports = {};\n")
    return repo


@pytest.fixture
def patchable_repository(db_session, patchable_repo):
    repository = Repository(name="patchable_repo", path=str(patchable_repo))
    db_session.add(repository)
    db_session.flush()
    return repository


async def test_apply_patch_replaces_unique_content(db_session, patchable_repository, patchable_repo):
    tool = ApplyPatchTool(db_session)
    result = await tool.run(
        ApplyPatchInput(
            repository_id=patchable_repository.id,
            path="service.py",
            old_content="def calculate_total(items):\n    return sum(items)",
            new_content="def calculate_total(items):\n    return sum(item.price for item in items)",
        )
    )

    assert result.path == "service.py"
    assert result.lines_added == 1
    assert result.lines_removed == 1
    assert "+    return sum(item.price for item in items)" in result.diff
    assert "-    return sum(items)" in result.diff

    new_content = (patchable_repo / "service.py").read_text()
    assert "return sum(item.price for item in items)" in new_content
    assert "def calculate_total_v2" in new_content  # untouched


async def test_apply_patch_rejects_missing_context(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="not found"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id,
                path="service.py",
                old_content="this text does not exist in the file",
                new_content="anything",
            )
        )


async def test_apply_patch_rejects_ambiguous_context(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="2 locations"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id,
                path="service.py",
                old_content="return sum(items)",  # appears in both calculate_total and calculate_total_v2
                new_content="return 0",
            )
        )


async def test_apply_patch_rejects_empty_old_content(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="must not be empty"):
        await tool.run(
            ApplyPatchInput(repository_id=patchable_repository.id, path="service.py", old_content="", new_content="x")
        )


async def test_apply_patch_rejects_missing_file(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="not a file"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id, path="nope.py", old_content="x", new_content="y"
            )
        )


async def test_apply_patch_rejects_path_outside_repo(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="escapes"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id,
                path="../../../../etc/passwd",
                old_content="x",
                new_content="y",
            )
        )


async def test_apply_patch_refuses_env_file(db_session, patchable_repository, patchable_repo):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="protected"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id, path=".env", old_content="SECRET=xyz", new_content="SECRET=abc"
            )
        )
    assert (patchable_repo / ".env").read_text() == "SECRET=xyz\n"  # untouched


async def test_apply_patch_refuses_ignored_directory(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError, match="ignored directory"):
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id,
                path="node_modules/pkg.js",
                old_content="module.exports = {};",
                new_content="module.exports = {x: 1};",
            )
        )


async def test_apply_patch_diff_uses_git_style_headers(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    result = await tool.run(
        ApplyPatchInput(
            repository_id=patchable_repository.id,
            path="service.py",
            old_content="def calculate_total(items):\n    return sum(items)",
            new_content="def calculate_total(items):\n    return 0",
        )
    )
    assert "--- a/service.py" in result.diff
    assert "+++ b/service.py" in result.diff


async def test_apply_patch_works_when_repository_path_is_a_symlink(db_session, tmp_path):
    """Regression test: repository.path stored as an unresolved path whose
    resolved form differs (e.g. a symlink) must not crash — this bit us for
    real via macOS's /tmp -> /private/tmp, found through live end-to-end
    testing, not code review."""
    real_dir = tmp_path / "real_repo"
    real_dir.mkdir()
    (real_dir / "service.py").write_text("def f():\n    return 1\n")

    symlink_path = tmp_path / "repo_via_symlink"
    symlink_path.symlink_to(real_dir)
    assert symlink_path.resolve() != symlink_path  # the scenario actually applies

    repository = Repository(name="symlinked_repo", path=str(symlink_path))
    db_session.add(repository)
    db_session.flush()

    tool = ApplyPatchTool(db_session)
    result = await tool.run(
        ApplyPatchInput(
            repository_id=repository.id, path="service.py", old_content="return 1", new_content="return 2"
        )
    )

    assert result.path == "service.py"
    assert (real_dir / "service.py").read_text() == "def f():\n    return 2\n"
