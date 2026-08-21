import pytest
from app.database.models import Repository
from app.tools.base import ToolError
from app.tools.patch_tools import ApplyPatchInput, ApplyPatchTool, _find_closest_match


def test_find_closest_match_finds_a_single_character_near_miss():
    original = 'students = {\n    1: {\n        "age"; 17,\n    }\n}\n'
    assert _find_closest_match(original, '"age": 17;') == '        "age"; 17,'


def test_find_closest_match_returns_none_for_unrelated_content():
    original = "def f():\n    return 1\n"
    assert _find_closest_match(original, "this is not related to the file at all") is None


def test_find_closest_match_returns_none_when_old_content_is_longer_than_the_file():
    original = "x = 1\n"
    assert _find_closest_match(original, "\n".join(f"line {i}" for i in range(20))) is None


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


async def test_apply_patch_error_omits_hint_when_nothing_plausible_matches(db_session, patchable_repository):
    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError) as exc_info:
        await tool.run(
            ApplyPatchInput(
                repository_id=patchable_repository.id,
                path="service.py",
                old_content="this text does not exist in the file",
                new_content="anything",
            )
        )
    assert "Closest match" not in str(exc_info.value)


async def test_apply_patch_error_includes_closest_match_for_a_near_miss(db_session, patchable_repository, tmp_path):
    """Reproduces a real observed failure: a model correctly locates the
    right line but transcribes old_content with punctuation in the wrong
    place ("age": 17; instead of the file's actual "age"; 17,) — apply_patch
    must still reject it (no fuzzy matching, no guessing), but the error
    should show what's actually there so a retry has a chance to succeed.
    """
    repo_dir = tmp_path / "near_miss_repo"
    repo_dir.mkdir()
    (repo_dir / "a.py").write_text('students = {\n    1: {\n        "age"; 17,\n    }\n}\n')
    repository = Repository(name="near_miss_repo", path=str(repo_dir))
    db_session.add(repository)
    db_session.flush()

    tool = ApplyPatchTool(db_session)
    with pytest.raises(ToolError) as exc_info:
        await tool.run(
            ApplyPatchInput(
                repository_id=repository.id,
                path="a.py",
                old_content='"age": 17;',
                new_content='"age": 17,',
            )
        )

    message = str(exc_info.value)
    assert "Closest match" in message
    assert '"age"; 17,' in message


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
