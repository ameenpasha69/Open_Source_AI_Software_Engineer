import pytest
from app.tools.base import ToolError
from app.tools.path_utils import resolve_safe_path


def test_resolve_safe_path_allows_path_inside_repo(tmp_path):
    (tmp_path / "src").mkdir()
    resolved = resolve_safe_path(tmp_path, "src")
    assert resolved == (tmp_path / "src").resolve()


def test_resolve_safe_path_rejects_relative_traversal(tmp_path):
    with pytest.raises(ToolError, match="escapes"):
        resolve_safe_path(tmp_path, "../../etc/passwd")


def test_resolve_safe_path_rejects_absolute_path_outside_repo(tmp_path):
    with pytest.raises(ToolError, match="escapes"):
        resolve_safe_path(tmp_path, "/etc/passwd")


def test_resolve_safe_path_allows_empty_string_as_repo_root(tmp_path):
    assert resolve_safe_path(tmp_path, "") == tmp_path.resolve()
