from pathlib import Path

from sqlalchemy.orm import Session

from app.database.models import Repository
from app.tools.base import ToolError


def get_repository(session: Session, repository_id: str) -> Repository:
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise ToolError(f"Repository '{repository_id}' not found")
    return repository


def get_repository_root(session: Session, repository_id: str) -> Path:
    """Always resolved (symlinks followed, e.g. macOS /tmp -> /private/tmp).
    `resolve_safe_path()` also returns resolved paths — callers that compare
    the two directly (relative_to, ignored-directory checks) need both sides
    resolved consistently or the comparison raises spuriously.
    """
    return Path(get_repository(session, repository_id).path).resolve()
