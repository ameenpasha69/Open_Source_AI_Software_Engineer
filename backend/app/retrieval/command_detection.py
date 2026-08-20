from collections import Counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import IndexedFile

# Only Python gets an auto-detected default — it's the language this project
# actually supports well (AST-based chunking, etc.); guessing a test/lint
# command for every ecosystem risks being wrong more often than it's right.
# Everything else must be configured explicitly via IndexRepositoryRequest.
_PYTHON_DEFAULTS = {
    "test_command": ["python3", "-m", "pytest"],
    "lint_command": ["ruff", "check", "."],
    "format_command": ["ruff", "format", "."],
}


def detect_default_commands(session: Session, repository_id: str) -> dict[str, list[str]]:
    """Best-effort defaults based on the repository's indexed file languages.
    Returns an empty dict if no language is clearly dominant (Python here
    means "more Python files than anything else," not "any Python files")."""
    languages = session.scalars(
        select(IndexedFile.language).where(IndexedFile.repository_id == repository_id)
    ).all()
    if not languages:
        return {}

    most_common_language, _ = Counter(languages).most_common(1)[0]
    if most_common_language == "python":
        return dict(_PYTHON_DEFAULTS)
    return {}
