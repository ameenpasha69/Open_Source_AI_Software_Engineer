import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.models import CodeChunk, IndexedFile, Repository, VectorRecord
from app.database.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.retrieval.command_detection import detect_default_commands
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from app.schemas.indexing import IndexRepositoryRequest, IndexRunResult, RepositorySummary

router = APIRouter(tags=["repositories"])


@router.post("/repositories/index", response_model=IndexRunResult)
async def index_repository(
    request: IndexRepositoryRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> IndexRunResult:
    indexer = RepositoryIndexer(
        session=session,
        chunk_max_lines=settings.chunk_max_lines,
        chunk_overlap_lines=settings.chunk_overlap_lines,
        max_file_size_bytes=settings.max_indexable_file_size_bytes,
    )
    try:
        result = indexer.index(Path(request.path), name=request.name)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if result.status == "completed":
        pipeline = EmbeddingPipeline(
            session=session, embedding_provider=embedding_provider, vector_index_dir=settings.vector_index_dir
        )
        # Chunking already succeeded and is persisted regardless of what happens
        # here — an unreachable embedding backend degrades search, it doesn't
        # lose indexing work.
        result.embedding = await pipeline.sync(result.repository_id)
        _apply_command_config(session, result.repository_id, request)
        session.commit()

    return result


@router.get("/repositories", response_model=list[RepositorySummary])
async def list_repositories(session: Session = Depends(get_db_session)) -> list[RepositorySummary]:
    repositories = session.scalars(select(Repository)).all()
    return [_to_summary(session, repo) for repo in repositories]


@router.get("/repositories/{repository_id}", response_model=RepositorySummary)
async def get_repository(
    repository_id: str, session: Session = Depends(get_db_session)
) -> RepositorySummary:
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail=f"Repository '{repository_id}' not found")
    return _to_summary(session, repository)


def _apply_command_config(session: Session, repository_id: str, request: IndexRepositoryRequest) -> None:
    """Explicit request fields always win. Otherwise, only fill in a command
    that has never been configured — re-indexing must never silently clear
    or overwrite a command someone already set."""
    repository = session.get(Repository, repository_id)
    defaults = detect_default_commands(session, repository_id)

    for field, request_value in (
        ("test_command", request.test_command),
        ("lint_command", request.lint_command),
        ("format_command", request.format_command),
    ):
        column = f"{field}_json"
        if request_value is not None:
            setattr(repository, column, json.dumps(request_value))
        elif getattr(repository, column) is None and field in defaults:
            setattr(repository, column, json.dumps(defaults[field]))


def _to_summary(session: Session, repository: Repository) -> RepositorySummary:
    file_count = session.scalar(
        select(func.count()).select_from(IndexedFile).where(IndexedFile.repository_id == repository.id)
    )
    chunk_count = session.scalar(
        select(func.count()).select_from(CodeChunk).where(CodeChunk.repository_id == repository.id)
    )
    embedded_count = session.scalar(
        select(func.count()).select_from(VectorRecord).where(VectorRecord.repository_id == repository.id)
    )
    return RepositorySummary(
        id=repository.id,
        name=repository.name,
        path=repository.path,
        created_at=repository.created_at,
        last_indexed_at=repository.last_indexed_at,
        indexed_file_count=file_count or 0,
        chunk_count=chunk_count or 0,
        embedded_chunk_count=embedded_count or 0,
        test_command=json.loads(repository.test_command_json) if repository.test_command_json else None,
        lint_command=json.loads(repository.lint_command_json) if repository.lint_command_json else None,
        format_command=json.loads(repository.format_command_json) if repository.format_command_json else None,
    )
