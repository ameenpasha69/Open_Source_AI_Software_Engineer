from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.models import CodeChunk, IndexedFile, Repository
from app.database.session import get_db_session
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from app.schemas.indexing import IndexRepositoryRequest, IndexRunResult, RepositorySummary

router = APIRouter(tags=["repositories"])


@router.post("/repositories/index", response_model=IndexRunResult)
async def index_repository(
    request: IndexRepositoryRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> IndexRunResult:
    indexer = RepositoryIndexer(
        session=session,
        chunk_max_lines=settings.chunk_max_lines,
        chunk_overlap_lines=settings.chunk_overlap_lines,
        max_file_size_bytes=settings.max_indexable_file_size_bytes,
    )
    try:
        return indexer.index(Path(request.path), name=request.name)
    except RepositoryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


def _to_summary(session: Session, repository: Repository) -> RepositorySummary:
    file_count = session.scalar(
        select(func.count()).select_from(IndexedFile).where(IndexedFile.repository_id == repository.id)
    )
    chunk_count = session.scalar(
        select(func.count()).select_from(CodeChunk).where(CodeChunk.repository_id == repository.id)
    )
    return RepositorySummary(
        id=repository.id,
        name=repository.name,
        path=repository.path,
        created_at=repository.created_at,
        last_indexed_at=repository.last_indexed_at,
        indexed_file_count=file_count or 0,
        chunk_count=chunk_count or 0,
    )
