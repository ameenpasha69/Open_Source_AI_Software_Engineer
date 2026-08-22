"""Re-indexing a repository, shared between the explicit `POST
/api/repositories/index` endpoint and the agent's own after-a-write refresh
(`AgentRunner`).

Pulled out as its own function specifically so those two callers can't drift:
duplicating "chunk, then embed if chunking succeeded" in two places would be
one more thing to keep in sync every time either half changes.
"""

from pathlib import Path

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from app.schemas.indexing import IndexRunResult

__all__ = ["RepositoryNotFoundError", "reindex_repository"]


async def reindex_repository(
    session: Session,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    repo_path: Path,
    name: str | None = None,
) -> IndexRunResult:
    indexer = RepositoryIndexer(
        session=session,
        chunk_max_lines=settings.chunk_max_lines,
        chunk_overlap_lines=settings.chunk_overlap_lines,
        max_file_size_bytes=settings.max_indexable_file_size_bytes,
    )
    result = indexer.index(repo_path, name=name)

    if result.status == "completed":
        pipeline = EmbeddingPipeline(
            session=session, embedding_provider=embedding_provider, vector_index_dir=settings.vector_index_dir
        )
        # Chunking already succeeded and is persisted regardless of what
        # happens here — an unreachable embedding backend degrades search,
        # it doesn't lose indexing work.
        result.embedding = await pipeline.sync(result.repository_id)

    return result
