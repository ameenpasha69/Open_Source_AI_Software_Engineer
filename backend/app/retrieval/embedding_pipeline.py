import time
from pathlib import Path

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import CodeChunk, VectorRecord
from app.embeddings.base import EmbeddingProvider
from app.retrieval.vector_store import FaissVectorStore
from app.schemas.indexing import EmbeddingSyncResult
from app.schemas.search import SearchResult


class EmbeddingPipeline:
    """Reconciles a repository's CodeChunk rows against its FAISS index.

    A VectorRecord row is the source of truth for "this chunk has been
    embedded" — chunks with no VectorRecord are embedded and added; existing
    VectorRecords whose chunk no longer exists (file changed or was deleted
    during re-indexing) are removed from the index. Unchanged chunks keep
    their id across re-indexes, so they're never re-embedded.
    """

    def __init__(self, session: Session, embedding_provider: EmbeddingProvider, vector_index_dir: Path):
        self._session = session
        self._embedding_provider = embedding_provider
        self._vector_index_dir = vector_index_dir

    async def sync(self, repository_id: str) -> EmbeddingSyncResult:
        started = time.monotonic()
        index_path = self._vector_index_dir / f"{repository_id}.faiss"

        current_chunk_ids = set(
            self._session.scalars(select(CodeChunk.id).where(CodeChunk.repository_id == repository_id))
        )
        vectorized = {
            record.chunk_id: record.faiss_id
            for record in self._session.scalars(
                select(VectorRecord).where(VectorRecord.repository_id == repository_id)
            )
        }
        to_remove_faiss_ids = [
            faiss_id for chunk_id, faiss_id in vectorized.items() if chunk_id not in current_chunk_ids
        ]
        to_add_chunk_ids = current_chunk_ids - vectorized.keys()

        if not to_remove_faiss_ids and not to_add_chunk_ids:
            return EmbeddingSyncResult(
                chunks_embedded=0, chunks_removed=0, duration_seconds=round(time.monotonic() - started, 3)
            )

        try:
            embedded_count = await self._apply(index_path, to_remove_faiss_ids, to_add_chunk_ids)
            self._session.commit()
        except Exception as exc:  # noqa: BLE001 - recorded on the result, then surfaced to the caller
            self._session.rollback()
            return EmbeddingSyncResult(
                chunks_embedded=0,
                chunks_removed=0,
                duration_seconds=round(time.monotonic() - started, 3),
                error=str(exc),
            )

        return EmbeddingSyncResult(
            chunks_embedded=embedded_count,
            chunks_removed=len(to_remove_faiss_ids),
            duration_seconds=round(time.monotonic() - started, 3),
        )

    async def _apply(self, index_path: Path, to_remove_faiss_ids: list[int], to_add_chunk_ids: set[str]) -> int:
        store = FaissVectorStore(index_path) if index_path.exists() else None

        if to_remove_faiss_ids:
            if store is not None:
                await store.delete(to_remove_faiss_ids)
            self._session.execute(sa_delete(VectorRecord).where(VectorRecord.faiss_id.in_(to_remove_faiss_ids)))

        embedded_count = 0
        if to_add_chunk_ids:
            chunks = self._session.scalars(select(CodeChunk).where(CodeChunk.id.in_(to_add_chunk_ids))).all()
            vectors = await self._embedding_provider.embed_documents([c.content for c in chunks])

            records = [VectorRecord(chunk_id=c.id, repository_id=c.repository_id) for c in chunks]
            self._session.add_all(records)
            self._session.flush()  # populate autoincrement faiss_id on each record

            if store is None:
                store = FaissVectorStore(index_path, dimension=len(vectors[0]))
            await store.add([r.faiss_id for r in records], vectors)
            embedded_count = len(chunks)

        if store is not None:
            await store.save()
        return embedded_count

    async def search(self, repository_id: str, query_text: str, top_k: int = 10) -> list[SearchResult]:
        index_path = self._vector_index_dir / f"{repository_id}.faiss"
        if not index_path.exists():
            return []

        query_vector = await self._embedding_provider.embed_text(query_text)
        store = FaissVectorStore(index_path)
        hits = await store.search(query_vector, top_k)
        if not hits:
            return []

        faiss_ids = [faiss_id for faiss_id, _ in hits]
        records = {
            r.faiss_id: r
            for r in self._session.scalars(select(VectorRecord).where(VectorRecord.faiss_id.in_(faiss_ids)))
        }
        chunk_ids = [r.chunk_id for r in records.values()]
        chunks_by_id = {
            c.id: c for c in self._session.scalars(select(CodeChunk).where(CodeChunk.id.in_(chunk_ids)))
        }

        results = []
        for faiss_id, score in hits:
            record = records.get(faiss_id)
            chunk = chunks_by_id.get(record.chunk_id) if record else None
            if chunk is None:
                continue  # stale hit from a not-yet-reconciled index; skip rather than error
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    file_path=chunk.relative_path,
                    language=chunk.language,
                    symbol=chunk.symbol,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    content=chunk.content,
                    score=score,
                )
            )
        return results
