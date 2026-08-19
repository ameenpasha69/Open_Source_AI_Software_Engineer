import asyncio
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np


class VectorStore(ABC):
    """Provider-agnostic interface for a local vector index. IDs are caller-
    assigned int64s (here, `VectorRecord.faiss_id`) so the store never needs
    to know about chunks, repositories, or SQLite."""

    @abstractmethod
    async def add(self, ids: list[int], vectors: list[list[float]]) -> None: ...

    @abstractmethod
    async def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        """Return up to `top_k` (id, similarity_score) pairs, most similar first."""

    @abstractmethod
    async def delete(self, ids: list[int]) -> None: ...

    @abstractmethod
    async def save(self) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class FaissVectorStore(VectorStore):
    """FAISS-backed VectorStore, persisted as a single index file on disk.

    Uses IndexIDMap2 around a flat inner-product index, with all vectors
    L2-normalized on add/search — inner product on normalized vectors is
    equivalent to cosine similarity, and IndexIDMap2 supports id-based
    removal and reconstruction, unlike FAISS's approximate index types.
    A flat index is exact (no recall loss) and fine for the chunk counts a
    single local repository produces; an approximate index (IVF/HNSW) would
    be the natural upgrade if this needed to scale to millions of vectors.
    """

    def __init__(self, index_path: Path, dimension: int | None = None):
        self._index_path = index_path
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        else:
            if dimension is None:
                raise ValueError(
                    f"No index exists at {index_path} yet — `dimension` is required to create one."
                )
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

    @property
    def dimension(self) -> int:
        return self._index.d

    @property
    def size(self) -> int:
        return self._index.ntotal

    async def add(self, ids: list[int], vectors: list[list[float]]) -> None:
        await asyncio.to_thread(self._add_sync, ids, vectors)

    def _add_sync(self, ids: list[int], vectors: list[list[float]]) -> None:
        matrix = _normalized_matrix(vectors)
        self._index.add_with_ids(matrix, np.array(ids, dtype=np.int64))

    async def search(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        return await asyncio.to_thread(self._search_sync, query_vector, top_k)

    def _search_sync(self, query_vector: list[float], top_k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        query = _normalized_matrix([query_vector])
        scores, ids = self._index.search(query, min(top_k, self._index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    async def delete(self, ids: list[int]) -> None:
        await asyncio.to_thread(self._delete_sync, ids)

    def _delete_sync(self, ids: list[int]) -> None:
        self._index.remove_ids(np.array(ids, dtype=np.int64))

    async def save(self) -> None:
        await asyncio.to_thread(self._save_sync)

    def _save_sync(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))


def _normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid dividing an all-zero vector by zero
    return matrix / norms
