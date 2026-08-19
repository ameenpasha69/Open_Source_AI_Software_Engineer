from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Provider-agnostic interface for local text embedding.

    Nothing outside `app/embeddings/` may depend on a specific backend.
    Swapping the embedding model or engine (e.g. to a sentence-transformers
    backend) means writing one new class here and registering it in
    `factory.py` — the vector store and retrieval pipeline are untouched.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single piece of text (e.g. a search query)."""

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents (e.g. code chunks) in one call."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and the model is available."""
