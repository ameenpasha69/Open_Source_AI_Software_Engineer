import httpx

from app.embeddings.base import EmbeddingProvider
from app.embeddings.exceptions import (
    EmbeddingConnectionError,
    EmbeddingModelNotFoundError,
    EmbeddingTimeoutError,
)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider backed by a local Ollama server's /api/embed endpoint,
    which accepts a batch of inputs and returns one vector per input."""

    def __init__(self, base_url: str, model: str, timeout_seconds: float,
                 batch_size: int = 64):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._batch_size = max(1, batch_size)

    @property
    def model(self) -> str:
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed every text, in batches.

        /api/embed accepts a list, but handing it the whole corpus at once is
        not safe: the request is served by a model runner subprocess, and on a
        small GPU a large enough batch kills it. The parent server survives, so
        the failure surfaces as a bare HTTP 400 whose body is a connection
        error to the runner's own port -- which looks like a malformed request
        rather than what it is. Indexing this repository (1313 chunks) on a
        4 GiB card reproduces it every time; 200 chunks succeed, 400 do not.

        Batching bounds the peak the runner has to absorb regardless of
        repository size, at the cost of more round trips.
        """
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(await self._embed_batch(texts[start:start + self._batch_size]))
        return vectors

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/embed",
                    json={"model": self._model, "input": texts},
                )
        except httpx.ConnectError as exc:
            raise EmbeddingConnectionError(
                f"Could not reach Ollama at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(f"Ollama did not respond within {self._timeout}s") from exc

        if resp.status_code == 404:
            raise EmbeddingModelNotFoundError(
                f"Model '{self._model}' not found on Ollama. Run `ollama pull {self._model}`."
            )
        resp.raise_for_status()
        return resp.json()["embeddings"]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(m == self._model or m.startswith(f"{self._model}:") for m in models)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
