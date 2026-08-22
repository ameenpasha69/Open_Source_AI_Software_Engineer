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

    def __init__(self, base_url: str, model: str, timeout_seconds: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
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
