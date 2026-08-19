class EmbeddingProviderError(Exception):
    """Base error for all embedding provider failures."""


class EmbeddingConnectionError(EmbeddingProviderError):
    """The backend could not be reached (e.g. Ollama is not running)."""


class EmbeddingModelNotFoundError(EmbeddingProviderError):
    """The configured embedding model is not pulled / available on the backend."""


class EmbeddingTimeoutError(EmbeddingProviderError):
    """The backend did not respond within the configured timeout."""
