class LLMProviderError(Exception):
    """Base error for all LLM provider failures."""


class LLMConnectionError(LLMProviderError):
    """The backend could not be reached (e.g. Ollama is not running)."""


class LLMModelNotFoundError(LLMProviderError):
    """The configured model is not pulled / available on the backend."""


class LLMTimeoutError(LLMProviderError):
    """The backend did not respond within the configured timeout."""
