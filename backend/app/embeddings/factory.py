from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.ollama_provider import OllamaEmbeddingProvider

_PROVIDERS = {
    "ollama": OllamaEmbeddingProvider,
}


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    provider_cls = _PROVIDERS.get(settings.embedding_provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Available: {sorted(_PROVIDERS)}"
        )
    return provider_cls(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider(get_settings())
