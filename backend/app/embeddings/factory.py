from fastapi import Depends
from sqlalchemy.orm import Session

from app.config.model_selection import get_active_model
from app.config.settings import Settings, get_settings
from app.database.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.ollama_provider import OllamaEmbeddingProvider

_PROVIDERS = {
    "ollama": OllamaEmbeddingProvider,
}


def build_embedding_provider(settings: Settings, model: str | None = None) -> EmbeddingProvider:
    """`model` overrides `settings.embedding_model` — see build_llm_provider."""
    provider_cls = _PROVIDERS.get(settings.embedding_provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Available: {sorted(_PROVIDERS)}"
        )
    return provider_cls(
        base_url=settings.embedding_base_url,
        model=model or settings.embedding_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        batch_size=settings.embedding_batch_size,
    )


def get_embedding_provider(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> EmbeddingProvider:
    """Per-request, reading the active embedding model — see get_llm_provider."""
    return build_embedding_provider(settings, model=get_active_model(session, settings, "embedding"))
