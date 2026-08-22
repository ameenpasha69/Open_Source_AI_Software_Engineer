from fastapi import Depends
from sqlalchemy.orm import Session

from app.config.model_selection import get_active_model
from app.config.settings import Settings, get_settings
from app.database.session import get_db_session
from app.llm.base import LLMProvider
from app.llm.management import ModelManager, OllamaModelManager
from app.llm.ollama_provider import OllamaProvider

_PROVIDERS = {
    "ollama": OllamaProvider,
}

_MANAGERS = {
    "ollama": OllamaModelManager,
}


def build_llm_provider(settings: Settings, model: str | None = None) -> LLMProvider:
    """`model` overrides `settings.llm_model` — that's how a model switched
    in the UI takes effect without the process's Settings changing."""
    provider_cls = _PROVIDERS.get(settings.llm_provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Available: {sorted(_PROVIDERS)}"
        )
    return provider_cls(
        base_url=settings.llm_base_url,
        model=model or settings.llm_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        default_temperature=settings.llm_default_temperature,
    )


def get_llm_provider(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
) -> LLMProvider:
    """Built per request rather than cached, so a model switch applies to the
    next run without a restart. Providers are cheap value objects — they hold
    a URL and a model name, and open a connection per call.

    Resolving the active model *here*, at request scope, is also what makes a
    switch safe mid-flight: a background agent run keeps the provider it was
    handed when it started instead of having the model changed underneath it.
    """
    return build_llm_provider(settings, model=get_active_model(session, settings, "chat"))


def build_model_manager(settings: Settings, base_url: str | None = None) -> ModelManager:
    manager_cls = _MANAGERS.get(settings.llm_provider)
    if manager_cls is None:
        raise ValueError(
            f"LLM_PROVIDER '{settings.llm_provider}' does not support model management. "
            f"Available: {sorted(_MANAGERS)}"
        )
    return manager_cls(base_url=base_url or settings.llm_base_url)


def get_model_manager(settings: Settings = Depends(get_settings)) -> ModelManager:
    return build_model_manager(settings)
