from functools import lru_cache

from app.config.settings import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider

_PROVIDERS = {
    "ollama": OllamaProvider,
}


def build_llm_provider(settings: Settings) -> LLMProvider:
    provider_cls = _PROVIDERS.get(settings.llm_provider)
    if provider_cls is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Available: {sorted(_PROVIDERS)}"
        )
    return provider_cls(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_request_timeout_seconds,
        default_temperature=settings.llm_default_temperature,
    )


@lru_cache
def get_llm_provider() -> LLMProvider:
    return build_llm_provider(get_settings())
