import httpx
import pytest
from app.config.settings import Settings
from app.llm.base import Message
from app.llm.exceptions import LLMConnectionError, LLMModelNotFoundError
from app.llm.factory import build_llm_provider
from app.llm.ollama_provider import OllamaProvider


async def test_fake_provider_records_messages(fake_llm_provider):
    messages = [Message(role="user", content="hello")]
    response = await fake_llm_provider.generate(messages)

    assert response.content == "fake response"
    assert fake_llm_provider.received_messages == [messages]


async def test_fake_provider_streams_tokens(fake_llm_provider):
    chunks = [c async for c in fake_llm_provider.stream([Message(role="user", content="hi")])]
    assert "".join(chunks).strip() == "fake response"


def test_factory_builds_ollama_provider_by_default():
    settings = Settings(_env_file=None)
    provider = build_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, llm_provider="not-a-real-provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm_provider(settings)


async def test_ollama_provider_raises_connection_error_when_unreachable(monkeypatch):
    provider = OllamaProvider(
        base_url="http://localhost:1",
        model="qwen2.5-coder:7b",
        timeout_seconds=1.0,
        default_temperature=0.2,
    )

    async def fake_post(self, url, json):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMConnectionError):
        await provider.generate([Message(role="user", content="hi")])


async def test_ollama_provider_raises_model_not_found(monkeypatch):
    provider = OllamaProvider(
        base_url="http://localhost:11434",
        model="does-not-exist",
        timeout_seconds=1.0,
        default_temperature=0.2,
    )

    async def fake_post(self, url, json):
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(LLMModelNotFoundError):
        await provider.generate([Message(role="user", content="hi")])


async def test_ollama_provider_health_check_false_when_unreachable():
    provider = OllamaProvider(
        base_url="http://localhost:1",
        model="qwen2.5-coder:7b",
        timeout_seconds=0.5,
        default_temperature=0.2,
    )
    assert await provider.health_check() is False
