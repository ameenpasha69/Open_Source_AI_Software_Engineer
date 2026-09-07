import httpx
import pytest
from app.config.settings import Settings
from app.embeddings.exceptions import EmbeddingConnectionError, EmbeddingModelNotFoundError
from app.embeddings.factory import build_embedding_provider
from app.embeddings.ollama_provider import OllamaEmbeddingProvider


async def test_fake_provider_embeds_and_records_texts(fake_embedding_provider):
    vectors = await fake_embedding_provider.embed_documents(["def foo(): pass", "def bar(): pass"])

    assert len(vectors) == 2
    assert len(vectors[0]) == fake_embedding_provider._dimension
    assert fake_embedding_provider.embedded_texts == ["def foo(): pass", "def bar(): pass"]


async def test_fake_provider_embed_text_matches_embed_documents(fake_embedding_provider):
    single = await fake_embedding_provider.embed_text("hello")
    batch = await fake_embedding_provider.embed_documents(["hello"])
    assert single == batch[0]


def test_factory_builds_ollama_provider_by_default():
    settings = Settings(_env_file=None)
    provider = build_embedding_provider(settings)
    assert isinstance(provider, OllamaEmbeddingProvider)


def test_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, embedding_provider="not-real")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_PROVIDER"):
        build_embedding_provider(settings)


async def test_ollama_embedding_provider_raises_connection_error(monkeypatch):
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:1", model="nomic-embed-text", timeout_seconds=1.0
    )

    async def fake_post(self, url, json):
        raise httpx.ConnectError("refused", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(EmbeddingConnectionError):
        await provider.embed_documents(["hello"])


async def test_ollama_embedding_provider_raises_model_not_found(monkeypatch):
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="does-not-exist", timeout_seconds=1.0
    )

    async def fake_post(self, url, json):
        return httpx.Response(404, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    with pytest.raises(EmbeddingModelNotFoundError):
        await provider.embed_documents(["hello"])


async def test_ollama_embedding_provider_parses_response(monkeypatch):
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text", timeout_seconds=1.0
    )

    async def fake_post(self, url, json):
        assert json == {"model": "nomic-embed-text", "input": ["a", "b"]}
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    vectors = await provider.embed_documents(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_ollama_embedding_provider_embed_documents_empty_list_short_circuits():
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:1", model="nomic-embed-text", timeout_seconds=1.0
    )
    assert await provider.embed_documents([]) == []


async def test_ollama_embedding_provider_health_check_false_when_unreachable():
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:1", model="nomic-embed-text", timeout_seconds=0.5
    )
    assert await provider.health_check() is False


async def test_ollama_embedding_provider_splits_large_input_into_batches(monkeypatch):
    """A corpus larger than the batch size goes out as several requests.

    Handing Ollama the whole corpus in one call kills the model runner on a
    small GPU, and the resulting HTTP 400 looks like a malformed request
    rather than a crashed subprocess. Batching is what keeps the peak bounded,
    so it is worth asserting rather than trusting.
    """
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text",
        timeout_seconds=1.0, batch_size=2,
    )
    seen: list[list[str]] = []

    async def fake_post(self, url, json):
        seen.append(json["input"])
        return httpx.Response(
            200,
            json={"embeddings": [[float(len(t))] for t in json["input"]]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    vectors = await provider.embed_documents(["a", "bb", "ccc", "dddd", "eeeee"])

    assert seen == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    # One vector per input, in the original order, across the batch boundaries.
    assert vectors == [[1.0], [2.0], [3.0], [4.0], [5.0]]


async def test_ollama_embedding_provider_batch_size_floor_is_one():
    """A zero or negative batch size would loop forever rather than fail."""
    provider = OllamaEmbeddingProvider(
        base_url="http://localhost:11434", model="nomic-embed-text",
        timeout_seconds=1.0, batch_size=0,
    )
    assert provider._batch_size == 1
