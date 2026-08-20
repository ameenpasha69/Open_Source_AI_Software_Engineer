import hashlib

import pytest
from app.config.settings import Settings, get_settings
from app.database.session import create_sqlite_engine, get_db_session, get_session_factory
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.llm.base import LLMProvider, LLMResponse, Message
from app.main import app
from httpx import ASGITransport


class FakeLLMProvider(LLMProvider):
    """Deterministic in-memory LLMProvider for tests — no network, no Ollama."""

    def __init__(self, response_text: str = "fake response", reachable: bool = True):
        self._response_text = response_text
        self._reachable = reachable
        self.received_messages: list[list[Message]] = []

    async def generate(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        self.received_messages.append(messages)
        return LLMResponse(
            content=self._response_text,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            finish_reason="stop",
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        for token in self._response_text.split():
            yield token + " "

    async def health_check(self) -> bool:
        return self._reachable


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider()


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic in-memory EmbeddingProvider for tests — no network, no
    Ollama. Uses hashed bag-of-words so texts sharing words end up with
    higher cosine similarity, which is enough for tests that assert on
    search *ordering* without needing a real embedding model."""

    def __init__(self, dimension: int = 32, reachable: bool = True):
        self._dimension = dimension
        self._reachable = reachable
        self.embedded_texts: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._hash_embed(t) for t in texts]

    async def health_check(self) -> bool:
        return self._reachable

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dimension
            vector[idx] += 1.0
        return vector


@pytest.fixture
def fake_embedding_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider()


@pytest.fixture
def client(tmp_path, fake_embedding_provider):
    """An ASGITransport for the FastAPI app with DB, embedding provider, and
    settings (data_dir) overridden to point at an isolated tmp_path — no
    live Ollama server or real project data touched by tests."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    session_factory = get_session_factory(engine)
    test_settings = Settings(_env_file=None, data_dir=tmp_path / "data")

    def override_get_db_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_embedding_provider] = lambda: fake_embedding_provider
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield ASGITransport(app=app)
    app.dependency_overrides.clear()


@pytest.fixture
def sample_repo(tmp_path):
    repo = tmp_path / "sample_repo"
    repo.mkdir()
    (repo / "main.py").write_text("def entrypoint():\n    pass\n")
    return repo
