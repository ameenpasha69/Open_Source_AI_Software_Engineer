import hashlib

import pytest
from app.agents.background import get_background_agent_runner
from app.config.settings import Settings, get_settings
from app.database.models import Repository
from app.database.session import (
    create_sqlite_engine,
    get_db_session,
    get_session_factory,
    get_session_factory_dependency,
)
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.llm.base import LLMProvider, LLMResponse, Message
from app.main import app
from httpx import ASGITransport


class FakeLLMProvider(LLMProvider):
    """Deterministic in-memory LLMProvider for tests — no network, no Ollama.

    Pass `responses` for tests that need a scripted sequence of replies (e.g.
    driving an agent loop through several iterations) — each call to
    `generate()` pops the next one. With no `responses`, every call returns
    `response_text` instead.
    """

    def __init__(
        self,
        response_text: str = "fake response",
        responses: list[str] | None = None,
        reachable: bool = True,
    ):
        self._response_text = response_text
        self._responses = list(responses) if responses is not None else None
        self._reachable = reachable
        self.received_messages: list[list[Message]] = []
        self.json_mode_calls: list[bool] = []
        self.temperatures: list[float | None] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False) -> LLMResponse:
        self.received_messages.append(messages)
        self.json_mode_calls.append(json_mode)
        self.temperatures.append(temperature)
        if self._responses is not None:
            if not self._responses:
                raise AssertionError("FakeLLMProvider.generate() called more times than scripted responses")
            content = self._responses.pop(0)
        else:
            content = self._response_text
        return LLMResponse(
            content=content,
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

    @property
    def model(self) -> str:
        return "fake-embedding-model"

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
    app.dependency_overrides[get_session_factory_dependency] = lambda: session_factory
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


@pytest.fixture
def db_session(tmp_path):
    """A SQLAlchemy session bound to an isolated on-disk SQLite DB — used by
    tests that exercise the retrieval/tools layer directly (not through HTTP)."""
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    session_factory = get_session_factory(engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def bare_repository(db_session, sample_repo):
    """A Repository row pointing at `sample_repo`, with no indexing done —
    for tools (list_files, read_file, git tools) that only need repo.path."""
    repository = Repository(name="sample_repo", path=str(sample_repo))
    db_session.add(repository)
    db_session.flush()
    return repository


async def wait_for_agent_run(run_id: str) -> None:
    """POST /api/agent/run now returns immediately (status "running") and
    finishes the run in a background asyncio.Task. Await that same tracked
    task directly instead of sleep-polling the API — deterministic and fast,
    since FakeLLMProvider has no real network latency to wait out."""
    task = get_background_agent_runner().get_task(run_id)
    if task is not None:
        await task
