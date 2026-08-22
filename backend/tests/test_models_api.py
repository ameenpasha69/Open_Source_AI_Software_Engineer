import json

import pytest
from app.config.model_selection import get_active_model, set_active_model
from app.config.settings import Settings, get_settings
from app.database.models import AppSetting, VectorRecord
from app.database.session import get_db_session
from app.llm.exceptions import LLMConnectionError
from app.llm.factory import build_llm_provider, get_model_manager
from app.llm.management import InstalledModel, ModelManagementError, ModelManager, PullProgress
from app.main import app
from httpx import AsyncClient


class FakeModelManager(ModelManager):
    """In-memory stand-in for a local inference backend — no Ollama, no
    network. `unreachable=True` makes every call fail the way a stopped
    server does."""

    def __init__(
        self,
        installed: list[InstalledModel] | None = None,
        loaded: list[str] | None = None,
        unreachable: bool = False,
        pull_updates: list[PullProgress] | None = None,
    ):
        self._installed = installed if installed is not None else list(DEFAULT_INSTALLED)
        self._loaded = loaded or []
        self._unreachable = unreachable
        self._pull_updates = pull_updates
        self.deleted: list[str] = []
        self.pulled: list[str] = []

    async def list_installed(self) -> list[InstalledModel]:
        self._guard()
        return list(self._installed)

    async def list_loaded(self) -> list[str]:
        self._guard()
        return list(self._loaded)

    async def pull(self, name: str):
        self.pulled.append(name)
        updates = self._pull_updates or [
            PullProgress(status="pulling manifest"),
            PullProgress(status="downloading", total=100, completed=50, percent=50.0),
            PullProgress(status="success", done=True),
        ]
        for update in updates:
            yield update

    def install(self, model: InstalledModel) -> None:
        """Stand-in for a completed pull."""
        self._installed.append(model)

    async def delete(self, name: str) -> None:
        self._guard()
        if not any(m.name == name for m in self._installed):
            raise ModelManagementError(f"Model '{name}' is not installed.")
        self._installed = [m for m in self._installed if m.name != name]
        self.deleted.append(name)

    async def is_reachable(self) -> bool:
        return not self._unreachable

    def _guard(self) -> None:
        if self._unreachable:
            raise LLMConnectionError("Could not reach the model backend at http://localhost:11434.")


DEFAULT_INSTALLED = [
    InstalledModel(
        name="qwen2.5-coder:7b", role="chat", size_bytes=4683087332, parameter_size="7.6B",
        quantization="Q4_K_M", family="qwen2",
    ),
    InstalledModel(
        name="llama3.1:8b", role="chat", size_bytes=4920753328, parameter_size="8.0B",
        quantization="Q4_K_M", family="llama",
    ),
    InstalledModel(
        name="nomic-embed-text:latest", role="embedding", size_bytes=274302450,
        parameter_size="137M", quantization="F16", family="nomic-bert",
    ),
    InstalledModel(
        name="mxbai-embed-large:latest", role="embedding", size_bytes=669615493,
        parameter_size="335M", quantization="F16", family="bert",
    ),
]


@pytest.fixture
def fake_manager():
    return FakeModelManager(loaded=["qwen2.5-coder:7b"])


@pytest.fixture
def models_client(client, fake_manager):
    app.dependency_overrides[get_model_manager] = lambda: fake_manager
    yield client
    app.dependency_overrides.pop(get_model_manager, None)


def session_from(models_client):
    """The same DB session factory the overridden request dependency uses, so
    a test can read back what an endpoint committed."""
    return next(app.dependency_overrides[get_db_session]())


# --- GET /api/models ---------------------------------------------------------


async def test_list_models_returns_installed_catalog_and_active_selection(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.get("/api/models")

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"]["reachable"] is True
    assert body["backend"]["error"] is None
    assert body["active_chat_model"] == Settings(_env_file=None).llm_model
    assert {m["name"] for m in body["installed"]} == {m.name for m in DEFAULT_INSTALLED}
    assert len(body["catalog"]) > 0


async def test_list_models_flags_the_active_and_loaded_models(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        body = (await http.get("/api/models")).json()

    by_name = {m["name"]: m for m in body["installed"]}
    assert by_name["qwen2.5-coder:7b"]["active"] is True
    assert by_name["qwen2.5-coder:7b"]["loaded"] is True
    assert by_name["llama3.1:8b"]["active"] is False
    assert by_name["llama3.1:8b"]["loaded"] is False


async def test_list_models_matches_the_active_model_across_an_implicit_latest_tag(models_client):
    # The default embedding model is configured as "nomic-embed-text"; the
    # backend reports it as "nomic-embed-text:latest".
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        body = (await http.get("/api/models")).json()

    by_name = {m["name"]: m for m in body["installed"]}
    assert by_name["nomic-embed-text:latest"]["active"] is True


async def test_list_models_attaches_catalog_metadata_to_known_installed_models(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        body = (await http.get("/api/models")).json()

    by_name = {m["name"]: m for m in body["installed"]}
    assert by_name["qwen2.5-coder:7b"]["catalog"]["context_window"] == 32768
    assert by_name["nomic-embed-text:latest"]["catalog"]["dimensions"] == 768


async def test_list_models_marks_catalog_entries_that_are_already_installed(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        body = (await http.get("/api/models")).json()

    by_name = {c["name"]: c for c in body["catalog"]}
    assert by_name["qwen2.5-coder:7b"]["installed"] is True
    assert by_name["nomic-embed-text"]["installed"] is True
    assert by_name["qwen2.5-coder:32b"]["installed"] is False


async def test_list_models_still_serves_the_catalog_when_the_backend_is_down(client):
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager(unreachable=True)
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/models")
    app.dependency_overrides.pop(get_model_manager, None)

    assert resp.status_code == 200
    body = resp.json()
    assert body["backend"]["reachable"] is False
    assert "Could not reach" in body["backend"]["error"]
    assert body["installed"] == []
    assert len(body["catalog"]) > 0


# --- POST /api/models/active -------------------------------------------------


async def test_switching_the_chat_model_persists_and_is_reflected_immediately(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post("/api/models/active", json={"role": "chat", "name": "llama3.1:8b"})
        listed = (await http.get("/api/models")).json()

    assert resp.status_code == 200
    assert resp.json()["active_chat_model"] == "llama3.1:8b"
    assert listed["active_chat_model"] == "llama3.1:8b"
    assert {m["name"] for m in listed["installed"] if m["active"]} == {
        "llama3.1:8b",
        "nomic-embed-text:latest",
    }


async def test_switching_the_chat_model_writes_an_override_row(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/models/active", json={"role": "chat", "name": "llama3.1:8b"})

    session = session_from(models_client)
    assert session.get(AppSetting, "active_chat_model").value == "llama3.1:8b"


async def test_the_active_model_is_what_the_llm_provider_gets_built_with(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/models/active", json={"role": "chat", "name": "llama3.1:8b"})

    settings = Settings(_env_file=None)
    session = session_from(models_client)
    provider = build_llm_provider(settings, model=get_active_model(session, settings, "chat"))

    assert provider.model == "llama3.1:8b"
    assert settings.llm_model != "llama3.1:8b"  # the configured default is untouched


async def test_switching_to_the_already_active_model_is_a_no_op(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post("/api/models/active", json={"role": "chat", "name": "qwen2.5-coder:7b"})

    assert resp.status_code == 200
    assert "already active" in resp.json()["message"]


async def test_switching_to_an_uninstalled_model_is_rejected(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post("/api/models/active", json={"role": "chat", "name": "qwen2.5-coder:32b"})

    assert resp.status_code == 404
    assert "Pull it first" in resp.json()["detail"]


async def test_an_embedding_model_cannot_be_used_as_the_chat_model(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post(
            "/api/models/active", json={"role": "chat", "name": "mxbai-embed-large:latest"}
        )

    assert resp.status_code == 400
    assert "can't be used as the chat model" in resp.json()["detail"]


async def test_a_chat_model_cannot_be_used_as_the_embedding_model(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post("/api/models/active", json={"role": "embedding", "name": "llama3.1:8b"})

    assert resp.status_code == 400
    assert "can't be used as the embedding model" in resp.json()["detail"]


async def test_switching_is_refused_when_the_backend_cannot_confirm_the_model(client):
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager(unreachable=True)
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/models/active", json={"role": "chat", "name": "llama3.1:8b"})
    app.dependency_overrides.pop(get_model_manager, None)

    assert resp.status_code == 503
    assert "was not applied" in resp.json()["detail"]


async def test_switching_the_embedding_model_is_free_when_nothing_is_indexed(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post(
            "/api/models/active", json={"role": "embedding", "name": "mxbai-embed-large:latest"}
        )

    assert resp.status_code == 200
    assert resp.json()["active_embedding_model"] == "mxbai-embed-large:latest"
    assert resp.json()["repositories_to_reindex"] == []


async def test_switching_the_embedding_model_needs_confirmation_once_vectors_exist(
    models_client, sample_repo
):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        resp = await http.post(
            "/api/models/active", json={"role": "embedding", "name": "mxbai-embed-large:latest"}
        )

    assert resp.status_code == 409
    assert "sample_repo" in resp.json()["detail"]
    # Nothing changed — the caller has to opt in first.
    session = session_from(models_client)
    assert session.get(AppSetting, "active_embedding_model") is None
    assert session.query(VectorRecord).count() > 0


async def test_confirming_an_embedding_switch_clears_vectors_and_names_the_repos(
    models_client, sample_repo
):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        resp = await http.post(
            "/api/models/active",
            json={"role": "embedding", "name": "mxbai-embed-large:latest", "confirm_reindex": True},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["active_embedding_model"] == "mxbai-embed-large:latest"
    assert body["repositories_to_reindex"] == ["sample_repo"]
    assert "re-index" in body["message"]

    session = session_from(models_client)
    assert session.query(VectorRecord).count() == 0


async def test_confirming_an_embedding_switch_removes_the_index_files(models_client, sample_repo):
    settings = app.dependency_overrides[get_settings]()
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        assert list(settings.vector_index_dir.glob("*.faiss"))
        await http.post(
            "/api/models/active",
            json={"role": "embedding", "name": "mxbai-embed-large:latest", "confirm_reindex": True},
        )

    # A stale .faiss file would be built at the old model's dimensions and
    # would reject the new model's vectors on the next index run.
    assert list(settings.vector_index_dir.glob("*.faiss")) == []


async def test_reindexing_after_an_embedding_switch_restores_search(models_client, sample_repo):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        await http.post(
            "/api/models/active",
            json={"role": "embedding", "name": "mxbai-embed-large:latest", "confirm_reindex": True},
        )
        reindex = await http.post("/api/repositories/index", json={"path": str(sample_repo)})

    body = reindex.json()
    assert body["embedding"]["error"] is None
    assert body["embedding"]["chunks_embedded"] > 0


# --- POST /api/models/pull ---------------------------------------------------


async def test_pull_streams_ndjson_progress(models_client, fake_manager):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.post("/api/models/pull", json={"name": "gemma3:4b"})

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    updates = [json.loads(line) for line in resp.text.splitlines() if line]
    assert [u["status"] for u in updates] == ["pulling manifest", "downloading", "success"]
    assert updates[1]["percent"] == 50.0
    assert updates[-1]["done"] is True
    assert fake_manager.pulled == ["gemma3:4b"]


async def test_a_failed_pull_reports_its_reason_in_the_stream(client):
    manager = FakeModelManager(
        pull_updates=[PullProgress(status="failed", done=True, error="file does not exist")]
    )
    app.dependency_overrides[get_model_manager] = lambda: manager
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/models/pull", json={"name": "nope:1b"})
    app.dependency_overrides.pop(get_model_manager, None)

    # A pull that fails part-way still has to answer 200: the failure is
    # delivered in-band, after headers are already on the wire.
    assert resp.status_code == 200
    assert json.loads(resp.text.strip())["error"] == "file does not exist"


async def test_a_pulled_model_becomes_switchable(models_client, fake_manager):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        fake_manager.install(InstalledModel(name="gemma3:4b", role="chat", size_bytes=3338801804))
        resp = await http.post("/api/models/active", json={"role": "chat", "name": "gemma3:4b"})

    assert resp.status_code == 200
    assert resp.json()["active_chat_model"] == "gemma3:4b"


# --- DELETE /api/models ------------------------------------------------------


async def test_delete_removes_an_unused_model(models_client, fake_manager):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.delete("/api/models", params={"name": "llama3.1:8b"})

    assert resp.status_code == 200
    assert fake_manager.deleted == ["llama3.1:8b"]


async def test_delete_refuses_to_remove_the_active_chat_model(models_client, fake_manager):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.delete("/api/models", params={"name": "qwen2.5-coder:7b"})

    assert resp.status_code == 409
    assert "active chat model" in resp.json()["detail"]
    assert fake_manager.deleted == []


async def test_delete_refuses_to_remove_the_active_embedding_model(models_client, fake_manager):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.delete("/api/models", params={"name": "nomic-embed-text:latest"})

    assert resp.status_code == 409
    assert "active embedding model" in resp.json()["detail"]
    assert fake_manager.deleted == []


async def test_delete_of_an_uninstalled_model_is_a_client_error(models_client):
    async with AsyncClient(transport=models_client, base_url="http://test") as http:
        resp = await http.delete("/api/models", params={"name": "never-pulled:1b"})

    assert resp.status_code == 400
    assert "not installed" in resp.json()["detail"]


async def test_delete_reports_an_unreachable_backend_as_unavailable(client):
    app.dependency_overrides[get_model_manager] = lambda: FakeModelManager(unreachable=True)
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.delete("/api/models", params={"name": "llama3.1:8b"})
    app.dependency_overrides.pop(get_model_manager, None)

    assert resp.status_code == 503


# --- selection persistence ---------------------------------------------------


def test_active_model_falls_back_to_the_configured_default(db_session):
    settings = Settings(_env_file=None)
    assert get_active_model(db_session, settings, "chat") == settings.llm_model
    assert get_active_model(db_session, settings, "embedding") == settings.embedding_model


def test_setting_the_active_model_twice_updates_the_same_row(db_session):
    settings = Settings(_env_file=None)
    set_active_model(db_session, "chat", "llama3.1:8b")
    set_active_model(db_session, "chat", "gemma3:4b")
    db_session.commit()

    assert db_session.query(AppSetting).count() == 1
    assert get_active_model(db_session, settings, "chat") == "gemma3:4b"


def test_an_empty_override_falls_back_to_the_default(db_session):
    settings = Settings(_env_file=None)
    set_active_model(db_session, "chat", "   ")
    db_session.commit()

    assert get_active_model(db_session, settings, "chat") == settings.llm_model
