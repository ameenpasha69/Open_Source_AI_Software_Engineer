import pytest
from app.config.settings import Settings, get_settings
from app.database.session import create_sqlite_engine, get_db_session, get_session_factory
from app.embeddings.factory import get_embedding_provider
from app.main import app
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def client(tmp_path, fake_embedding_provider):
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


async def test_index_repository_endpoint(client, sample_repo):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/repositories/index", json={"path": str(sample_repo)})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["files_indexed"] == 1
    assert body["chunks_created"] >= 1
    assert body["embedding"]["chunks_embedded"] == body["chunks_created"]
    assert body["embedding"]["error"] is None


async def test_index_missing_path_returns_404(client, tmp_path):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/repositories/index", json={"path": str(tmp_path / "nope")}
        )

    assert resp.status_code == 404


async def test_list_and_get_repository(client, sample_repo):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        index_resp = await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        repository_id = index_resp.json()["repository_id"]

        list_resp = await http.get("/api/repositories")
        assert list_resp.status_code == 200
        assert len(list_resp.json()) == 1
        assert list_resp.json()[0]["indexed_file_count"] == 1
        assert list_resp.json()[0]["embedded_chunk_count"] == list_resp.json()[0]["chunk_count"]

        get_resp = await http.get(f"/api/repositories/{repository_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == repository_id


async def test_get_unknown_repository_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/repositories/does-not-exist")
    assert resp.status_code == 404
