from httpx import AsyncClient


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


async def test_index_auto_detects_python_test_command(client, sample_repo):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        index_resp = await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        repository_id = index_resp.json()["repository_id"]
        get_resp = await http.get(f"/api/repositories/{repository_id}")

    body = get_resp.json()
    assert body["test_command"] == ["python3", "-m", "pytest"]
    assert body["lint_command"] == ["ruff", "check", "."]
    assert body["format_command"] == ["ruff", "format", "."]


async def test_index_explicit_test_command_overrides_default(client, sample_repo):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        index_resp = await http.post(
            "/api/repositories/index",
            json={"path": str(sample_repo), "test_command": ["make", "test"]},
        )
        repository_id = index_resp.json()["repository_id"]
        get_resp = await http.get(f"/api/repositories/{repository_id}")

    assert get_resp.json()["test_command"] == ["make", "test"]


async def test_reindex_without_command_fields_preserves_existing_config(client, sample_repo):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        first = await http.post(
            "/api/repositories/index",
            json={"path": str(sample_repo), "test_command": ["make", "test"]},
        )
        repository_id = first.json()["repository_id"]

        # Re-index without specifying test_command — must not clear it.
        await http.post("/api/repositories/index", json={"path": str(sample_repo)})
        get_resp = await http.get(f"/api/repositories/{repository_id}")

    assert get_resp.json()["test_command"] == ["make", "test"]
