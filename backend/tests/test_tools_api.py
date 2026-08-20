from httpx import AsyncClient


async def _index(client, repo_path) -> str:
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/repositories/index", json={"path": str(repo_path)})
    assert resp.status_code == 200
    return resp.json()["repository_id"]


async def test_list_tools_returns_all_registered_tools(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.get("/api/tools")

    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()}
    assert names == {
        "list_files",
        "read_file",
        "get_file_context",
        "search_code",
        "find_symbol",
        "find_references",
        "get_git_status",
        "get_git_diff",
        "get_git_log",
        "apply_patch",
        "run_tests",
        "run_command",
        "run_linter",
        "run_formatter",
    }


async def test_execute_read_file_via_api(client, sample_repo):
    repository_id = await _index(client, sample_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/tools/execute",
            json={"tool_name": "read_file", "input": {"repository_id": repository_id, "path": "main.py"}},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "def entrypoint" in body["output"]["content"]


async def test_execute_unknown_tool_returns_200_with_failed_result(client):
    """Tool failures are data (a failed ToolResult), not HTTP errors — the
    caller (eventually the agent) needs to see *which* tool failed and why,
    not just get a generic 4xx."""
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/tools/execute", json={"tool_name": "not_a_tool", "input": {}})

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert "Unknown tool" in body["error"]


async def test_execute_tool_with_invalid_input_returns_failed_result(client, sample_repo):
    repository_id = await _index(client, sample_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/tools/execute",
            json={"tool_name": "read_file", "input": {"repository_id": repository_id}},  # missing "path"
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert "Invalid input" in body["error"]


async def test_execute_search_code_via_api(client, sample_repo):
    repository_id = await _index(client, sample_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/tools/execute",
            json={
                "tool_name": "search_code",
                "input": {"repository_id": repository_id, "query": "def entrypoint():", "top_k": 3},
            },
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["output"]["results"]
