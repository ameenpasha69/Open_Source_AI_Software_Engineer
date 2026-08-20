import pytest
from httpx import AsyncClient


@pytest.fixture
def orders_repo(tmp_path):
    repo = tmp_path / "orders_repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "def calculate_total(items):\n    return sum(item.price for item in items)\n\n\n"
        "def apply_discount(total, pct):\n    return total * (1 - pct)\n"
    )
    (repo / "payment.py").write_text(
        "def charge_card(amount, card_token):\n    return {'status': 'charged', 'amount': amount}\n"
    )
    return repo


async def _index(client, repo_path) -> str:
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post("/api/repositories/index", json={"path": str(repo_path)})
    assert resp.status_code == 200
    return resp.json()["repository_id"]


async def test_search_returns_ranked_results_with_locations(client, orders_repo):
    repository_id = await _index(client, orders_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search",
            json={"repository_id": repository_id, "query": "def calculate_total(items):", "top_k": 5},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "def calculate_total(items):"
    assert body["reranking_applied"] is False
    assert body["results"]
    top = body["results"][0]
    assert top["symbol"] == "calculate_total"
    assert top["location"] == f"service.py:{top['start_line']}-{top['end_line']}"
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


async def test_search_respects_top_k(client, orders_repo):
    repository_id = await _index(client, orders_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search", json={"repository_id": repository_id, "query": "total", "top_k": 1}
        )

    assert len(resp.json()["results"]) == 1


async def test_search_rejects_top_k_out_of_range(client, orders_repo):
    repository_id = await _index(client, orders_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search", json={"repository_id": repository_id, "query": "total", "top_k": 0}
        )

    assert resp.status_code == 422


async def test_search_unknown_repository_returns_404(client):
    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search", json={"repository_id": "does-not-exist", "query": "total"}
        )
    assert resp.status_code == 404


async def test_search_per_request_rerank_overrides_server_default(client, orders_repo):
    repository_id = await _index(client, orders_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search",
            json={"repository_id": repository_id, "query": "total", "top_k": 5, "rerank": True},
        )

    assert resp.status_code == 200
    assert resp.json()["reranking_applied"] is True


async def test_search_returns_weak_matches_rather_than_erroring(client, tmp_path):
    empty_repo = tmp_path / "empty_repo"
    empty_repo.mkdir()
    (empty_repo / "main.py").write_text("x = 1\n")
    repository_id = await _index(client, empty_repo)

    async with AsyncClient(transport=client, base_url="http://test") as http:
        resp = await http.post(
            "/api/search", json={"repository_id": repository_id, "query": "anything", "top_k": 5}
        )

    assert resp.status_code == 200
    assert resp.json()["results"]  # the module-level chunk itself is a valid, if weak, match
