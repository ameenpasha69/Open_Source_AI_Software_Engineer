from httpx import AsyncClient


async def test_response_carries_a_generated_request_id(client):
    async with AsyncClient(transport=client, base_url="http://test") as http_client:
        response = await http_client.get("/api/health")

    assert response.status_code == 200
    assert response.headers["x-request-id"]


async def test_inbound_request_id_is_echoed_back(client):
    async with AsyncClient(transport=client, base_url="http://test") as http_client:
        response = await http_client.get("/api/health", headers={"X-Request-ID": "caller-supplied-id"})

    assert response.headers["x-request-id"] == "caller-supplied-id"


async def test_two_requests_get_different_generated_request_ids(client):
    async with AsyncClient(transport=client, base_url="http://test") as http_client:
        first = await http_client.get("/api/health")
        second = await http_client.get("/api/health")

    assert first.headers["x-request-id"] != second.headers["x-request-id"]
