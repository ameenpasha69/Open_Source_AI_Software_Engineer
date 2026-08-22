import json

import httpx
import pytest
from app.config.settings import Settings
from app.llm.exceptions import LLMConnectionError
from app.llm.factory import build_model_manager
from app.llm.management import ModelManagementError, OllamaModelManager

TAGS_RESPONSE = {
    "models": [
        {
            "name": "qwen2.5-coder:7b",
            "size": 4683087332,
            "modified_at": "2026-08-01T10:00:00Z",
            "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"},
        },
        {
            "name": "nomic-embed-text:latest",
            "size": 274302450,
            "modified_at": "2026-07-20T10:00:00Z",
            "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"},
        },
    ]
}


def mock_backend(monkeypatch, handler):
    """Route every AsyncClient this process builds through a MockTransport,
    so the manager's own client construction is exercised unchanged."""
    real_client = httpx.AsyncClient

    def build(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", build)


def manager() -> OllamaModelManager:
    return OllamaModelManager(base_url="http://localhost:11434", timeout_seconds=1.0)


async def test_list_installed_maps_backend_fields_and_infers_roles(monkeypatch):
    mock_backend(monkeypatch, lambda request: httpx.Response(200, json=TAGS_RESPONSE))

    installed = await manager().list_installed()

    assert [m.name for m in installed] == ["nomic-embed-text:latest", "qwen2.5-coder:7b"]
    embed, chat = installed
    assert chat.role == "chat"
    assert chat.size_bytes == 4683087332
    assert chat.parameter_size == "7.6B"
    assert chat.quantization == "Q4_K_M"
    assert embed.role == "embedding"


async def test_list_installed_returns_empty_when_backend_has_no_models(monkeypatch):
    mock_backend(monkeypatch, lambda request: httpx.Response(200, json={"models": []}))
    assert await manager().list_installed() == []


async def test_list_installed_raises_connection_error_when_backend_is_down(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    mock_backend(monkeypatch, refuse)

    with pytest.raises(LLMConnectionError, match="Is `ollama serve` running"):
        await manager().list_installed()


async def test_list_loaded_returns_resident_model_names(monkeypatch):
    mock_backend(
        monkeypatch,
        lambda request: httpx.Response(200, json={"models": [{"name": "qwen2.5-coder:7b"}]}),
    )
    assert await manager().list_loaded() == ["qwen2.5-coder:7b"]


async def test_is_reachable_is_false_when_the_backend_refuses(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    mock_backend(monkeypatch, refuse)
    assert await manager().is_reachable() is False


async def test_pull_yields_progress_with_a_percentage_and_a_done_marker(monkeypatch):
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling abc123", "digest": "sha256:abc123", "total": 1000, "completed": 250},
        {"status": "pulling abc123", "digest": "sha256:abc123", "total": 1000, "completed": 1000},
        {"status": "verifying sha256 digest"},
        {"status": "success"},
    ]
    body = "".join(json.dumps(line) + "\n" for line in lines).encode()
    mock_backend(monkeypatch, lambda request: httpx.Response(200, content=body))

    updates = [u async for u in manager().pull("qwen2.5-coder:7b")]

    assert [u.status for u in updates] == [
        "pulling manifest",
        "pulling abc123",
        "pulling abc123",
        "verifying sha256 digest",
        "success",
    ]
    assert [u.percent for u in updates] == [None, 25.0, 100.0, None, None]
    assert [u.done for u in updates] == [False, False, False, False, True]


async def test_pull_reports_a_backend_error_as_a_terminal_update(monkeypatch):
    # The caller is streaming these to a browser — an exception raised after
    # the response headers are out would strand the UI with no reason shown.
    mock_backend(
        monkeypatch,
        lambda request: httpx.Response(200, content=b'{"error":"file does not exist"}\n'),
    )

    updates = [u async for u in manager().pull("not-a-model")]

    assert len(updates) == 1
    assert updates[0].done is True
    assert updates[0].error == "file does not exist"


async def test_pull_reports_an_http_rejection_as_a_terminal_update(monkeypatch):
    mock_backend(monkeypatch, lambda request: httpx.Response(500, json={"error": "pull model manifest: 404"}))

    updates = [u async for u in manager().pull("nope:1b")]

    assert updates[-1].done is True
    assert updates[-1].error == "pull model manifest: 404"


async def test_pull_reports_an_unreachable_backend_as_a_terminal_update(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    mock_backend(monkeypatch, refuse)

    updates = [u async for u in manager().pull("qwen2.5-coder:7b")]

    assert updates[-1].done is True
    assert "Is `ollama serve` running" in updates[-1].error


async def test_pull_skips_unparseable_lines(monkeypatch):
    mock_backend(
        monkeypatch,
        lambda request: httpx.Response(200, content=b'not json\n{"status":"success"}\n'),
    )

    updates = [u async for u in manager().pull("qwen2.5-coder:7b")]

    assert [u.status for u in updates] == ["success"]


async def test_delete_succeeds_on_a_200(monkeypatch):
    seen = {}

    def handler(request):
        seen["method"] = request.method
        seen["body"] = json.loads(request.content)
        return httpx.Response(200)

    mock_backend(monkeypatch, handler)
    await manager().delete("codellama:7b")

    assert seen["method"] == "DELETE"
    assert seen["body"] == {"model": "codellama:7b"}


async def test_delete_raises_when_the_model_is_not_installed(monkeypatch):
    mock_backend(monkeypatch, lambda request: httpx.Response(404))

    with pytest.raises(ModelManagementError, match="not installed"):
        await manager().delete("codellama:7b")


def test_factory_builds_a_manager_for_the_configured_provider():
    assert isinstance(build_model_manager(Settings(_env_file=None)), OllamaModelManager)


def test_factory_rejects_a_provider_with_no_model_management():
    settings = Settings(_env_file=None, llm_provider="not-a-real-provider")
    with pytest.raises(ValueError, match="does not support model management"):
        build_model_manager(settings)
