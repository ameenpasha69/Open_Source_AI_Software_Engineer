"""Installing, removing, and inspecting the models available on the local
inference backend.

Kept behind a `ModelManager` interface for the same reason `LLMProvider` is:
nothing outside `app/llm/` should know that the backend happens to be
Ollama. Swapping in another local engine means one new class here plus a
line in `factory.py`.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx
from pydantic import BaseModel

from app.llm.catalog import ModelRole, infer_role
from app.llm.exceptions import LLMConnectionError, LLMProviderError, LLMTimeoutError


class ModelManagementError(LLMProviderError):
    """A model install/remove operation was rejected by the backend."""


class InstalledModel(BaseModel):
    """A model already present on the backend and ready to be used."""

    name: str
    role: ModelRole
    size_bytes: int
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    modified_at: str | None = None
    # The model's real context window in tokens, as the backend reports it.
    # Newer Ollama includes this in /api/tags; older versions don't, hence
    # None rather than a guessed default — the UI falls back to the catalog.
    context_length: int | None = None


class PullProgress(BaseModel):
    """One progress update from an in-flight download.

    `percent` is only meaningful while layer bytes are being transferred —
    the surrounding phases ("pulling manifest", "verifying sha256 digest")
    report status text with no byte counts, and leave it None.
    """

    status: str
    digest: str | None = None
    completed: int | None = None
    total: int | None = None
    percent: float | None = None
    done: bool = False
    error: str | None = None


class ModelManager(ABC):
    @abstractmethod
    async def list_installed(self) -> list[InstalledModel]:
        """Every model pulled onto the backend, chat and embedding alike."""

    @abstractmethod
    async def list_loaded(self) -> list[str]:
        """Names of models currently resident in memory — these answer with
        no load latency, which is worth surfacing when choosing between two
        otherwise equivalent models."""

    @abstractmethod
    def pull(self, name: str) -> AsyncIterator[PullProgress]:
        """Download a model, yielding progress as it goes.

        Failures are yielded as a terminal `PullProgress(error=...)` rather
        than raised: the caller is streaming these to a browser and needs to
        deliver the reason over the same channel, not lose it to an
        exception thrown after the response headers are already sent.
        """

    @abstractmethod
    async def delete(self, name: str) -> None:
        """Remove a model from the backend, freeing its disk space."""

    @abstractmethod
    async def is_reachable(self) -> bool:
        """True if the backend is up, regardless of which models it holds."""


class OllamaModelManager(ModelManager):
    def __init__(self, base_url: str, timeout_seconds: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

    @property
    def base_url(self) -> str:
        return self._base_url

    async def list_installed(self) -> list[InstalledModel]:
        data = await self._get("/api/tags")
        models = []
        for raw in data.get("models", []):
            details = raw.get("details") or {}
            name = raw.get("name") or raw.get("model", "")
            if not name:
                continue
            models.append(
                InstalledModel(
                    name=name,
                    role=infer_role(name, details.get("family")),
                    size_bytes=raw.get("size", 0),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                    family=details.get("family"),
                    modified_at=raw.get("modified_at"),
                    context_length=details.get("context_length"),
                )
            )
        return sorted(models, key=lambda m: m.name)

    async def list_loaded(self) -> list[str]:
        data = await self._get("/api/ps")
        return [m.get("name") or m.get("model", "") for m in data.get("models", []) if m.get("name") or m.get("model")]

    async def pull(self, name: str) -> AsyncIterator[PullProgress]:
        # No read timeout: a multi-gigabyte download legitimately runs for
        # many minutes, and Ollama emits a progress line frequently enough
        # that a stalled connection shows up as silence in the UI instead.
        timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
        try:
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    "POST", f"{self._base_url}/api/pull", json={"model": name, "stream": True}
                ) as resp,
            ):
                if resp.status_code >= 400:
                    body = await resp.aread()
                    yield PullProgress(
                        status="failed", done=True, error=_error_detail(body, name, resp.status_code)
                    )
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    progress = _parse_pull_line(line)
                    if progress is not None:
                        yield progress
        except httpx.ConnectError:
            yield PullProgress(
                status="failed",
                done=True,
                error=f"Could not reach the model backend at {self._base_url}. Is `ollama serve` running?",
            )
        except httpx.HTTPError as exc:
            yield PullProgress(status="failed", done=True, error=f"Download failed: {exc}")

    async def delete(self, name: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.request(
                    "DELETE", f"{self._base_url}/api/delete", json={"model": name}
                )
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Could not reach the model backend at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"The model backend did not respond within {self._timeout}s") from exc

        if resp.status_code == 404:
            raise ModelManagementError(f"Model '{name}' is not installed.")
        if resp.status_code >= 400:
            raise ModelManagementError(_error_detail(resp.content, name, resp.status_code))

    async def is_reachable(self) -> bool:
        try:
            await self._get("/api/tags")
        except LLMProviderError:
            return False
        return True

    async def _get(self, path: str) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}{path}")
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Could not reach the model backend at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"The model backend did not respond within {self._timeout}s") from exc
        resp.raise_for_status()
        return resp.json()


def _parse_pull_line(line: str) -> PullProgress | None:
    try:
        chunk = json.loads(line)
    except json.JSONDecodeError:
        return None

    if error := chunk.get("error"):
        return PullProgress(status="failed", done=True, error=str(error))

    status = str(chunk.get("status", ""))
    total = chunk.get("total")
    completed = chunk.get("completed")
    percent = None
    if isinstance(total, int) and total > 0 and isinstance(completed, int):
        percent = round(min(completed / total, 1.0) * 100, 1)

    return PullProgress(
        status=status,
        digest=chunk.get("digest"),
        completed=completed,
        total=total,
        percent=percent,
        # Ollama's final line for a successful pull is literally {"status": "success"}.
        done=status == "success",
    )


def _error_detail(body: bytes, name: str, status_code: int) -> str:
    try:
        detail = json.loads(body).get("error")
    except (json.JSONDecodeError, AttributeError):
        detail = None
    if detail:
        return str(detail)
    return f"The model backend rejected the request for '{name}' (HTTP {status_code})."
