import json
import time
from collections.abc import AsyncIterator

import httpx

from app.llm.base import LLMProvider, LLMResponse, Message
from app.llm.exceptions import LLMConnectionError, LLMModelNotFoundError, LLMTimeoutError


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server (https://ollama.com).

    Uses Ollama's /api/chat endpoint (not /api/generate) so the abstraction is
    already message/role based, which the agent's tool-calling loop needs.
    """

    def __init__(self, base_url: str, model: str, timeout_seconds: float, default_temperature: float):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._default_temperature = default_temperature

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = self._build_payload(messages, temperature, max_tokens, stream=False)
        started = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(f"{self._base_url}/api/chat", json=payload)
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not respond within {self._timeout}s") from exc

        if resp.status_code == 404:
            raise LLMModelNotFoundError(
                f"Model '{self._model}' not found on Ollama. Run `ollama pull {self._model}`."
            )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.monotonic() - started) * 1000

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=self._model,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason"),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._build_payload(messages, temperature, max_tokens, stream=True)
        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream("POST", f"{self._base_url}/api/chat", json=payload) as resp,
            ):
                if resp.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model '{self._model}' not found on Ollama. Run `ollama pull {self._model}`."
                    )
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                f"Could not reach Ollama at {self._base_url}. Is `ollama serve` running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Ollama did not respond within {self._timeout}s") from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(m == self._model or m.startswith(f"{self._model}:") for m in models)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def _build_payload(
        self,
        messages: list[Message],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict:
        options: dict = {"temperature": temperature if temperature is not None else self._default_temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        return {
            "model": self._model,
            "messages": [m.model_dump() for m in messages],
            "stream": stream,
            "options": options,
        }
