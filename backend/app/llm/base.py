from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Provider-agnostic interface for chat-style local LLM inference.

    Nothing outside `app/llm/` may depend on a specific backend (e.g. Ollama).
    Swapping the local inference engine means writing one new class here and
    registering it in `factory.py` — the agent runtime is untouched.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Run a single non-streaming chat completion.

        `json_mode` constrains decoding to valid JSON where the backend
        supports it (Ollama's `format: "json"`) — this is what the agent
        loop uses for reliable structured tool selection with local models
        that don't consistently populate a native tool_calls field. It only
        guarantees syntactically valid JSON, not schema conformance — callers
        still validate the parsed result against their own Pydantic model.
        """

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield response text incrementally as it is generated."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and the model is available."""
