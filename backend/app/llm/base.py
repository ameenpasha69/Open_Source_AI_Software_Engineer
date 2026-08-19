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
    ) -> LLMResponse:
        """Run a single non-streaming chat completion."""

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
