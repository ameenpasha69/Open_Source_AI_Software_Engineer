import pytest
from app.llm.base import LLMProvider, LLMResponse, Message


class FakeLLMProvider(LLMProvider):
    """Deterministic in-memory LLMProvider for tests — no network, no Ollama."""

    def __init__(self, response_text: str = "fake response", reachable: bool = True):
        self._response_text = response_text
        self._reachable = reachable
        self.received_messages: list[list[Message]] = []

    async def generate(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse:
        self.received_messages.append(messages)
        return LLMResponse(
            content=self._response_text,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=1.0,
            finish_reason="stop",
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None):
        for token in self._response_text.split():
            yield token + " "

    async def health_check(self) -> bool:
        return self._reachable


@pytest.fixture
def fake_llm_provider() -> FakeLLMProvider:
    return FakeLLMProvider()
