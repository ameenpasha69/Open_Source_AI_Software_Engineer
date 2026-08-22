"""Counting what a run actually spent on the model.

Every LLM call in a run goes through one of two places — the planner and the
agent loop — and both already had a provider handed to them. Rather than
threading a counter through both call sites (and every future one), the
runner wraps the provider once and the wrapper accumulates. Nothing that
calls `generate()` needs to know it is being metered.

Token counts come from the backend's own reported usage (Ollama's
`prompt_eval_count` / `eval_count`), not from a client-side estimate — an
approximation would be worse than useless for a context-window gauge, since
the whole point is knowing how close a prompt is to the model's real limit.
"""

from collections.abc import AsyncIterator

from pydantic import BaseModel

from app.llm.base import LLMProvider, LLMResponse, Message


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    call_count: int = 0
    # The largest single prompt sent — this, not the running total, is what
    # a context-window gauge means: how close one request came to the model's
    # limit. Totals across a run can exceed the window many times over
    # without any individual call being anywhere near it.
    peak_prompt_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, response: LLMResponse) -> None:
        self.call_count += 1
        prompt = response.prompt_tokens or 0
        self.prompt_tokens += prompt
        self.completion_tokens += response.completion_tokens or 0
        self.peak_prompt_tokens = max(self.peak_prompt_tokens, prompt)

    def merge(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            call_count=self.call_count + other.call_count,
            peak_prompt_tokens=max(self.peak_prompt_tokens, other.peak_prompt_tokens),
        )


class UsageTrackingLLMProvider(LLMProvider):
    """Delegates every call to the real provider, accumulating token usage.

    `stream()` is passed through unmetered: Ollama only reports token counts
    on the final chunk of a stream, and nothing in the agent loop streams
    today — counting it wrongly would be worse than not counting it, so this
    stays honest about what it measures.
    """

    def __init__(self, inner: LLMProvider):
        self._inner = inner
        self.usage = TokenUsage()

    @property
    def model(self) -> str:
        return self._inner.model

    async def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        response = await self._inner.generate(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        self.usage.record(response)
        return response

    async def stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async for chunk in self._inner.stream(messages, temperature=temperature, max_tokens=max_tokens):
            yield chunk

    async def health_check(self) -> bool:
        return await self._inner.health_check()
