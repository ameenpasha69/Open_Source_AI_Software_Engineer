import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Raised by a tool for an *expected* failure — bad input, not found,
    a git error, a path escaping the repository. Caught by ToolExecutor and
    turned into a failed ToolResult rather than propagating as a crash, so
    the agent sees "the tool reported a problem" as an observation instead
    of the run dying.
    """


class Tool(ABC):
    """A single agent-callable capability. Subclasses declare `name`,
    `description`, `input_schema`, and `output_schema` as class attributes —
    ToolRegistry.list_specs() turns these into what an LLM sees when
    choosing a tool, and ToolExecutor uses input_schema to validate
    arguments before `run()` is ever called.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    timeout_seconds: ClassVar[float] = 30.0

    @abstractmethod
    async def run(self, input_data: BaseModel) -> BaseModel: ...


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float


class ToolSpec(BaseModel):
    """What's shown to an LLM (or an API caller) about an available tool —
    deliberately excludes the implementation."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool '{name}'. Available: {sorted(self._tools)}")
        return tool

    def list_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema.model_json_schema())
            for t in self._tools.values()
        ]


class ToolExecutor:
    """Validates input, enforces the tool's timeout, and converts every
    failure mode (bad input, timeout, ToolError, and — as a last resort —
    any other exception) into a ToolResult instead of letting it propagate.
    A misbehaving tool should degrade one agent step, not crash the run.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, tool_name: str, raw_input: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        try:
            tool = self._registry.get(tool_name)
        except ToolError as exc:
            return self._failure(tool_name, str(exc), started)

        try:
            parsed_input = tool.input_schema.model_validate(raw_input)
        except ValidationError as exc:
            return self._failure(tool_name, f"Invalid input: {exc}", started)

        try:
            output = await asyncio.wait_for(tool.run(parsed_input), timeout=tool.timeout_seconds)
        except TimeoutError:
            return self._failure(tool_name, f"Tool '{tool_name}' timed out after {tool.timeout_seconds}s", started)
        except ToolError as exc:
            return self._failure(tool_name, str(exc), started)
        except Exception as exc:  # noqa: BLE001 - last-resort safety net, see class docstring
            return self._failure(tool_name, f"Unexpected tool error: {exc}", started)

        return ToolResult(
            tool_name=tool_name,
            success=True,
            output=output.model_dump(),
            duration_seconds=round(time.monotonic() - started, 3),
        )

    @staticmethod
    def _failure(tool_name: str, error: str, started: float) -> ToolResult:
        return ToolResult(
            tool_name=tool_name, success=False, error=error, duration_seconds=round(time.monotonic() - started, 3)
        )
