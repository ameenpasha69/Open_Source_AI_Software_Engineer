import asyncio

from app.tools.base import Tool, ToolError, ToolExecutor, ToolRegistry
from pydantic import BaseModel


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    value: str


class EchoTool(Tool):
    name = "echo"
    description = "Echoes its input back."
    input_schema = EchoInput
    output_schema = EchoOutput
    timeout_seconds = 5.0

    async def run(self, input_data: EchoInput) -> EchoOutput:
        return EchoOutput(value=input_data.value)


class FailingTool(Tool):
    name = "failing"
    description = "Always raises a ToolError."
    input_schema = EchoInput
    output_schema = EchoOutput

    async def run(self, input_data: EchoInput) -> EchoOutput:
        raise ToolError("deliberate failure")


class CrashingTool(Tool):
    name = "crashing"
    description = "Raises a plain, unexpected exception."
    input_schema = EchoInput
    output_schema = EchoOutput

    async def run(self, input_data: EchoInput) -> EchoOutput:
        raise RuntimeError("bug!")


class SlowTool(Tool):
    name = "slow"
    description = "Sleeps longer than its own timeout."
    input_schema = EchoInput
    output_schema = EchoOutput
    timeout_seconds = 0.05

    async def run(self, input_data: EchoInput) -> EchoOutput:
        await asyncio.sleep(1)
        return EchoOutput(value=input_data.value)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in (EchoTool(), FailingTool(), CrashingTool(), SlowTool()):
        registry.register(tool)
    return registry


def test_registry_lists_specs_with_json_schema():
    specs = _registry().list_specs()
    echo_spec = next(s for s in specs if s.name == "echo")
    assert echo_spec.description == "Echoes its input back."
    assert "value" in echo_spec.input_schema["properties"]


def test_registry_get_unknown_tool_raises_tool_error():
    try:
        _registry().get("does-not-exist")
        raise AssertionError("expected ToolError")
    except ToolError as exc:
        assert "does-not-exist" in str(exc)


async def test_executor_success_path():
    result = await ToolExecutor(_registry()).execute("echo", {"value": "hi"})
    assert result.success is True
    assert result.output == {"value": "hi"}
    assert result.error is None


async def test_executor_unknown_tool_returns_failed_result_not_exception():
    result = await ToolExecutor(_registry()).execute("nope", {"value": "hi"})
    assert result.success is False
    assert "Unknown tool" in result.error


async def test_executor_invalid_input_returns_failed_result():
    result = await ToolExecutor(_registry()).execute("echo", {"wrong_field": 1})
    assert result.success is False
    assert "Invalid input" in result.error


async def test_executor_tool_error_is_surfaced_as_failed_result():
    result = await ToolExecutor(_registry()).execute("failing", {"value": "x"})
    assert result.success is False
    assert result.error == "deliberate failure"


async def test_executor_unexpected_exception_does_not_propagate():
    result = await ToolExecutor(_registry()).execute("crashing", {"value": "x"})
    assert result.success is False
    assert "bug!" in result.error


async def test_executor_enforces_timeout():
    result = await ToolExecutor(_registry()).execute("slow", {"value": "x"})
    assert result.success is False
    assert "timed out" in result.error


async def test_executor_result_always_reports_duration():
    result = await ToolExecutor(_registry()).execute("echo", {"value": "x"})
    assert result.duration_seconds >= 0
