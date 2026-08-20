from app.agents.context_manager import ContextManager, _format_tool_signature, format_observation
from app.agents.state import AgentState, ToolCallRecord
from app.llm.base import Message
from app.tools.base import ToolResult, ToolSpec


def _tool_call(tool_name, input_data, success=True, output=None, error=None):
    return ToolCallRecord(
        iteration=1,
        tool_name=tool_name,
        input=input_data,
        result=ToolResult(tool_name=tool_name, success=success, output=output, error=error, duration_seconds=0.1),
    )


def test_format_observation_for_failed_call():
    record = _tool_call("read_file", {"path": "x.py"}, success=False, error="not a file")
    assert format_observation(record) == "[iter 1] read_file(path='x.py') -> FAILED: not a file"


def test_format_observation_for_search_code():
    record = _tool_call(
        "search_code",
        {"query": "total"},
        output={
            "results": [
                {"location": "a.py:1-5", "symbol": "foo", "score": 0.91},
                {"location": "b.py:10-20", "symbol": None, "score": 0.5},
            ]
        },
    )
    text = format_observation(record)
    assert "a.py:1-5 (foo, score=0.91)" in text
    assert "b.py:10-20 (module-level, score=0.50)" in text


def test_format_observation_for_find_symbol_no_matches():
    record = _tool_call("find_symbol", {"symbol": "nope"}, output={"matches": []})
    assert format_observation(record).endswith("no matches")


def test_format_observation_truncates_long_file_content():
    long_content = "x" * 2000
    record = _tool_call(
        "read_file",
        {"path": "big.py"},
        output={"content": long_content, "start_line": 1, "end_line": 50, "total_lines": 50},
    )
    text = format_observation(record)
    assert "[truncated]" in text
    assert len(text) < 2000


def test_format_observation_unknown_tool_falls_back_to_generic_summary():
    record = _tool_call("some_future_tool", {}, output={"anything": "goes"})
    assert "anything" in format_observation(record)


def test_build_messages_includes_task_plan_and_tool_descriptions():
    state = AgentState(run_id="r1", task="Fix the 500 error", repository_id="repo1", plan=["Step one", "Step two"])
    tool_specs = [
        ToolSpec(
            name="read_file",
            description="Reads a file.",
            input_schema={
                "properties": {
                    "repository_id": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["repository_id", "path"],
            },
        )
    ]

    messages = ContextManager().build_messages(state, tool_specs, max_iterations=8)

    assert messages[0].role == "system"
    tool_line = "read_file(path: string): Reads a file."
    assert tool_line in messages[0].content
    assert "repository_id" not in tool_line
    assert messages[1].role == "user"
    assert "Fix the 500 error" in messages[1].content
    assert "1. Step one" in messages[1].content
    assert "2. Step two" in messages[1].content
    assert "this is the first iteration" in messages[1].content


def test_build_messages_includes_recent_observations():
    state = AgentState(run_id="r1", task="task", repository_id="repo1", observations=["obs one", "obs two"])
    messages = ContextManager().build_messages(state, [], max_iterations=8)
    assert "obs one" in messages[1].content
    assert "obs two" in messages[1].content


def test_build_messages_caps_to_max_recent_observations():
    state = AgentState(
        run_id="r1", task="task", repository_id="repo1", observations=[f"obs {i}" for i in range(20)]
    )
    messages = ContextManager(max_recent_observations=3).build_messages(state, [], max_iterations=8)
    assert "obs 19" in messages[1].content
    assert "obs 0" not in messages[1].content


def test_build_messages_shows_iteration_budget():
    state = AgentState(run_id="r1", task="task", repository_id="repo1", iteration=1)
    messages = ContextManager().build_messages(state, [], max_iterations=8)
    assert "Iteration 2 of 8" in messages[1].content


def test_build_messages_nudges_to_finish_near_iteration_limit():
    state = AgentState(run_id="r1", task="task", repository_id="repo1", iteration=6)
    messages = ContextManager().build_messages(state, [], max_iterations=8)
    assert "close to the iteration limit" in messages[1].content


def test_build_messages_no_nudge_when_far_from_iteration_limit():
    state = AgentState(run_id="r1", task="task", repository_id="repo1", iteration=1)
    messages = ContextManager().build_messages(state, [], max_iterations=8)
    assert "close to the iteration limit" not in messages[1].content


def test_estimate_context_chars_sums_message_lengths():
    messages = [Message(role="system", content="12345"), Message(role="user", content="123")]
    assert ContextManager().estimate_context_chars(messages) == 8


def test_format_tool_signature_shows_required_and_optional_params():
    spec = ToolSpec(
        name="find_symbol",
        description="Find where a symbol is defined.",
        input_schema={
            "properties": {
                "repository_id": {"type": "string"},
                "symbol": {"type": "string"},
                "exact": {"type": "boolean", "default": False},
            },
            "required": ["repository_id", "symbol"],
        },
    )
    signature = _format_tool_signature(spec)
    assert signature == "find_symbol(symbol: string, exact: boolean = False): Find where a symbol is defined."


def test_format_tool_signature_handles_optional_union_type():
    spec = ToolSpec(
        name="read_file",
        description="Reads a file.",
        input_schema={
            "properties": {
                "repository_id": {"type": "string"},
                "path": {"type": "string"},
                "start_line": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
            },
            "required": ["repository_id", "path"],
        },
    )
    signature = _format_tool_signature(spec)
    assert "start_line: integer = None" in signature
