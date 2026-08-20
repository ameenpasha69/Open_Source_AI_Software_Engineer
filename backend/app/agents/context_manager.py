from app.agents.state import AgentState, ToolCallRecord
from app.llm.base import Message
from app.tools.base import ToolSpec

_MAX_CONTENT_PREVIEW_CHARS = 800
_MAX_RECENT_OBSERVATIONS = 12

_SYSTEM_PROMPT_TEMPLATE = """You are a software engineering agent investigating a reported issue in a \
local repository. Gather evidence with the tools below before changing anything: read the relevant code, \
understand the root cause, and only then make the smallest safe fix with apply_patch. Never cite a \
location, or claim a root cause, you have not actually observed through a tool result.

Available tools (you are already scoped to one repository — never include "repository_id" yourself):
{tool_descriptions}

apply_patch replaces an exact, unique excerpt of a file (old_content) with new content — it is not a \
unified diff, and old_content must match the file's actual current content exactly (re-read the file \
first if you're not sure). Prefer the smallest change that addresses the root cause over rewriting \
whole functions.

After applying a patch, run_tests to check your work. If tests fail, read the failure (failed_tests, \
failure_category) and iterate: adjust your patch and run_tests again, rather than finishing on a change \
you haven't verified. run_tests may not be available for every repository (no test command is configured) \
— if so, say the fix is unverified rather than claiming it works.

Each turn, respond with ONLY a JSON object matching this schema, no other text:
{{"thought": string, "action": {{"tool": string, "input": object}} | null, "finish": {{"answer": string, \
"root_cause": string | null}} | null}}

Set exactly one of "action" or "finish". Use "action" to call a tool and gather more evidence, apply a \
fix, or verify one, with "input" containing exactly the parameters listed for that tool. Use "finish" \
once you have enough evidence to explain the root cause (having applied and, where possible, verified a \
fix, if the issue calls for a code change), or if you have exhausted reasonable avenues of investigation \
and must report what you found so far."""


def _format_tool_signature(spec: ToolSpec) -> str:
    """Renders `name(param: type, optional_param: type = default): description`
    from the tool's JSON schema, omitting `repository_id` — the runtime
    injects that automatically (see AgentRunner._execute_action), so showing
    it as a parameter the model must fill in would just invite a wrong guess.
    """
    properties = spec.input_schema.get("properties", {})
    required = set(spec.input_schema.get("required", []))
    params = []
    for name, prop_schema in properties.items():
        if name == "repository_id":
            continue
        type_name = _schema_type(prop_schema)
        if name in required:
            params.append(f"{name}: {type_name}")
        else:
            params.append(f"{name}: {type_name} = {prop_schema.get('default')!r}")
    return f"{spec.name}({', '.join(params)}): {spec.description}"


def _schema_type(prop_schema: dict) -> str:
    if "type" in prop_schema:
        return prop_schema["type"]
    for option in prop_schema.get("anyOf", []):
        if option.get("type") != "null":
            return option.get("type", "any")
    return "any"


def format_observation(record: ToolCallRecord) -> str:
    """Turns a raw ToolResult into a compact, LLM-readable summary — this is
    the "Observation Handler": what actually accumulates in agent memory and
    gets fed back into the next prompt is a digest, not the full JSON blob,
    which is what keeps the context small as iterations pile up.
    """
    call_desc = f"[iter {record.iteration}] {record.tool_name}({_format_input(record.input)})"
    if not record.result.success:
        return f"{call_desc} -> FAILED: {record.result.error}"

    output = record.result.output or {}
    summary = _summarize_output(record.tool_name, output)
    return f"{call_desc} -> {summary}"


def _format_input(input_data: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in input_data.items())


def _summarize_output(tool_name: str, output: dict) -> str:
    if tool_name == "search_code":
        results = output.get("results", [])
        if not results:
            return "no results"
        lines = [f"{r['location']} ({r['symbol'] or 'module-level'}, score={r['score']:.2f})" for r in results]
        return f"{len(results)} result(s): " + "; ".join(lines)

    if tool_name == "find_symbol":
        matches = output.get("matches", [])
        if not matches:
            return "no matches"
        lines = [f"{m['file_path']}:{m['start_line']}-{m['end_line']} ({m['symbol']})" for m in matches]
        return f"{len(matches)} match(es): " + "; ".join(lines)

    if tool_name == "find_references":
        refs = output.get("references", [])
        if not refs:
            return "no references"
        lines = [f"{r['file_path']}:{r['start_line']}-{r['end_line']}" for r in refs]
        return f"{len(refs)} reference(s): " + "; ".join(lines)

    if tool_name == "list_files":
        entries = output.get("entries", [])
        names = [e["path"] + ("/" if e["is_dir"] else "") for e in entries]
        return f"{len(entries)} entries: " + ", ".join(names)

    if tool_name in ("read_file", "get_file_context"):
        content = output.get("content", "")
        preview = _truncate(content)
        line_range = f"lines {output.get('start_line')}-{output.get('end_line')} of {output.get('total_lines')}"
        return f"{line_range}:\n{preview}"

    if tool_name == "get_git_status":
        return (
            f"branch={output.get('branch')}, clean={output.get('is_clean')}, "
            f"staged={output.get('staged')}, unstaged={output.get('unstaged')}, "
            f"untracked={output.get('untracked')}"
        )

    if tool_name == "get_git_diff":
        return _truncate(output.get("diff", ""))

    if tool_name == "get_git_log":
        commits = output.get("commits", [])
        lines = [f"{c['commit_hash'][:8]} {c['message']}" for c in commits]
        return f"{len(commits)} commit(s): " + "; ".join(lines)

    if tool_name == "apply_patch":
        stats = f"+{output.get('lines_added', 0)}/-{output.get('lines_removed', 0)} lines"
        header = f"patched {output.get('path')} ({stats}, unverified — no test run yet)"
        return f"{header}:\n{_truncate(output.get('diff', ''))}"

    if tool_name == "run_tests":
        return _summarize_test_run(output)

    if tool_name in ("run_command", "run_linter", "run_formatter"):
        status = "passed" if output.get("passed") else f"failed (exit {output.get('exit_code')})"
        combined = (output.get("stdout", "") + output.get("stderr", "")).strip()
        return f"{output.get('command')} -> {status}:\n{_truncate(combined)}"

    # No tool-specific summarizer — fall back to a truncated generic dump.
    return _truncate(str(output))


def _summarize_test_run(output: dict) -> str:
    if output.get("passed"):
        total = output.get("total_tests")
        counts = f"{output.get('passed_tests')}/{total} passed" if total is not None else "passed"
        return f"PASSED ({output.get('scope')} run, {counts})"

    category = output.get("failure_category")
    if category == "test_failure":
        failed = output.get("failed_tests") or []
        failed_preview = "; ".join(failed[:10]) + (" ..." if len(failed) > 10 else "")
        return f"FAILED ({output.get('scope')} run) — {len(failed)} failing test(s): {failed_preview}"

    # syntax_error / dependency_error / environment_error / timeout: the run
    # itself didn't complete meaningfully, so surface raw output instead of a
    # test-count summary that wouldn't mean anything here.
    combined = (output.get("stdout", "") + output.get("stderr", "")).strip()
    return f"FAILED TO RUN ({category}):\n{_truncate(combined)}"


def _truncate(text: str) -> str:
    if len(text) <= _MAX_CONTENT_PREVIEW_CHARS:
        return text
    return text[:_MAX_CONTENT_PREVIEW_CHARS] + "... [truncated]"


class ContextManager:
    """Builds the message list sent to the LLM each iteration, from
    AgentState rather than an ever-growing chat history. Only the most
    recent `max_recent_observations` tool results are included in full —
    older ones are dropped from the prompt (they remain in the persisted
    AgentState/DB either way, just not re-sent to the model every turn).
    This is the "smallest useful context" tradeoff from the retrieval
    design goals, applied to agent memory instead of code retrieval.
    """

    def __init__(self, max_recent_observations: int = _MAX_RECENT_OBSERVATIONS):
        self._max_recent_observations = max_recent_observations

    def build_messages(self, state: AgentState, tool_specs: list[ToolSpec], max_iterations: int) -> list[Message]:
        tool_descriptions = "\n".join(f"- {_format_tool_signature(spec)}" for spec in tool_specs)
        system = _SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)

        plan_text = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(state.plan))
        recent = state.observations[-self._max_recent_observations :]
        observations_text = "\n".join(recent) if recent else "(none yet — this is the first iteration)"

        remaining = max_iterations - state.iteration
        budget_line = f"Iteration {state.iteration + 1} of {max_iterations} ({remaining} remaining after this one)."
        if remaining <= 2:
            budget_line += (
                " You are close to the iteration limit — if you have enough evidence to explain the root "
                "cause, use 'finish' now rather than continuing to investigate."
            )

        user = (
            f"{budget_line}\n\n"
            f"Task:\n{state.task}\n\n"
            f"Plan:\n{plan_text}\n\n"
            f"Observations so far:\n{observations_text}\n\n"
            f"What do you want to do next?"
        )
        return [Message(role="system", content=system), Message(role="user", content=user)]

    def estimate_context_chars(self, messages: list[Message]) -> int:
        """A cheap proxy for prompt size (character count, not real token
        count) used for the observability logging in section 21 — good
        enough to answer "did context grow unexpectedly," not meant to be
        an exact token estimate.
        """
        return sum(len(m.content) for m in messages)
