from app.agents.repetition import describe_call
from app.agents.state import AgentState, ConversationTurn, ToolCallRecord
from app.llm.base import Message
from app.tools.base import ToolSpec

_MAX_CONTENT_PREVIEW_CHARS = 800
_MAX_RECENT_OBSERVATIONS = 12
_MAX_LISTED_CALL_SIGNATURES = 25
_MAX_CONVERSATION_TURNS = 8
_MAX_TURN_PREVIEW_CHARS = 600

_SYSTEM_PROMPT_TEMPLATE = """You are a software engineering agent working on a local repository. Some tasks are \
bug reports to diagnose; others simply ask you to write or change code. Either way: read the relevant \
code first, then act. Gather only the evidence you actually need — enough to make the change correctly — \
and then make the smallest safe change with apply_patch. Never cite a location, or claim a root cause, \
you have not actually observed through a tool result.

Available tools (you are already scoped to one repository — never include "repository_id" yourself):
{tool_descriptions}

The search tools only ever look inside THIS repository's own indexed code. They are not a web search, \
a reference manual, or a way to look up how to write an algorithm — if a query comes back with nothing \
relevant, the answer is not in this repository and searching again with different wording will not find \
it. Write the code yourself instead.

search_code finds code by MEANING, not by filename — it embeds your query text and compares it against \
what each chunk of code actually says, so searching for a literal path like "adv_calc.py" mostly checks \
whether that string appears inside some file's content, not whether a file by that name exists. An empty \
or irrelevant search_code result is never proof that a file is missing. To find out whether a specific \
path exists, use list_files (to see what's actually in a directory) or read_file / get_file_context (to \
open the exact path directly) — never conclude "this file does not exist" from search_code alone. Files \
you (or an earlier run) create, change, or delete are re-indexed automatically, so search_code and \
find_symbol reflect an edit on the very next call — but that only helps once you've actually looked; it \
doesn't help if you never call list_files or read_file to check.

Once you have read the code you need to change, change it. Re-reading it, or searching for a better way \
to phrase what you already understand, is not progress — it burns the iteration budget and reaches no \
conclusion. When the task asks you to add or fix code and you can see the code in your observations, the \
next action is apply_patch, create_file, or delete_file.

apply_patch replaces an exact, unique excerpt of an EXISTING file (old_content) with new content — it is \
not a unified diff, and old_content must match the file's actual current content exactly (re-read the \
file first if you're not sure). Prefer the smallest change that addresses the root cause over rewriting \
whole functions. It cannot create a file that doesn't exist yet — for that, use create_file instead \
(it also creates any missing parent directory, e.g. a repository with no tests/ folder yet). There is no \
other way to create or delete a file: a shell command like touch, mkdir, or rm is not on the allowed \
command list, and retrying one after it's rejected wastes iterations without ever succeeding.\n\nThere is no separate rename or move tool. To rename a file, call create_file ONCE with the new path \
and the complete final content already in it, then delete_file the old path — do not leave the old file \
behind and call that done. Never call create_file with empty or placeholder content meaning to fill it \
in afterward: apply_patch cannot add content to an empty file (old_content can never match nothing), so \
that leaves the file permanently stuck empty. Write the real, complete content in the same create_file \
call that makes the file.

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

    if tool_name in ("apply_patch", "create_file", "delete_file"):
        stats = f"+{output.get('lines_added', 0)}/-{output.get('lines_removed', 0)} lines"
        verb = {"apply_patch": "patched", "create_file": "created", "delete_file": "deleted"}[tool_name]
        header = f"{verb} {output.get('path')} ({stats}, unverified — no test run yet)"
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


def format_conversation(turns: list[ConversationTurn]) -> str:
    """Render the session's earlier turns for the prompt.

    Only the last few, and each one truncated: earlier turns are context for
    *what is being asked now*, not evidence to reason from — this run gathers
    its own evidence through tools. Letting a long session's full transcript
    into every prompt would crowd out the observations that actually matter
    and push the context window up for no gain.
    """
    if not turns:
        return ""
    recent = turns[-_MAX_CONVERSATION_TURNS:]
    lines = []
    for turn in recent:
        label = "User" if turn.role == "user" else "You"
        body = turn.content.strip()
        if len(body) > _MAX_TURN_PREVIEW_CHARS:
            body = body[:_MAX_TURN_PREVIEW_CHARS] + "... [truncated]"
        lines.append(f"{label}: {body}")
    dropped = len(turns) - len(recent)
    header = "Earlier in this conversation"
    if dropped:
        header += f" (showing the last {len(recent)} of {len(turns)} turns)"
    return (
        f"{header} — the request below may refer back to it, but re-verify anything "
        f"you intend to act on with a tool rather than trusting it:\n" + "\n".join(lines)
    )


def _wrap_up_threshold(max_iterations: int) -> int:
    """How many iterations from the end to start telling the model to wrap
    up. Proportional, not a fixed 2: with MAX_AGENT_ITERATIONS=100 a fixed
    threshold fires at iteration 98, long after a stuck run is worth
    salvaging. At least 2 so a short budget still gets a warning at all.
    """
    return max(2, max_iterations // 4)


def _format_calls_already_made(state: AgentState) -> str:
    """A flat list of every distinct call made this run, with the iterations
    it happened on.

    Observations are a sliding window of the most recent few, which is what
    keeps the prompt small — but it means a model that starts repeating sees
    a window full of its own repetitions and no memory of the distinct things
    it tried twenty iterations ago. This list is outside the window and costs
    one line per distinct call, so "what have I already tried" survives even
    when the observation itself has scrolled off.
    """
    iterations_by_call: dict[str, list[int]] = {}
    for record in state.tool_calls:
        iterations_by_call.setdefault(describe_call(record.tool_name, record.input), []).append(record.iteration)

    if not iterations_by_call:
        return "Calls already made:\n(none yet — this is the first iteration)"

    lines = []
    for call, iterations in list(iterations_by_call.items())[-_MAX_LISTED_CALL_SIGNATURES:]:
        where = ", ".join(str(i) for i in iterations)
        lines.append(f"- {call} [iter {where}]")

    return (
        "Calls already made (repeating one with the same arguments returns the same result "
        "and will be skipped, unless you have changed the repository since):\n" + "\n".join(lines)
    )


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
        if remaining <= _wrap_up_threshold(max_iterations):
            budget_line += (
                " You are close to the iteration limit — if you have enough evidence to explain the root "
                "cause, use 'finish' now rather than continuing to investigate."
            )

        conversation_text = format_conversation(state.conversation)
        conversation_block = f"{conversation_text}\n\n" if conversation_text else ""

        user = (
            f"{budget_line}\n\n"
            f"{conversation_block}"
            f"Task:\n{state.task}\n\n"
            f"Plan (a rough starting guide, written before any investigation — not a literal script. "
            f"A step may already be satisfied by something in Observations below, may turn out not to "
            f"apply, or may need to be done differently than worded. Decide your next action from what "
            f"Observations actually show, not by restating a plan step verbatim):\n{plan_text}\n\n"
            f"{_format_calls_already_made(state)}\n\n"
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
