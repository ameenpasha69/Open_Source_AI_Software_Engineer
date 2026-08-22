import json
import logging

from app.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are planning the first few steps for an autonomous coding agent working alone in one "
    "repository — no branches, no design-document step, no peer review, no ticket system. It reads "
    "and searches code, applies changes directly to files, and can run tests if a test command is "
    "configured. Given a task and a repository name, write 3 to 6 concrete steps, each one something "
    "this agent can actually do with those capabilities: read/search for the relevant code, work out "
    "what needs to change, make the change, and verify it if a test command exists. Never include a "
    "step with no corresponding capability here (branching, a design document, peer review, filing a "
    "ticket) — every step must cash out as reading/searching, changing a file, or running tests.\n\n"
    "The task may be a bug report to diagnose, or it may simply ask for code to be added, changed, "
    "renamed, or rewritten — plan for whichever this one actually is; do not assume it is always a "
    "bug to investigate. Do not propose the actual fix or content yet, just the steps to get there.\n\n"
    'Respond with ONLY a JSON object matching this schema: {"steps": [string, ...]}. '
    "No prose outside the JSON."
)

_FALLBACK_PLAN = [
    "Search the codebase semantically for code related to the task.",
    "Read the most relevant files in full to understand the surrounding logic.",
    "Locate the specific function, file, or code path the task concerns.",
    "Work out exactly what needs to change, citing specific file and line locations.",
    "Make the change, then run tests to verify it if a test command is configured.",
]


class Planner:
    """Produces the initial investigation plan, stored in AgentState before
    the tool-calling loop starts (spec section 13: the plan is explicit,
    persisted state — not something implied by a chat transcript).
    """

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    async def create_plan(self, task: str, repository_name: str) -> list[str]:
        response = await self._llm.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=f"Repository: {repository_name}\n\nIssue:\n{task}"),
            ],
            json_mode=True,
        )

        try:
            data = json.loads(response.content)
            steps = [s.strip() for s in data["steps"] if isinstance(s, str) and s.strip()]
            if not steps:
                raise ValueError("empty plan")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Planner: could not parse plan from LLM response, using fallback plan")
            return list(_FALLBACK_PLAN)

        return steps
