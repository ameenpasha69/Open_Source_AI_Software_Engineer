import json
import logging

from app.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a senior software engineer. Given a bug report and a repository name, "
    "produce a short investigation plan — 3 to 6 concrete steps a systematic engineer "
    "would take to locate the root cause before proposing a fix. Do not propose a fix yet.\n\n"
    'Respond with ONLY a JSON object matching this schema: {"steps": [string, ...]}. '
    "No prose outside the JSON."
)

_FALLBACK_PLAN = [
    "Search the codebase semantically for code related to the reported issue.",
    "Read the most relevant files in full to understand the surrounding logic.",
    "Look for the specific function or code path the issue describes.",
    "Check recent git history and diffs for anything that could explain the behavior.",
    "Form a hypothesis about the root cause, citing specific file and line locations.",
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
