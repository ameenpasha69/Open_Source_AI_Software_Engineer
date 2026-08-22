"""Naming a session from what the user actually asked, using the model
instead of a truncated first line.

A truncated first line ("fix the thing that is broken in the module and also
handle the empty case and make sure existing t...") is a bad session name —
it's a byte offset, not a summary. Asking the model for a few words is one
extra call, so it happens as a fire-and-forget background task rather than
adding latency to the message the user is waiting on.
"""

import logging

from app.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

_MAX_TITLE_CHARS = 60
_SYSTEM_PROMPT = (
    "Write a short title for a coding assistant conversation, summarizing what the user is asking for. "
    "3 to 6 words. Plain text only — no quotes, no punctuation at the end, no prefixes like 'Title:'. "
    "Respond with ONLY the title, nothing else."
)


async def generate_session_title(llm: LLMProvider, first_message: str) -> str | None:
    """Returns a short title, or None if the model's response wasn't usable
    (caller falls back to a truncated first line — always has one)."""
    try:
        response = await llm.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(role="user", content=first_message),
            ],
            temperature=0.3,
            max_tokens=20,
        )
    except Exception:  # a naming failure must never break the chat
        logger.warning("session title generation failed", exc_info=True)
        return None

    title = _clean(response.content)
    return title or None


def _clean(raw: str) -> str:
    # Small models reliably wrap the answer in quotes or add a trailing
    # period despite the instruction not to — stripped rather than retried,
    # since a retry costs another full LLM round trip for a cosmetic fix.
    title = raw.strip().strip("\"'“”‘’").strip()
    title = title.splitlines()[0].strip() if title else ""
    title = title.rstrip(".!")
    if not title:
        return ""
    if len(title) > _MAX_TITLE_CHARS:
        title = title[: _MAX_TITLE_CHARS - 1].rstrip() + "…"
    return title
