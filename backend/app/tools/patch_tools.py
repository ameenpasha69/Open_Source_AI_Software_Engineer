import difflib

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.retrieval.file_walker import IGNORED_DIR_NAMES
from app.tools.base import Tool, ToolError
from app.tools.path_utils import resolve_safe_path
from app.tools.repo_utils import get_repository_root

# Never patch env/secret files even though they're inside the repo root —
# resolve_safe_path only guarantees "inside the repo," not "safe to overwrite."
_DENIED_FILENAMES = frozenset({".env", ".env.local", ".env.development", ".env.production", ".env.test"})


class ApplyPatchInput(BaseModel):
    repository_id: str
    path: str
    old_content: str
    new_content: str


class ApplyPatchOutput(BaseModel):
    path: str
    diff: str
    lines_added: int
    lines_removed: int


class ApplyPatchTool(Tool):
    """Search-and-replace patching, not raw unified-diff application.

    The caller supplies the exact text it expects to find (`old_content` —
    the "surrounding context" spec section 10 asks for) and its replacement.
    This is deliberately not a unified-diff-with-line-numbers tool: small
    local models are unreliable at producing correct line numbers and hunk
    headers, but are reasonably good at reproducing a short, exact excerpt
    of code they just read via `read_file`. `old_content` must appear in the
    file exactly once — zero occurrences means the expected context wasn't
    found (the model may be misremembering the file, or it changed since
    it was last read); more than one means the patch is ambiguous about
    *which* occurrence to change, and applying the wrong one would be worse
    than refusing.
    """

    name = "apply_patch"
    description = (
        "Replace an exact, unique excerpt of a file's content with new content. "
        "old_content must match the file exactly once."
    )
    input_schema = ApplyPatchInput
    output_schema = ApplyPatchOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: ApplyPatchInput) -> ApplyPatchOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        target = resolve_safe_path(repo_root, input_data.path)

        if target.name in _DENIED_FILENAMES:
            raise ToolError(f"Refusing to patch '{input_data.path}': environment/secret files are protected")
        if any(part in IGNORED_DIR_NAMES for part in target.relative_to(repo_root).parts[:-1]):
            raise ToolError(f"Refusing to patch '{input_data.path}': inside an ignored directory")
        if not target.is_file():
            raise ToolError(f"'{input_data.path}' is not a file")

        if not input_data.old_content:
            raise ToolError("old_content must not be empty")

        try:
            original = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"'{input_data.path}' is not a UTF-8 text file") from exc

        occurrences = original.count(input_data.old_content)
        if occurrences == 0:
            message = (
                f"old_content not found in '{input_data.path}' — it must match the file's "
                "current content exactly. Re-read the file to confirm its current content."
            )
            hint = _find_closest_match(original, input_data.old_content)
            if hint is not None:
                message += (
                    f"\n\nClosest match actually in the file:\n{hint}\n\n"
                    "Compare this character-by-character against old_content — a single wrong "
                    "character (mismatched punctuation, whitespace) is enough to cause this error."
                )
            raise ToolError(message)
        if occurrences > 1:
            raise ToolError(
                f"old_content matches {occurrences} locations in '{input_data.path}' — "
                "it must be unique. Include more surrounding context to disambiguate."
            )

        updated = original.replace(input_data.old_content, input_data.new_content, 1)
        target.write_text(updated, encoding="utf-8")

        diff = _unified_diff(input_data.path, original, updated)
        added, removed = _count_changes(diff)
        return ApplyPatchOutput(path=input_data.path, diff=diff, lines_added=added, lines_removed=removed)


# Below this, a "not found" error gets no closest-match hint at all — the
# content isn't a plausible near-miss of anything in the file, and a
# low-similarity suggestion would be noise, not a correction signal.
_CLOSE_MATCH_THRESHOLD = 0.6


def _find_closest_match(original: str, old_content: str) -> str | None:
    """Best-effort "did you mean" hint for a failed exact match: slides a
    window the same number of lines as old_content over the file and
    returns the closest-scoring window's text, if any window is a
    plausible near-miss rather than unrelated content.

    Targets a specific, observed failure mode: a model correctly locates
    the right bug but transcribes old_content with one character wrong (a
    swapped punctuation mark, mismatched whitespace) — the bare "not found"
    error gives it nothing to correct from, so it can end up repeating the
    same wrong transcription on retry instead of noticing the mismatch.
    """
    old_lines = old_content.splitlines()
    file_lines = original.splitlines()
    window_size = len(old_lines)
    if window_size == 0 or window_size > len(file_lines):
        return None

    # Scored on stripped lines: a model that drops or miscounts leading
    # indentation (common — read_file output isn't guaranteed to be copied
    # whitespace-for-whitespace) shouldn't score as "unrelated" just because
    # of that. The hint shown back still uses the real, unstripped line —
    # this only affects which window is picked as closest.
    old_normalized = "\n".join(line.strip() for line in old_lines)

    best_ratio = 0.0
    best_window: str | None = None
    for start in range(len(file_lines) - window_size + 1):
        window_lines = file_lines[start : start + window_size]
        window_normalized = "\n".join(line.strip() for line in window_lines)
        ratio = difflib.SequenceMatcher(None, old_normalized, window_normalized).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_window = "\n".join(window_lines)

    return best_window if best_ratio >= _CLOSE_MATCH_THRESHOLD else None


def _unified_diff(path: str, old_text: str, new_text: str) -> str:
    diff_lines = difflib.unified_diff(
        old_text.splitlines(keepends=True),
        new_text.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff_lines)


def _count_changes(diff_text: str) -> tuple[int, int]:
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed
