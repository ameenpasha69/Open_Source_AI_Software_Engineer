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
            if original == "":
                # A dead end otherwise: old_content can never match an empty
                # file, and this tool cannot fill one in — observed live, a
                # model created an empty placeholder via create_file, then
                # spent the rest of its iteration budget trying to apply_patch
                # content into it that could never match. Naming the actual
                # recovery (start over with delete_file + create_file, or
                # write the real content the first time) is the only way out.
                raise ToolError(
                    f"'{input_data.path}' is currently empty (0 bytes) — old_content can never match "
                    "anything in an empty file, so apply_patch cannot be used to add content to it. "
                    "Either delete_file this path and create_file it again with the full content in "
                    "one call, or — better — call create_file with the complete content directly "
                    "instead of creating an empty file first."
                )
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


class CreateFileInput(BaseModel):
    repository_id: str
    path: str
    content: str = ""


class CreateFileTool(Tool):
    """Creates a new file — apply_patch's counterpart for when there's
    nothing to replace yet.

    Observed live: a model needing to add a test file had no valid way to do
    it. `apply_patch` refuses on a nonexistent target (there's no old_content
    to match), and the only other route, `run_command`, doesn't allow `touch`
    — a reasonable restriction on its own, but one that left file creation
    with no path at all, so the model looped on the exact same disallowed
    command. This tool is a small, explicit, safe alternative: it plugs that
    gap without widening the shell allowlist.
    """

    name = "create_file"
    description = "Create a new file with the given content. Fails if the file already exists — use apply_patch to modify one."
    input_schema = CreateFileInput
    output_schema = ApplyPatchOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: CreateFileInput) -> ApplyPatchOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        target = resolve_safe_path(repo_root, input_data.path)

        if target.name in _DENIED_FILENAMES:
            raise ToolError(f"Refusing to create '{input_data.path}': environment/secret files are protected")
        if any(part in IGNORED_DIR_NAMES for part in target.relative_to(repo_root).parts[:-1]):
            raise ToolError(f"Refusing to create '{input_data.path}': inside an ignored directory")
        if target.exists():
            raise ToolError(f"'{input_data.path}' already exists — use apply_patch to modify it")

        # The missing directory is often exactly why the model reached for
        # this tool in the first place (e.g. a repository with no tests/
        # directory yet) — creating it here is what makes that case work in
        # one call instead of needing a separate, unavailable mkdir step.
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(input_data.content, encoding="utf-8")

        diff = _unified_diff(input_data.path, "", input_data.content)
        added, removed = _count_changes(diff)
        return ApplyPatchOutput(path=input_data.path, diff=diff, lines_added=added, lines_removed=removed)


class DeleteFileInput(BaseModel):
    repository_id: str
    path: str


class DeleteFileTool(Tool):
    """Deletes a file — the one operation still missing after apply_patch
    (modify) and create_file (add). Together they're what "rename" actually
    is: create_file at the new path, then delete_file the old one. There is
    no separate rename/move tool because a rename in this project's tasks is
    rarely a pure move — it's usually paired with a content change (as it
    was in the task that exposed this gap), so composing two explicit,
    already-understood primitives is more predictable than one tool trying
    to do both at once.
    """

    name = "delete_file"
    description = "Delete a file. To rename one, create_file the new path and then delete_file the old path."
    input_schema = DeleteFileInput
    output_schema = ApplyPatchOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: DeleteFileInput) -> ApplyPatchOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        target = resolve_safe_path(repo_root, input_data.path)

        if target.name in _DENIED_FILENAMES:
            raise ToolError(f"Refusing to delete '{input_data.path}': environment/secret files are protected")
        if any(part in IGNORED_DIR_NAMES for part in target.relative_to(repo_root).parts[:-1]):
            raise ToolError(f"Refusing to delete '{input_data.path}': inside an ignored directory")
        if not target.exists():
            raise ToolError(f"'{input_data.path}' does not exist — nothing to delete")
        if not target.is_file():
            raise ToolError(f"'{input_data.path}' is a directory, not a file — delete_file only removes files")

        original = target.read_text(encoding="utf-8", errors="replace")
        target.unlink()

        diff = _unified_diff(input_data.path, original, "")
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
