
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.retrieval.file_walker import IGNORED_DIR_NAMES
from app.tools.base import Tool, ToolError
from app.tools.path_utils import resolve_safe_path
from app.tools.repo_utils import get_repository_root


class FileEntry(BaseModel):
    path: str
    is_dir: bool
    size_bytes: int | None = None


class ListFilesInput(BaseModel):
    repository_id: str
    path: str = ""
    recursive: bool = False


class ListFilesOutput(BaseModel):
    entries: list[FileEntry]


class ListFilesTool(Tool):
    name = "list_files"
    description = "List files and directories at a path within the repository (non-recursive by default)."
    input_schema = ListFilesInput
    output_schema = ListFilesOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: ListFilesInput) -> ListFilesOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        target = resolve_safe_path(repo_root, input_data.path)
        if not target.is_dir():
            raise ToolError(
                f"'{input_data.path}' is not a directory (it may not exist). "
                "Use list_files(path=\"\") to see the repository root, or search_code / find_symbol "
                "to locate something by name instead of guessing a path."
            )

        iterator = target.rglob("*") if input_data.recursive else target.iterdir()
        entries = []
        for item in sorted(iterator):
            if any(part in IGNORED_DIR_NAMES for part in item.relative_to(repo_root).parts):
                continue
            entries.append(
                FileEntry(
                    path=str(item.relative_to(repo_root)),
                    is_dir=item.is_dir(),
                    size_bytes=None if item.is_dir() else item.stat().st_size,
                )
            )
        return ListFilesOutput(entries=entries)


class ReadFileInput(BaseModel):
    repository_id: str
    path: str
    start_line: int | None = None
    end_line: int | None = None


class ReadFileOutput(BaseModel):
    path: str
    content: str
    start_line: int
    end_line: int
    total_lines: int


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read a file's content, optionally restricted to a line range."
    input_schema = ReadFileInput
    output_schema = ReadFileOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session, max_file_size_bytes: int):
        self._session = session
        self._max_file_size_bytes = max_file_size_bytes

    async def run(self, input_data: ReadFileInput) -> ReadFileOutput:
        repo_root = get_repository_root(self._session, input_data.repository_id)
        target = resolve_safe_path(repo_root, input_data.path)
        if not target.is_file():
            raise ToolError(
                f"'{input_data.path}' is not a file (it may not exist, or the path is wrong). "
                "Use list_files(path=\"\") to see the repository root, or search_code / find_symbol "
                "to locate the right file by name instead of guessing a path."
            )
        if target.stat().st_size > self._max_file_size_bytes:
            raise ToolError(f"'{input_data.path}' exceeds the {self._max_file_size_bytes}-byte read limit")

        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"'{input_data.path}' is not a UTF-8 text file") from exc

        lines = text.splitlines()
        start = max(input_data.start_line or 1, 1)
        end = min(input_data.end_line or len(lines), len(lines))
        if start > end:
            raise ToolError(f"start_line ({start}) is after end_line ({end})")

        return ReadFileOutput(
            path=input_data.path,
            content="\n".join(lines[start - 1 : end]),
            start_line=start,
            end_line=end,
            total_lines=len(lines),
        )


class GetFileContextInput(BaseModel):
    repository_id: str
    path: str
    line: int
    context_lines: int = 20


class GetFileContextTool(Tool):
    name = "get_file_context"
    description = "Read a window of lines centered on a given line number — useful after a search hit."
    input_schema = GetFileContextInput
    output_schema = ReadFileOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session, max_file_size_bytes: int):
        self._read_file = ReadFileTool(session, max_file_size_bytes)

    async def run(self, input_data: GetFileContextInput) -> ReadFileOutput:
        if input_data.line < 1:
            raise ToolError("line must be >= 1")
        start = max(input_data.line - input_data.context_lines, 1)
        end = input_data.line + input_data.context_lines
        return await self._read_file.run(
            ReadFileInput(
                repository_id=input_data.repository_id,
                path=input_data.path,
                start_line=start,
                end_line=end,
            )
        )
