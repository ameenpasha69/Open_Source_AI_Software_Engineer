import datetime

from pydantic import BaseModel


class CodeChunkOut(BaseModel):
    """A single indexed unit of source code, with enough metadata to locate
    and cite it (this is what retrieval will eventually return to the agent)."""

    chunk_id: str
    repository: str
    file_path: str
    language: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str


class IndexRepositoryRequest(BaseModel):
    path: str
    name: str | None = None
    # Explicit overrides for run_tests/run_linter/run_formatter. None means
    # "don't change" on re-index, not "clear" — a Python-majority repository
    # gets a sensible default automatically if never set at all.
    test_command: list[str] | None = None
    lint_command: list[str] | None = None
    format_command: list[str] | None = None


class EmbeddingSyncResult(BaseModel):
    chunks_embedded: int
    chunks_removed: int
    duration_seconds: float
    error: str | None = None


class IndexRunResult(BaseModel):
    index_run_id: str
    repository_id: str
    status: str
    files_scanned: int
    files_indexed: int
    files_skipped_unchanged: int
    files_ignored: int
    files_deleted: int
    chunks_created: int
    duration_seconds: float
    error: str | None = None
    embedding: EmbeddingSyncResult | None = None


class RepositorySummary(BaseModel):
    id: str
    name: str
    path: str
    created_at: datetime.datetime
    last_indexed_at: datetime.datetime | None
    indexed_file_count: int
    chunk_count: int
    embedded_chunk_count: int
    test_command: list[str] | None
    lint_command: list[str] | None
    format_command: list[str] | None
