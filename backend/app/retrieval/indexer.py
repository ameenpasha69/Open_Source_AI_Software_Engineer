import datetime
import hashlib
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import CodeChunk, IndexedFile, IndexRun, Repository
from app.retrieval.chunker import get_chunker
from app.retrieval.file_walker import walk_repository
from app.schemas.indexing import IndexRunResult


class RepositoryNotFoundError(Exception):
    pass


class RepositoryIndexer:
    """Walks a repository, (re-)chunks changed files, and persists the result.

    Re-indexing is incremental: a file's content hash is compared against the
    last indexed hash, and unchanged files are skipped entirely (no re-parse,
    no re-chunk) — this is also what lets Milestone 3 skip re-embedding
    unchanged chunks. Files that disappeared from disk since the last run
    have their chunks removed so the index stays accurate.
    """

    def __init__(self, session: Session, chunk_max_lines: int, chunk_overlap_lines: int, max_file_size_bytes: int):
        self._session = session
        self._chunk_max_lines = chunk_max_lines
        self._chunk_overlap_lines = chunk_overlap_lines
        self._max_file_size_bytes = max_file_size_bytes

    def index(self, repo_path: Path, name: str | None = None) -> IndexRunResult:
        repo_path = repo_path.resolve()
        if not repo_path.is_dir():
            raise RepositoryNotFoundError(f"'{repo_path}' is not a directory")

        repository = self._get_or_create_repository(repo_path, name)
        run = IndexRun(repository_id=repository.id, status="running")
        self._session.add(run)
        self._session.flush()

        started = time.monotonic()
        try:
            counts = self._run_index(repository)
            run.status = "completed"
        except Exception as exc:
            self._session.rollback()
            run.status = "failed"
            run.error = str(exc)
            self._session.add(run)
            run.finished_at = datetime.datetime.now(datetime.UTC)
            self._session.commit()
            raise

        for field, value in counts.items():
            setattr(run, field, value)
        run.finished_at = datetime.datetime.now(datetime.UTC)
        repository.last_indexed_at = run.finished_at
        self._session.commit()

        return IndexRunResult(
            index_run_id=run.id,
            repository_id=repository.id,
            status=run.status,
            files_scanned=run.files_scanned,
            files_indexed=run.files_indexed,
            files_skipped_unchanged=run.files_skipped_unchanged,
            files_ignored=run.files_ignored,
            files_deleted=run.files_deleted,
            chunks_created=run.chunks_created,
            duration_seconds=round(time.monotonic() - started, 3),
            error=run.error,
        )

    def _run_index(self, repository: Repository) -> dict[str, int]:
        counts = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_skipped_unchanged": 0,
            "files_ignored": 0,
            "files_deleted": 0,
            "chunks_created": 0,
        }
        existing_files = {
            f.relative_path: f
            for f in self._session.scalars(
                select(IndexedFile).where(IndexedFile.repository_id == repository.id)
            )
        }
        seen_paths: set[str] = set()

        for candidate in walk_repository(Path(repository.path), self._max_file_size_bytes):
            counts["files_scanned"] += 1
            try:
                text = candidate.absolute_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                counts["files_ignored"] += 1
                continue

            seen_paths.add(candidate.relative_path)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            existing = existing_files.get(candidate.relative_path)

            if existing is not None and existing.content_hash == content_hash:
                counts["files_skipped_unchanged"] += 1
                continue

            raw_chunks = get_chunker(
                candidate.language, self._chunk_max_lines, self._chunk_overlap_lines
            ).chunk(text)

            if existing is not None:
                self._session.query(CodeChunk).filter(CodeChunk.file_id == existing.id).delete()
                existing.content_hash = content_hash
                existing.chunk_count = len(raw_chunks)
                indexed_file = existing
            else:
                indexed_file = IndexedFile(
                    repository_id=repository.id,
                    relative_path=candidate.relative_path,
                    language=candidate.language,
                    content_hash=content_hash,
                    chunk_count=len(raw_chunks),
                )
                self._session.add(indexed_file)
                self._session.flush()

            for raw in raw_chunks:
                self._session.add(
                    CodeChunk(
                        file_id=indexed_file.id,
                        repository_id=repository.id,
                        relative_path=candidate.relative_path,
                        language=candidate.language,
                        symbol=raw.symbol,
                        start_line=raw.start_line,
                        end_line=raw.end_line,
                        content=raw.content,
                    )
                )
            counts["files_indexed"] += 1
            counts["chunks_created"] += len(raw_chunks)

        for relative_path, stale_file in existing_files.items():
            if relative_path not in seen_paths:
                self._session.delete(stale_file)
                counts["files_deleted"] += 1

        return counts

    def _get_or_create_repository(self, repo_path: Path, name: str | None) -> Repository:
        repository = self._session.scalar(select(Repository).where(Repository.path == str(repo_path)))
        if repository is None:
            repository = Repository(name=name or repo_path.name, path=str(repo_path))
            self._session.add(repository)
            self._session.flush()
        return repository
