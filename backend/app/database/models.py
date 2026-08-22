import datetime
import uuid

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


class Base(DeclarativeBase):
    pass


class Repository(Base):
    __tablename__ = "repositories"
    __table_args__ = (UniqueConstraint("path", name="uq_repositories_path"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    last_indexed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    # JSON-encoded argv lists (e.g. ["python3", "-m", "pytest"]). Set explicitly
    # via IndexRepositoryRequest, or auto-defaulted post-indexing for a
    # Python-majority repository — see retrieval/command_detection.py. None
    # means "not configured"; run_tests/run_linter/run_formatter raise a
    # clear ToolError rather than guessing.
    test_command_json: Mapped[str | None] = mapped_column(Text)
    lint_command_json: Mapped[str | None] = mapped_column(Text)
    format_command_json: Mapped[str | None] = mapped_column(Text)

    index_runs: Mapped[list["IndexRun"]] = relationship(back_populates="repository", cascade="all, delete-orphan")
    indexed_files: Mapped[list["IndexedFile"]] = relationship(
        back_populates="repository", cascade="all, delete-orphan"
    )


class IndexRun(Base):
    __tablename__ = "index_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    status: Mapped[str] = mapped_column(String(32), default="running")  # running | completed | failed
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    files_scanned: Mapped[int] = mapped_column(Integer, default=0)
    files_indexed: Mapped[int] = mapped_column(Integer, default=0)
    files_skipped_unchanged: Mapped[int] = mapped_column(Integer, default=0)
    files_ignored: Mapped[int] = mapped_column(Integer, default=0)
    files_deleted: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)

    repository: Mapped[Repository] = relationship(back_populates="index_runs")


class IndexedFile(Base):
    __tablename__ = "indexed_files"
    __table_args__ = (UniqueConstraint("repository_id", "relative_path", name="uq_indexed_file_path"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    relative_path: Mapped[str] = mapped_column(String(1024))
    language: Mapped[str] = mapped_column(String(64))
    content_hash: Mapped[str] = mapped_column(String(64))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    last_indexed_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    repository: Mapped[Repository] = relationship(back_populates="indexed_files")
    chunks: Mapped[list["CodeChunk"]] = relationship(back_populates="file", cascade="all, delete-orphan")


class CodeChunk(Base):
    __tablename__ = "code_chunks"
    __table_args__ = (Index("ix_code_chunks_file_id", "file_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    file_id: Mapped[str] = mapped_column(ForeignKey("indexed_files.id"))
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    relative_path: Mapped[str] = mapped_column(String(1024))
    language: Mapped[str] = mapped_column(String(64))
    symbol: Mapped[str | None] = mapped_column(String(512))
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)

    file: Mapped[IndexedFile] = relationship(back_populates="chunks")


class VectorRecord(Base):
    """Maps a CodeChunk to its integer id inside a repository's FAISS index
    (FAISS requires int64 ids; chunk ids are uuid4 hex strings). Existence of
    a row here is also how the embedding pipeline knows a chunk has already
    been embedded, so unchanged chunks are never re-embedded.
    """

    __tablename__ = "vector_records"
    __table_args__ = (Index("ix_vector_records_repository_id", "repository_id"),)

    faiss_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("code_chunks.id"), unique=True
    )
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    # Nullable: a run can still be started standalone (the eval harness does
    # exactly that). A run inside a session also sees that session's earlier
    # turns as context.
    session_id: Mapped[str | None] = mapped_column(ForeignKey("chat_sessions.id"))
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="running")
    plan_json: Mapped[str] = mapped_column(Text, default="[]")
    final_answer: Mapped[str | None] = mapped_column(Text)
    root_cause: Mapped[str | None] = mapped_column(Text)
    iteration_count: Mapped[int] = mapped_column(Integer, default=0)
    # "not_applicable" (no files modified) or "unverified" (modified, but no
    # test run exists yet to confirm the fix — Milestone 8 adds VERIFIED /
    # PARTIALLY_VERIFIED / FAILED once run_tests exists).
    verification_status: Mapped[str] = mapped_column(String(32), default="not_applicable")
    started_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    # As reported by the inference backend, summed over every LLM call the run
    # made (the plan plus one per iteration, plus any JSON-repair retries).
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_call_count: Mapped[int] = mapped_column(Integer, default=0)
    # The largest single prompt this run sent — what a context-window gauge
    # actually means, unlike the running total.
    peak_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(255))

    events: Mapped[list["AgentEvent"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    tool_calls: Mapped[list["AgentToolCall"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    modified_files: Mapped[list["ModifiedFile"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    test_runs: Mapped[list["TestRun"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class AgentEvent(Base):
    """One entry in the agent run's timeline (plan_created, tool_called,
    tool_completed, finished, error, ...) — the audit trail behind
    GET /api/agent/{run_id}/events, and later the source of streamed events
    once Milestone 9 adds live streaming.
    """

    __tablename__ = "agent_events"
    __table_args__ = (Index("ix_agent_events_run_id", "run_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    iteration: Mapped[int] = mapped_column(Integer)
    event_type: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="events")


class AgentToolCall(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_run_id", "run_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    iteration: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(128))
    input_json: Mapped[str] = mapped_column(Text, default="{}")
    success: Mapped[bool] = mapped_column(default=False)
    output_json: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")


class ModifiedFile(Base):
    """One successful apply_patch call — the per-file diff, kept independent
    of whether the target repository is even git-tracked, so
    GET /api/agent/{run_id}/diff doesn't depend on git.
    """

    __tablename__ = "modified_files"
    __table_args__ = (Index("ix_modified_files_run_id", "run_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"))
    relative_path: Mapped[str] = mapped_column(String(1024))
    diff: Mapped[str] = mapped_column(Text)
    lines_added: Mapped[int] = mapped_column(Integer, default=0)
    lines_removed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[AgentRun] = relationship(back_populates="modified_files")


class TestRun(Base):
    """One run_tests execution — repository_id-scoped so it works whether or
    not it happened inside an agent run (run_id nullable, for a standalone
    POST /api/tools/execute call)."""

    __test__ = False  # not a pytest test class — this name just mirrors the domain
    __tablename__ = "test_runs"
    __table_args__ = (Index("ix_test_runs_run_id", "run_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"))
    command: Mapped[str] = mapped_column(String(1024))
    scope: Mapped[str] = mapped_column(String(16))  # "full" | "targeted"
    exit_code: Mapped[int] = mapped_column(Integer)
    passed: Mapped[bool] = mapped_column(default=False)
    total_tests: Mapped[int | None] = mapped_column(Integer)
    passed_tests: Mapped[int | None] = mapped_column(Integer)
    failed_tests_json: Mapped[str] = mapped_column(Text, default="[]")
    failure_category: Mapped[str] = mapped_column(String(32))
    duration_seconds: Mapped[float] = mapped_column(default=0.0)
    stdout: Mapped[str] = mapped_column(Text, default="")
    stderr: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    run: Mapped[AgentRun | None] = relationship(back_populates="test_runs")


class AppSetting(Base):
    """A single mutable application preference, keyed by name.

    Settings (`app/config/settings.py`) stays the source of the *defaults* —
    it's read-only, per-machine, and loaded from .env. A row here is a
    runtime override the user chose in the UI (currently: which pulled model
    the agent and the indexer use), which has to outlive a server restart.
    """

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ChatSession(Base):
    """A named conversation against one repository.

    Sessions are what make the agent conversational rather than one-shot: a
    run started inside a session is given the session's earlier turns as
    context, so "now also handle the empty case" resolves against what was
    just discussed instead of starting from nothing. The repository is fixed
    at creation — it's what scopes retrieval, so letting it change mid-session
    would silently invalidate every earlier turn's evidence.
    """

    __tablename__ = "chat_sessions"
    __table_args__ = (Index("ix_chat_sessions_repository_id", "repository_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    repository_id: Mapped[str] = mapped_column(ForeignKey("repositories.id"))
    title: Mapped[str] = mapped_column(String(255), default="New session")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    repository: Mapped[Repository] = relationship()
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at"
    )


class ChatMessage(Base):
    """One turn in a session. A user turn is what someone typed; an assistant
    turn is the answer an agent run produced, linked back to that run so the
    full timeline, diff, and test output stay reachable from the transcript.
    """

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_messages_session_id", "session_id"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id"))
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, default="")
    # Assistant turns only — the run that produced this answer.
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id"))
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[ChatSession] = relationship(back_populates="messages")
