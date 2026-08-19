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
