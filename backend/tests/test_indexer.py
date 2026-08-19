
import pytest
from app.database.models import CodeChunk, IndexedFile, Repository
from app.database.session import create_sqlite_engine, get_session_factory
from app.retrieval.indexer import RepositoryIndexer, RepositoryNotFoundError
from sqlalchemy import select


@pytest.fixture
def db_session(tmp_path):
    engine = create_sqlite_engine(f"sqlite:///{tmp_path / 'test.db'}")
    session_factory = get_session_factory(engine)
    session = session_factory()
    yield session
    session.close()


@pytest.fixture
def sample_repo(tmp_path):
    repo = tmp_path / "sample_repo"
    (repo / "app").mkdir(parents=True)
    (repo / "app" / "service.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n"
    )
    (repo / "app" / "utils.js").write_text("function noop() {}\n")
    return repo


def _make_indexer(session):
    return RepositoryIndexer(
        session=session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000
    )


def test_index_creates_repository_and_chunks(db_session, sample_repo):
    result = _make_indexer(db_session).index(sample_repo)

    assert result.status == "completed"
    assert result.files_scanned == 2
    assert result.files_indexed == 2
    assert result.files_skipped_unchanged == 0
    assert result.chunks_created > 0

    repository = db_session.scalar(select(Repository))
    assert repository.name == "sample_repo"
    assert repository.last_indexed_at is not None

    chunks = db_session.scalars(select(CodeChunk)).all()
    assert any(c.symbol == "calculate_total" for c in chunks)


def test_reindex_unchanged_repository_skips_all_files(db_session, sample_repo):
    _make_indexer(db_session).index(sample_repo)
    second_run = _make_indexer(db_session).index(sample_repo)

    assert second_run.files_indexed == 0
    assert second_run.files_skipped_unchanged == 2
    assert second_run.chunks_created == 0

    # Same repository row reused, not duplicated.
    assert db_session.scalar(select(Repository).where(Repository.path == str(sample_repo.resolve()))) is not None
    repo_count = len(db_session.scalars(select(Repository)).all())
    assert repo_count == 1


def test_reindex_after_file_change_rechunks_only_that_file(db_session, sample_repo):
    _make_indexer(db_session).index(sample_repo)

    (sample_repo / "app" / "service.py").write_text(
        "def calculate_total(items):\n    return sum(items) + 1\n\n\ndef new_func():\n    pass\n"
    )
    result = _make_indexer(db_session).index(sample_repo)

    assert result.files_indexed == 1
    assert result.files_skipped_unchanged == 1

    symbols = {c.symbol for c in db_session.scalars(select(CodeChunk)).all()}
    assert "new_func" in symbols


def test_reindex_after_file_deletion_removes_its_chunks(db_session, sample_repo):
    _make_indexer(db_session).index(sample_repo)
    (sample_repo / "app" / "utils.js").unlink()

    result = _make_indexer(db_session).index(sample_repo)

    assert result.files_deleted == 1
    remaining_paths = {f.relative_path for f in db_session.scalars(select(IndexedFile)).all()}
    assert "app/utils.js" not in remaining_paths
    assert not any(c.relative_path == "app/utils.js" for c in db_session.scalars(select(CodeChunk)).all())


def test_index_nonexistent_path_raises(db_session, tmp_path):
    with pytest.raises(RepositoryNotFoundError):
        _make_indexer(db_session).index(tmp_path / "does_not_exist")


def test_index_repository_twice_by_path_reuses_repository_row(db_session, sample_repo):
    first = _make_indexer(db_session).index(sample_repo, name="first-name")
    second = _make_indexer(db_session).index(sample_repo, name="second-name")

    assert first.repository_id == second.repository_id
