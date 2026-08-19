import pytest
from app.database.models import VectorRecord
from app.database.session import create_sqlite_engine, get_session_factory
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.indexer import RepositoryIndexer
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
    repo.mkdir()
    (repo / "orders.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n\n\n"
        "def apply_discount(total, pct):\n    return total * (1 - pct)\n"
    )
    return repo


@pytest.fixture
def vector_index_dir(tmp_path):
    d = tmp_path / "vector_indexes"
    d.mkdir()
    return d


def _index(session, repo_path):
    indexer = RepositoryIndexer(
        session=session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000
    )
    return indexer.index(repo_path)


async def test_sync_embeds_all_new_chunks(db_session, sample_repo, vector_index_dir, fake_embedding_provider):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)

    sync_result = await pipeline.sync(result.repository_id)

    assert sync_result.error is None
    assert sync_result.chunks_embedded == result.chunks_created
    assert sync_result.chunks_removed == 0

    records = db_session.scalars(
        select(VectorRecord).where(VectorRecord.repository_id == result.repository_id)
    ).all()
    assert len(records) == result.chunks_created
    assert (vector_index_dir / f"{result.repository_id}.faiss").exists()


async def test_sync_is_idempotent_for_unchanged_repository(
    db_session, sample_repo, vector_index_dir, fake_embedding_provider
):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    await pipeline.sync(result.repository_id)

    fake_embedding_provider.embedded_texts.clear()
    second_sync = await pipeline.sync(result.repository_id)

    assert second_sync.chunks_embedded == 0
    assert second_sync.chunks_removed == 0
    assert fake_embedding_provider.embedded_texts == []


async def test_sync_embeds_only_new_chunks_after_file_change(
    db_session, sample_repo, vector_index_dir, fake_embedding_provider
):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    await pipeline.sync(result.repository_id)

    (sample_repo / "orders.py").write_text(
        "def calculate_total(items):\n    return sum(items)\n\n\n"
        "def apply_discount(total, pct):\n    return total * (1 - pct)\n\n\n"
        "def new_function():\n    pass\n"
    )
    second_index = _index(db_session, sample_repo)
    second_sync = await pipeline.sync(result.repository_id)

    assert second_index.files_indexed == 1
    # calculate_total and apply_discount were removed+recreated with new ids
    # (whole file re-chunked), new_function is genuinely new.
    assert second_sync.chunks_embedded == second_index.chunks_created
    assert second_sync.chunks_removed > 0  # stale vectors from the old chunk ids


async def test_sync_removes_vectors_for_deleted_files(
    db_session, sample_repo, vector_index_dir, fake_embedding_provider
):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    await pipeline.sync(result.repository_id)

    (sample_repo / "orders.py").unlink()
    _index(db_session, sample_repo)
    second_sync = await pipeline.sync(result.repository_id)

    assert second_sync.chunks_removed == result.chunks_created
    assert second_sync.chunks_embedded == 0

    remaining = db_session.scalars(
        select(VectorRecord).where(VectorRecord.repository_id == result.repository_id)
    ).all()
    assert remaining == []


async def test_sync_with_no_changes_does_not_touch_embedding_provider(
    db_session, sample_repo, vector_index_dir, fake_embedding_provider
):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    await pipeline.sync(result.repository_id)
    call_count_before = len(fake_embedding_provider.embedded_texts)

    await pipeline.sync(result.repository_id)

    assert len(fake_embedding_provider.embedded_texts) == call_count_before


async def test_search_returns_relevant_chunk_first(
    db_session, sample_repo, vector_index_dir, fake_embedding_provider
):
    result = _index(db_session, sample_repo)
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    await pipeline.sync(result.repository_id)

    results = await pipeline.search(result.repository_id, "def calculate_total(items):", top_k=5)

    assert results
    assert results[0].symbol == "calculate_total"
    assert all(r.file_path == "orders.py" for r in results)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


async def test_search_on_unindexed_repository_returns_empty(db_session, vector_index_dir, fake_embedding_provider):
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, vector_index_dir)
    assert await pipeline.search("no-such-repo", "anything", top_k=5) == []
