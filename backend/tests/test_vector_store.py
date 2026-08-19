import pytest
from app.retrieval.vector_store import FaissVectorStore


def test_creating_store_without_dimension_raises_when_no_index_exists(tmp_path):
    with pytest.raises(ValueError, match="dimension"):
        FaissVectorStore(tmp_path / "missing.faiss")


async def test_add_and_search_returns_closest_vector_first(tmp_path):
    store = FaissVectorStore(tmp_path / "index.faiss", dimension=4)

    await store.add(
        ids=[1, 2, 3],
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
        ],
    )

    results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=2)

    assert [r[0] for r in results] == [1, 3]  # id 1 exact match, id 3 next closest
    assert results[0][1] > results[1][1]  # closer vector scores higher


async def test_search_on_empty_store_returns_empty_list(tmp_path):
    store = FaissVectorStore(tmp_path / "index.faiss", dimension=4)
    assert await store.search([1.0, 0.0, 0.0, 0.0], top_k=5) == []


async def test_delete_removes_vector_from_results(tmp_path):
    store = FaissVectorStore(tmp_path / "index.faiss", dimension=4)
    await store.add(ids=[1, 2], vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    await store.delete([1])
    results = await store.search([1.0, 0.0, 0.0, 0.0], top_k=5)

    assert [r[0] for r in results] == [2]
    assert store.size == 1


async def test_save_and_reload_preserves_vectors(tmp_path):
    index_path = tmp_path / "index.faiss"
    store = FaissVectorStore(index_path, dimension=4)
    await store.add(ids=[1, 2], vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    await store.save()

    reloaded = FaissVectorStore(index_path)

    assert reloaded.dimension == 4
    assert reloaded.size == 2
    results = await reloaded.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == 1


async def test_zero_vector_does_not_raise_on_normalize(tmp_path):
    store = FaissVectorStore(tmp_path / "index.faiss", dimension=4)
    await store.add(ids=[1], vectors=[[0.0, 0.0, 0.0, 0.0]])
    assert store.size == 1
