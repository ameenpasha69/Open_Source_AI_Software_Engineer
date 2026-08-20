import pytest
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.indexer import RepositoryIndexer
from app.tools.base import ToolError
from app.tools.code_search_tools import (
    FindReferencesInput,
    FindReferencesTool,
    FindSymbolInput,
    FindSymbolTool,
    SearchCodeInput,
    SearchCodeTool,
)


@pytest.fixture
def orders_repo(tmp_path):
    repo = tmp_path / "orders_repo"
    repo.mkdir()
    (repo / "service.py").write_text(
        "def calculate_total(items):\n    return sum(item.price for item in items)\n\n\n"
        "def apply_discount(total, pct):\n    return total * (1 - pct)\n\n\n"
        "def checkout(items):\n    total = calculate_total(items)\n    return total\n"
    )
    return repo


@pytest.fixture
def indexed_repository_id(db_session, orders_repo, fake_embedding_provider, tmp_path):
    indexer = RepositoryIndexer(
        session=db_session, chunk_max_lines=200, chunk_overlap_lines=20, max_file_size_bytes=1_000_000
    )
    result = indexer.index(orders_repo)
    return result.repository_id


async def _embed(db_session, fake_embedding_provider, tmp_path, repository_id):
    pipeline = EmbeddingPipeline(db_session, fake_embedding_provider, tmp_path / "vectors")
    await pipeline.sync(repository_id)


async def test_find_symbol_substring_match(db_session, indexed_repository_id):
    tool = FindSymbolTool(db_session)
    result = await tool.run(FindSymbolInput(repository_id=indexed_repository_id, symbol="total", exact=False))
    symbols = {m.symbol for m in result.matches}
    assert "calculate_total" in symbols


async def test_find_symbol_exact_match(db_session, indexed_repository_id):
    tool = FindSymbolTool(db_session)
    result = await tool.run(
        FindSymbolInput(repository_id=indexed_repository_id, symbol="calculate_total", exact=True)
    )
    assert {m.symbol for m in result.matches} == {"calculate_total"}


async def test_find_symbol_exact_match_excludes_partial_names(db_session, indexed_repository_id):
    tool = FindSymbolTool(db_session)
    result = await tool.run(FindSymbolInput(repository_id=indexed_repository_id, symbol="total", exact=True))
    assert result.matches == []


async def test_find_symbol_unknown_repository_raises(db_session):
    tool = FindSymbolTool(db_session)
    with pytest.raises(ToolError, match="not found"):
        await tool.run(FindSymbolInput(repository_id="nope", symbol="anything"))


async def test_find_references_finds_usage_site_not_just_definition(db_session, indexed_repository_id):
    tool = FindReferencesTool(db_session)
    result = await tool.run(FindReferencesInput(repository_id=indexed_repository_id, symbol="calculate_total"))
    symbols_containing_hit = {r.symbol for r in result.references}
    # defined in calculate_total, and called from inside checkout
    assert "calculate_total" in symbols_containing_hit
    assert "checkout" in symbols_containing_hit


async def test_find_references_respects_word_boundaries(db_session, indexed_repository_id):
    tool = FindReferencesTool(db_session)
    # calculate_total's own body has no standalone "total" token — only the
    # substring inside its own name — so a whole-word search for "total"
    # must not match it via a naive substring search.
    result = await tool.run(FindReferencesInput(repository_id=indexed_repository_id, symbol="total"))
    matched_symbols = {r.symbol for r in result.references}
    assert "calculate_total" not in matched_symbols
    # apply_discount and checkout both use `total` as a standalone identifier.
    assert {"apply_discount", "checkout"} <= matched_symbols


async def test_search_code_returns_relevant_results(db_session, indexed_repository_id, fake_embedding_provider, tmp_path):
    await _embed(db_session, fake_embedding_provider, tmp_path, indexed_repository_id)
    tool = SearchCodeTool(db_session, fake_embedding_provider, tmp_path / "vectors")
    result = await tool.run(
        SearchCodeInput(repository_id=indexed_repository_id, query="def calculate_total(items):", top_k=5)
    )
    assert result.results
    assert result.results[0].symbol == "calculate_total"


async def test_search_code_unknown_repository_raises(db_session, fake_embedding_provider, tmp_path):
    tool = SearchCodeTool(db_session, fake_embedding_provider, tmp_path / "vectors")
    with pytest.raises(ToolError, match="not found"):
        await tool.run(SearchCodeInput(repository_id="nope", query="anything"))
