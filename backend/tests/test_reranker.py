from app.retrieval.reranker import KeywordOverlapReranker, NoopReranker, get_reranker
from app.schemas.search import SearchResult


def _result(chunk_id: str, symbol: str | None, file_path: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        file_path=file_path,
        language="python",
        symbol=symbol,
        start_line=1,
        end_line=5,
        content="...",
        score=score,
    )


def test_noop_reranker_returns_results_unchanged():
    results = [_result("a", "foo", "a.py", 0.5), _result("b", "bar", "b.py", 0.9)]
    assert NoopReranker().rerank("anything", results) == results


def test_keyword_overlap_reranker_boosts_symbol_match_above_higher_raw_score():
    # "other" scores higher on raw similarity, but "calculate_total" is an
    # exact symbol match for the query — the lexical boost should flip the order.
    calculate_total = _result("a", "calculate_total", "orders/service.py", 0.60)
    other = _result("b", "unrelated_helper", "utils.py", 0.62)

    reranked = KeywordOverlapReranker(weight=0.5).rerank(
        "calculate_total for an order", [other, calculate_total]
    )

    assert reranked[0].chunk_id == "a"


def test_keyword_overlap_reranker_is_stable_when_no_lexical_signal():
    results = [_result("a", "foo", "a.py", 0.9), _result("b", "bar", "b.py", 0.5)]
    reranked = KeywordOverlapReranker(weight=0.5).rerank("xyz completely unrelated", results)
    assert [r.chunk_id for r in reranked] == ["a", "b"]


def test_keyword_overlap_reranker_handles_empty_query():
    results = [_result("a", "foo", "a.py", 0.9), _result("b", "bar", "b.py", 0.5)]
    assert KeywordOverlapReranker().rerank("   ", results) == results


def test_get_reranker_dispatches_by_flag():
    assert isinstance(get_reranker(enabled=False), NoopReranker)
    assert isinstance(get_reranker(enabled=True), KeywordOverlapReranker)
