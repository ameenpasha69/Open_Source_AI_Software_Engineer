from abc import ABC, abstractmethod

from app.schemas.search import SearchResult


class Reranker(ABC):
    """Optional second pass over vector-search results. Kept as an interface
    (rather than a hardcoded on/off branch) so the evaluation framework can
    compare rerankers, and so a cross-encoder-based reranker can slot in later
    without touching the search endpoint.
    """

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]: ...


class NoopReranker(Reranker):
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results


class KeywordOverlapReranker(Reranker):
    """Blends vector similarity with lexical overlap between the query and each
    chunk's symbol name and file path. Doesn't need an extra model: it exists
    to correct cases where the embedding model misses an exact identifier or
    filename match that's an obvious lexical signal (e.g. a query mentioning
    `calculate_total` should favor a chunk literally named that).
    """

    def __init__(self, weight: float = 0.2):
        self._weight = weight

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = {t for t in query.lower().split() if t}
        if not query_terms:
            return results

        def combined_score(result: SearchResult) -> float:
            haystack = f"{result.symbol or ''} {result.file_path}".lower()
            overlap = sum(1 for term in query_terms if term in haystack) / len(query_terms)
            return result.score + self._weight * overlap

        return sorted(results, key=combined_score, reverse=True)


def get_reranker(enabled: bool) -> Reranker:
    return KeywordOverlapReranker() if enabled else NoopReranker()
