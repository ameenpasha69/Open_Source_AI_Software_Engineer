from pydantic import BaseModel, Field, computed_field


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    language: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    score: float

    @computed_field
    @property
    def location(self) -> str:
        """e.g. "orders/service.py:142-188" — a citeable source location."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class SearchRequest(BaseModel):
    repository_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    rerank: bool | None = None  # None = use the server default (SEARCH_RERANKING_ENABLED)


class SearchResponse(BaseModel):
    query: str
    repository_id: str
    reranking_applied: bool
    results: list[SearchResult]
