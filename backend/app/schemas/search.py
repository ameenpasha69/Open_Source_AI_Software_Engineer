from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: str
    file_path: str
    language: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str
    score: float
