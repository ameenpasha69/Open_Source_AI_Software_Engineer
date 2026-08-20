import re
from pathlib import Path

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import CodeChunk
from app.embeddings.base import EmbeddingProvider
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.schemas.search import SearchResult
from app.tools.base import Tool
from app.tools.repo_utils import get_repository


class SearchCodeInput(BaseModel):
    repository_id: str
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class SearchCodeOutput(BaseModel):
    results: list[SearchResult]


class SearchCodeTool(Tool):
    name = "search_code"
    description = "Semantic search over the indexed repository — find code by meaning, not just keywords."
    input_schema = SearchCodeInput
    output_schema = SearchCodeOutput
    timeout_seconds = 30.0

    def __init__(self, session: Session, embedding_provider: EmbeddingProvider, vector_index_dir: Path):
        self._session = session
        self._pipeline = EmbeddingPipeline(session, embedding_provider, vector_index_dir)

    async def run(self, input_data: SearchCodeInput) -> SearchCodeOutput:
        get_repository(self._session, input_data.repository_id)  # raises ToolError if unknown
        results = await self._pipeline.search(input_data.repository_id, input_data.query, top_k=input_data.top_k)
        return SearchCodeOutput(results=results)


class SymbolMatch(BaseModel):
    chunk_id: str
    file_path: str
    language: str
    symbol: str
    start_line: int
    end_line: int
    content: str


class FindSymbolInput(BaseModel):
    repository_id: str
    symbol: str
    exact: bool = False


class FindSymbolOutput(BaseModel):
    matches: list[SymbolMatch]


class FindSymbolTool(Tool):
    name = "find_symbol"
    description = "Find where a function, class, or method is defined, by name."
    input_schema = FindSymbolInput
    output_schema = FindSymbolOutput
    timeout_seconds = 10.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: FindSymbolInput) -> FindSymbolOutput:
        get_repository(self._session, input_data.repository_id)
        stmt = select(CodeChunk).where(
            CodeChunk.repository_id == input_data.repository_id, CodeChunk.symbol.is_not(None)
        )
        if input_data.exact:
            stmt = stmt.where(CodeChunk.symbol == input_data.symbol)
        else:
            stmt = stmt.where(CodeChunk.symbol.ilike(f"%{input_data.symbol}%"))

        chunks = self._session.scalars(stmt).all()
        return FindSymbolOutput(
            matches=[
                SymbolMatch(
                    chunk_id=c.id,
                    file_path=c.relative_path,
                    language=c.language,
                    symbol=c.symbol,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content,
                )
                for c in chunks
            ]
        )


class Reference(BaseModel):
    chunk_id: str
    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    content: str


class FindReferencesInput(BaseModel):
    repository_id: str
    symbol: str


class FindReferencesOutput(BaseModel):
    references: list[Reference]


class FindReferencesTool(Tool):
    """Lexical (word-boundary) search for where a symbol name appears in
    indexed code — usage sites as well as its definition. This is grep-like,
    not a real cross-reference/call-graph index (that would need per-language
    static analysis this project doesn't have yet); it's honest about that
    trade-off rather than pretending to be more precise than it is.
    """

    name = "find_references"
    description = "Find where a symbol name is mentioned in code (usages, not just its definition)."
    input_schema = FindReferencesInput
    output_schema = FindReferencesOutput
    timeout_seconds = 15.0

    def __init__(self, session: Session):
        self._session = session

    async def run(self, input_data: FindReferencesInput) -> FindReferencesOutput:
        get_repository(self._session, input_data.repository_id)
        pattern = re.compile(r"\b" + re.escape(input_data.symbol) + r"\b")

        chunks = self._session.scalars(
            select(CodeChunk).where(CodeChunk.repository_id == input_data.repository_id)
        ).all()
        matches = [c for c in chunks if pattern.search(c.content)]

        return FindReferencesOutput(
            references=[
                Reference(
                    chunk_id=c.id,
                    file_path=c.relative_path,
                    symbol=c.symbol,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    content=c.content,
                )
                for c in matches
            ]
        )
