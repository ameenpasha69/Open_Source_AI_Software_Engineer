from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.models import Repository
from app.database.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.retrieval.embedding_pipeline import EmbeddingPipeline
from app.retrieval.reranker import get_reranker
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> SearchResponse:
    repository = session.get(Repository, request.repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail=f"Repository '{request.repository_id}' not found")

    pipeline = EmbeddingPipeline(
        session=session, embedding_provider=embedding_provider, vector_index_dir=settings.vector_index_dir
    )
    results = await pipeline.search(request.repository_id, request.query, top_k=request.top_k)

    reranking_enabled = settings.search_reranking_enabled if request.rerank is None else request.rerank
    if reranking_enabled:
        results = get_reranker(enabled=True).rerank(request.query, results)

    return SearchResponse(
        query=request.query,
        repository_id=request.repository_id,
        reranking_applied=reranking_enabled,
        results=results,
    )
