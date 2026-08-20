from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings
from app.database.session import get_db_session
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import get_embedding_provider
from app.schemas.tools import ExecuteToolRequest
from app.tools.base import ToolExecutor, ToolResult, ToolSpec
from app.tools.registry_factory import build_tool_registry

router = APIRouter(tags=["tools"])


@router.get("/tools", response_model=list[ToolSpec])
async def list_tools(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> list[ToolSpec]:
    registry = build_tool_registry(session, settings, embedding_provider)
    return registry.list_specs()


@router.post("/tools/execute", response_model=ToolResult)
async def execute_tool(
    request: ExecuteToolRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    embedding_provider: EmbeddingProvider = Depends(get_embedding_provider),
) -> ToolResult:
    registry = build_tool_registry(session, settings, embedding_provider)
    executor = ToolExecutor(registry)
    return await executor.execute(request.tool_name, request.input)
