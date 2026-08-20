from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.tools.base import ToolRegistry
from app.tools.code_search_tools import FindReferencesTool, FindSymbolTool, SearchCodeTool
from app.tools.file_tools import GetFileContextTool, ListFilesTool, ReadFileTool
from app.tools.git_tools import GetGitDiffTool, GetGitLogTool, GetGitStatusTool


def build_tool_registry(session: Session, settings: Settings, embedding_provider: EmbeddingProvider) -> ToolRegistry:
    """Assembles every available tool with its request-scoped dependencies
    (DB session, embedding provider, config). Built fresh per request/agent
    run rather than as a singleton, since the DB session itself is
    request-scoped.
    """
    registry = ToolRegistry()
    registry.register(ListFilesTool(session))
    registry.register(ReadFileTool(session, settings.max_indexable_file_size_bytes))
    registry.register(GetFileContextTool(session, settings.max_indexable_file_size_bytes))
    registry.register(SearchCodeTool(session, embedding_provider, settings.vector_index_dir))
    registry.register(FindSymbolTool(session))
    registry.register(FindReferencesTool(session))
    registry.register(GetGitStatusTool(session))
    registry.register(GetGitDiffTool(session))
    registry.register(GetGitLogTool(session))
    return registry
