from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.embeddings.base import EmbeddingProvider
from app.execution.subprocess_runner import SandboxSettings
from app.tools.base import ToolRegistry
from app.tools.code_search_tools import FindReferencesTool, FindSymbolTool, SearchCodeTool
from app.tools.execution_tools import RunCommandTool, RunFormatterTool, RunLinterTool, RunTestsTool
from app.tools.file_tools import GetFileContextTool, ListFilesTool, ReadFileTool
from app.tools.git_tools import GetGitDiffTool, GetGitLogTool, GetGitStatusTool
from app.tools.patch_tools import ApplyPatchTool, CreateFileTool, DeleteFileTool


def build_tool_registry(session: Session, settings: Settings, embedding_provider: EmbeddingProvider) -> ToolRegistry:
    """Assembles every available tool with its request-scoped dependencies
    (DB session, embedding provider, config). Built fresh per request/agent
    run rather than as a singleton, since the DB session itself is
    request-scoped.
    """
    sandbox = SandboxSettings(
        backend=settings.sandbox_backend,
        docker_image=settings.sandbox_docker_image,
        docker_memory_limit=settings.sandbox_docker_memory_limit,
        docker_cpu_limit=settings.sandbox_docker_cpu_limit,
    )
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
    registry.register(ApplyPatchTool(session))
    registry.register(CreateFileTool(session))
    registry.register(DeleteFileTool(session))
    registry.register(RunTestsTool(session, settings.test_execution_timeout_seconds, sandbox))
    registry.register(RunCommandTool(session, sandbox))
    registry.register(RunLinterTool(session, sandbox))
    registry.register(RunFormatterTool(session, sandbox))
    return registry
