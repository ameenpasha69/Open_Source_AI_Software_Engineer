from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration, loaded from environment / .env.

    No component should read os.environ directly or hard-code a model name —
    everything that varies by machine or experiment lives here.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "local-ai-software-engineer"
    log_level: str = "INFO"
    # "text" for human-readable console output during development; "json" for
    # one-object-per-line structured logs suitable for a log aggregator.
    log_format: str = "text"
    data_dir: Path = Path("./data")
    # The Next.js dev server's origin — the only one allowed to call this API
    # cross-origin. Everything here runs on localhost; this isn't a
    # public-internet CORS policy, just what a browser-based frontend needs.
    frontend_origin: str = "http://localhost:3000"

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_request_timeout_seconds: float = 120.0
    llm_default_temperature: float = 0.2

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"
    # Texts per /api/embed request. Bounded because a large batch can kill
    # the Ollama model runner on a small GPU -- see OllamaEmbeddingProvider.
    embedding_batch_size: int = 64

    max_agent_iterations: int = 8

    # --- Indexing / chunking ---
    # Max lines per chunk before the generic sliding-window chunker splits further
    # (also applies to oversized Python functions/classes). Tune this for retrieval
    # quality experiments — smaller chunks are more precise, larger ones carry more context.
    chunk_max_lines: int = 200
    chunk_overlap_lines: int = 20
    # Files larger than this are skipped entirely (binary blobs, generated assets, etc.)
    max_indexable_file_size_bytes: int = 1_000_000

    # --- Search ---
    search_default_top_k: int = 10
    # Off by default: raw vector similarity is already a reasonable baseline, and
    # this needs to be comparable on/off for retrieval-quality experiments. Can
    # also be overridden per-request via SearchRequest.rerank.
    search_reranking_enabled: bool = False

    # --- Execution / testing ---
    # Test suites can legitimately take a while; kept separate from the other,
    # much shorter tool timeouts (git commands, file reads).
    test_execution_timeout_seconds: float = 120.0

    # --- Sandboxing ---
    # "subprocess" (default) needs nothing but Python — env-var isolation and an
    # allowlist, but the process still sees the host filesystem and network.
    # "docker" additionally isolates the filesystem (bind-mounts only the target
    # repo), disables networking (--network none), and caps memory/CPU — real
    # container isolation, at the cost of requiring Docker and a prebuilt image
    # (see docker/sandbox.Dockerfile). Falls back to a clear error, not to
    # subprocess, if Docker isn't available — silently downgrading a security
    # boundary the user explicitly asked for would be worse than failing loudly.
    sandbox_backend: str = "subprocess"
    sandbox_docker_image: str = "local-ai-softeng-sandbox:latest"
    sandbox_docker_memory_limit: str = "512m"
    sandbox_docker_cpu_limit: str = "1.0"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'app.db'}"

    @property
    def vector_index_dir(self) -> Path:
        return self.data_dir / "vector_indexes"


@lru_cache
def get_settings() -> Settings:
    return Settings()
