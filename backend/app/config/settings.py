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

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'app.db'}"

    @property
    def vector_index_dir(self) -> Path:
        return self.data_dir / "vector_indexes"


@lru_cache
def get_settings() -> Settings:
    return Settings()
