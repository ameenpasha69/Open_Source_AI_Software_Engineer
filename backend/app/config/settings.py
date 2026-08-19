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

    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5-coder:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_request_timeout_seconds: float = 120.0
    llm_default_temperature: float = 0.2

    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str = "http://localhost:11434"

    max_agent_iterations: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
