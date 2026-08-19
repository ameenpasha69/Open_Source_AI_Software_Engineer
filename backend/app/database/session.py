from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import Settings, get_settings
from app.database.models import Base


def create_sqlite_engine(database_url: str) -> Engine:
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return engine


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_sqlite_engine(settings.database_url)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def get_db_session() -> Generator[Session]:
    """FastAPI dependency yielding a request-scoped session. Override in tests
    with a session factory bound to a temporary database."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Test-only: clear the cached engine so a fresh Settings.data_dir takes effect."""
    get_engine.cache_clear()


def build_session_factory_for_settings(settings: Settings) -> sessionmaker[Session]:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(settings.database_url)
    return get_session_factory(engine)
