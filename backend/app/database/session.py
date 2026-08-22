from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import Settings, get_settings
from app.database.migrations import add_missing_columns
from app.database.models import Base


def create_sqlite_engine(database_url: str) -> Engine:
    engine = create_engine(database_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    # create_all() adds missing tables but never missing columns, so an
    # existing database silently stays behind the models — see migrations.py.
    add_missing_columns(engine)
    return engine


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return create_sqlite_engine(settings.database_url)


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def get_session_factory_dependency() -> sessionmaker[Session]:
    """FastAPI-injectable accessor for the session *factory* itself, not a
    session — for code that must create its own session(s) outside the
    request lifecycle (a background asyncio.Task can't use `Depends()`).
    Override this alongside `get_db_session` in tests, pointed at the same
    factory, so a background task in a test doesn't fall through to the
    real on-disk database.
    """
    return get_session_factory()


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
