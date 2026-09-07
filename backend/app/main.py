import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.auth import require_token, resolve_auth_token
from app.api.routes.agent import router as agent_router
from app.api.routes.health import router as health_router
from app.api.routes.models import router as models_router
from app.api.routes.repositories import router as repositories_router
from app.api.routes.search import router as search_router
from app.api.routes.sessions import router as sessions_router
from app.api.routes.tools import router as tools_router
from app.config.settings import get_settings
from app.database.session import get_engine
from app.observability.logging import configure_logging
from app.observability.middleware import RequestIDMiddleware

settings = get_settings()
configure_logging(settings.log_level, settings.log_format)


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_engine()  # creates the SQLite file and tables on first use
    _announce_auth()
    yield


def _announce_auth() -> None:
    """Say, at startup, whether this instance is protected and how to reach it.

    A generated token is no use if nobody can find it, and an unauthenticated
    instance should be impossible to run without noticing.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        logger.warning(
            "AUTH IS DISABLED. This API runs commands and writes files; anything "
            "that can reach it can run code on this machine. Only do this behind "
            "something else that authenticates."
        )
        return
    token = resolve_auth_token(settings)
    if settings.api_auth_token:
        logger.info("Auth enabled, using the configured API_AUTH_TOKEN.")
    else:
        logger.info(
            "Auth enabled. Generated token (also stored in %s): %s  "
            "Send it as 'Authorization: Bearer <token>', or set it as "
            "NEXT_PUBLIC_API_AUTH_TOKEN for the frontend.",
            Path(settings.data_dir) / "auth_token",
            token,
        )


app = FastAPI(
    title="Local AI Software Engineer",
    description="A local, autonomous AI software engineering agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.frontend_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

# Every router is behind the token. Nothing here is safe to expose
# unauthenticated: even /api/health reports which models are loaded.
_auth = [Depends(require_token)]

app.include_router(health_router, prefix="/api", dependencies=_auth)
app.include_router(models_router, prefix="/api", dependencies=_auth)
app.include_router(repositories_router, prefix="/api", dependencies=_auth)
app.include_router(search_router, prefix="/api", dependencies=_auth)
app.include_router(tools_router, prefix="/api", dependencies=_auth)
app.include_router(agent_router, prefix="/api", dependencies=_auth)
app.include_router(sessions_router, prefix="/api", dependencies=_auth)
