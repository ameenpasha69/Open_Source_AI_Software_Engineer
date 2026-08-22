from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    get_engine()  # creates the SQLite file and tables on first use
    yield


app = FastAPI(
    title="Local AI Software Engineer",
    description="A local, autonomous AI software engineering agent.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIDMiddleware)

app.include_router(health_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(repositories_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(tools_router, prefix="/api")
app.include_router(agent_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
