import logging

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.config.settings import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="Local AI Software Engineer",
    description="A local, autonomous AI software engineering agent.",
    version="0.1.0",
)

app.include_router(health_router, prefix="/api")
