from fastapi import APIRouter, Depends

from app.config.settings import Settings, get_settings
from app.llm.base import LLMProvider
from app.llm.factory import get_llm_provider
from app.schemas.health import HealthResponse, LLMHealth

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings),
    llm: LLMProvider = Depends(get_llm_provider),
) -> HealthResponse:
    llm_reachable = await llm.health_check()
    return HealthResponse(
        status="ok" if llm_reachable else "degraded",
        app_name=settings.app_name,
        llm=LLMHealth(
            provider=settings.llm_provider,
            model=settings.llm_model,
            reachable=llm_reachable,
        ),
    )
