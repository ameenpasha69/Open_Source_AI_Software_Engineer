from pydantic import BaseModel


class LLMHealth(BaseModel):
    provider: str
    model: str
    reachable: bool


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm: LLMHealth
