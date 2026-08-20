from typing import Any

from pydantic import BaseModel


class ExecuteToolRequest(BaseModel):
    tool_name: str
    input: dict[str, Any]
