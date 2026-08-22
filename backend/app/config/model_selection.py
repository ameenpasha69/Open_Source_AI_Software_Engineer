"""Which pulled model is currently in use, for each role.

`Settings.llm_model` / `Settings.embedding_model` are the configured
defaults and never change at runtime. When a user switches models in the UI
we record an override in `app_settings` and read through it here, so the
choice survives a restart without rewriting anyone's .env.

Reads fall back to the configured default whenever no override exists — a
fresh database behaves exactly as it did before this table existed.
"""

from sqlalchemy.orm import Session

from app.config.settings import Settings
from app.database.models import AppSetting
from app.llm.catalog import ModelRole

_KEYS: dict[ModelRole, str] = {
    "chat": "active_chat_model",
    "embedding": "active_embedding_model",
}


def default_model(settings: Settings, role: ModelRole) -> str:
    return settings.llm_model if role == "chat" else settings.embedding_model


def get_active_model(session: Session, settings: Settings, role: ModelRole) -> str:
    row = session.get(AppSetting, _KEYS[role])
    if row is None or not row.value.strip():
        return default_model(settings, role)
    return row.value


def set_active_model(session: Session, role: ModelRole, name: str) -> None:
    """Records the override. The caller commits — switching the embedding
    model also has to clear stale vectors, and that has to land in the same
    transaction as the switch itself."""
    key = _KEYS[role]
    row = session.get(AppSetting, key)
    if row is None:
        session.add(AppSetting(key=key, value=name))
    else:
        row.value = name
