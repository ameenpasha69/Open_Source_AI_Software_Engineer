"""Managing which local models are installed and which ones the agent uses.

Everything here talks to whatever local inference backend is configured via
the `ModelManager` abstraction — this module never assumes Ollama.
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config.model_selection import get_active_model, set_active_model
from app.config.settings import Settings, get_settings
from app.database.models import Repository, VectorRecord
from app.database.session import get_db_session
from app.llm.catalog import MODEL_CATALOG, ModelRole, find_catalog_entry, names_match
from app.llm.exceptions import LLMProviderError
from app.llm.factory import get_model_manager
from app.llm.management import InstalledModel, ModelManagementError, ModelManager
from app.schemas.models import (
    ActiveModelsResponse,
    CatalogEntryOut,
    InstalledModelOut,
    ModelBackendInfo,
    ModelsResponse,
    PullModelRequest,
    SetActiveModelRequest,
)

router = APIRouter(tags=["models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    manager: ModelManager = Depends(get_model_manager),
) -> ModelsResponse:
    """Installed models plus the pullable catalog, in one call.

    Answers 200 even when the backend is unreachable — the catalog is static
    and a user staring at an empty page can't tell "Ollama is down" from
    "this feature is broken". `backend.error` carries the reason instead.
    """
    active_chat = get_active_model(session, settings, "chat")
    active_embedding = get_active_model(session, settings, "embedding")

    installed: list[InstalledModel] = []
    loaded: set[str] = set()
    error: str | None = None
    try:
        installed = await manager.list_installed()
        loaded = set(await manager.list_loaded())
    except LLMProviderError as exc:
        error = str(exc)

    installed_out = [
        InstalledModelOut(
            **model.model_dump(),
            active=names_match(model.name, active_chat if model.role == "chat" else active_embedding),
            loaded=model.name in loaded,
            catalog=find_catalog_entry(model.name),
        )
        for model in installed
    ]
    installed_names = [m.name for m in installed]

    catalog_out = [
        CatalogEntryOut(
            **entry.model_dump(),
            installed=any(names_match(name, entry.name) for name in installed_names),
            active=names_match(entry.name, active_chat if entry.role == "chat" else active_embedding),
        )
        for entry in MODEL_CATALOG
    ]

    return ModelsResponse(
        backend=ModelBackendInfo(
            provider=settings.llm_provider,
            base_url=settings.llm_base_url,
            reachable=error is None,
            error=error,
        ),
        active_chat_model=active_chat,
        active_embedding_model=active_embedding,
        installed=installed_out,
        catalog=catalog_out,
    )


@router.post("/models/active", response_model=ActiveModelsResponse)
async def set_active(
    request: SetActiveModelRequest,
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    manager: ModelManager = Depends(get_model_manager),
) -> ActiveModelsResponse:
    """Switch the model used for a role. Takes effect on the next run — an
    agent run already in flight keeps the model it started with."""
    await _assert_usable(manager, request.name, request.role)

    current = get_active_model(session, settings, request.role)
    if names_match(current, request.name):
        return _active_response(session, settings, message=f"'{request.name}' is already active.")

    repositories_to_reindex: list[str] = []
    if request.role == "embedding":
        repositories_to_reindex = _repositories_with_vectors(session)
        if repositories_to_reindex and not request.confirm_reindex:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Switching the embedding model from '{current}' to '{request.name}' discards "
                    f"the embeddings for {len(repositories_to_reindex)} repository(ies) "
                    f"({', '.join(repositories_to_reindex)}) — vectors from two different models "
                    "aren't comparable and often aren't even the same size. Re-send with "
                    "confirm_reindex=true, then re-index those repositories."
                ),
            )
        _discard_embeddings(session, settings)

    set_active_model(session, request.role, request.name)
    session.commit()

    message = f"Now using '{request.name}' for {request.role}."
    if repositories_to_reindex:
        message += (
            f" Cleared existing embeddings — re-index "
            f"{', '.join(repositories_to_reindex)} to restore code search."
        )
    return _active_response(session, settings, message, repositories_to_reindex)


@router.post("/models/pull")
async def pull_model(
    request: PullModelRequest,
    manager: ModelManager = Depends(get_model_manager),
) -> StreamingResponse:
    """Download a model, streaming progress as NDJSON — one `PullProgress`
    object per line.

    NDJSON rather than SSE because this is a POST (a pull is a mutation, and
    EventSource can only issue GETs), and the client is parsing the body
    itself either way.
    """

    async def stream() -> AsyncIterator[str]:
        async for progress in manager.pull(request.name):
            yield progress.model_dump_json() + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        # Progress is useless if a proxy holds it until the pull finishes.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/models", response_model=ActiveModelsResponse)
async def delete_model(
    name: str = Query(min_length=1, description="Model tag to remove, e.g. 'codellama:7b'"),
    session: Session = Depends(get_db_session),
    settings: Settings = Depends(get_settings),
    manager: ModelManager = Depends(get_model_manager),
) -> ActiveModelsResponse:
    """Remove a model from the backend. Refuses to delete a model that's
    currently in use — switch first, so a run can't fail on a model that
    vanished underneath it."""
    for role in ("chat", "embedding"):
        if names_match(get_active_model(session, settings, role), name):  # type: ignore[arg-type]
            raise HTTPException(
                status_code=409,
                detail=f"'{name}' is the active {role} model. Switch to another model before removing it.",
            )

    try:
        await manager.delete(name)
    except ModelManagementError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return _active_response(session, settings, message=f"Removed '{name}'.")


async def _assert_usable(manager: ModelManager, name: str, role: ModelRole) -> None:
    """Reject a switch to a model that isn't installed, or that can't fill
    the requested role. Both failures are much cheaper to catch here than as
    a 404 from the backend in the middle of an agent run."""
    entry = find_catalog_entry(name)
    if entry is not None and entry.role != role:
        raise HTTPException(
            status_code=400,
            detail=f"'{name}' is a {entry.role} model and can't be used as the {role} model.",
        )

    try:
        installed = await manager.list_installed()
    except LLMProviderError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc} Can't confirm '{name}' is installed, so the switch was not applied.",
        ) from exc

    if not any(names_match(model.name, name) for model in installed):
        raise HTTPException(
            status_code=404,
            detail=f"'{name}' is not installed. Pull it first, then switch to it.",
        )


def _repositories_with_vectors(session: Session) -> list[str]:
    return sorted(
        session.scalars(
            select(Repository.name)
            .join(VectorRecord, VectorRecord.repository_id == Repository.id)
            .distinct()
        )
    )


def _discard_embeddings(session: Session, settings: Settings) -> None:
    """Drop every vector and its index file.

    Both halves have to go: a VectorRecord row is what tells the embedding
    pipeline a chunk is already embedded, so leaving the rows would mean the
    next index run silently keeps the old model's vectors, while leaving the
    .faiss files would mean appending new vectors to an index built at the
    old model's dimensions. The chunks themselves are untouched — re-indexing
    re-embeds them without re-reading the repository.
    """
    session.execute(sa_delete(VectorRecord))
    index_dir = settings.vector_index_dir
    if index_dir.exists():
        for index_file in index_dir.glob("*.faiss"):
            index_file.unlink(missing_ok=True)


def _active_response(
    session: Session,
    settings: Settings,
    message: str,
    repositories_to_reindex: list[str] | None = None,
) -> ActiveModelsResponse:
    return ActiveModelsResponse(
        active_chat_model=get_active_model(session, settings, "chat"),
        active_embedding_model=get_active_model(session, settings, "embedding"),
        repositories_to_reindex=repositories_to_reindex or [],
        message=message,
    )
