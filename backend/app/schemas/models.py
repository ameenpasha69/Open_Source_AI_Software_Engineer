from pydantic import BaseModel, Field

from app.llm.catalog import CatalogEntry, ModelRole


class InstalledModelOut(BaseModel):
    """A model present on the backend, annotated with everything the UI needs
    to render it without a second lookup."""

    name: str
    role: ModelRole
    size_bytes: int
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    modified_at: str | None = None
    context_length: int | None = None
    # True for the model this role currently runs on.
    active: bool = False
    # Resident in the backend's memory right now, so it answers without a
    # cold-start delay.
    loaded: bool = False
    # The catalog entry this matches, when it's one of the curated models —
    # gives the UI a description and a context window for known models.
    catalog: CatalogEntry | None = None


class CatalogEntryOut(CatalogEntry):
    installed: bool = False
    active: bool = False


class ModelBackendInfo(BaseModel):
    provider: str
    base_url: str
    reachable: bool
    # Why the backend couldn't be listed, when it couldn't. The endpoint
    # still answers 200 with the catalog in that case — "Ollama isn't
    # running" is a state the models page exists to help fix, not an error
    # that should blank the page.
    error: str | None = None


class ModelsResponse(BaseModel):
    backend: ModelBackendInfo
    active_chat_model: str
    active_embedding_model: str
    installed: list[InstalledModelOut]
    catalog: list[CatalogEntryOut]


class SetActiveModelRequest(BaseModel):
    role: ModelRole
    name: str = Field(min_length=1)
    # Switching the embedding model invalidates every vector already indexed
    # (different models produce incompatible vectors, often of a different
    # dimension). The switch is refused with 409 until the caller opts in.
    confirm_reindex: bool = False


class ActiveModelsResponse(BaseModel):
    active_chat_model: str
    active_embedding_model: str
    # Set when an embedding switch cleared existing vectors — these
    # repositories return no search results until they're re-indexed.
    repositories_to_reindex: list[str] = []
    message: str


class PullModelRequest(BaseModel):
    name: str = Field(min_length=1)
