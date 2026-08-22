"""A curated catalog of local models the user can pull, with the metadata a
UI needs to make an informed choice (size on disk, parameter count, context
window, what it's good at).

This is deliberately a hand-maintained list rather than a live scrape of a
registry: it stays useful with no network, and every entry here has been
picked as a sensible option for *this* application (code understanding,
patching, and retrieval) rather than being a dump of everything published.
Users are never limited to it — anything installed on the backend shows up
in the installed list, catalogued or not.

Sizes are approximate download sizes for the default quantization of each
tag; the exact on-disk size of an installed model is always read back from
the backend rather than taken from here.
"""

from typing import Literal

from pydantic import BaseModel

ModelRole = Literal["chat", "embedding"]

_MB = 1024 * 1024
_GB = 1024 * _MB


class CatalogEntry(BaseModel):
    """One pullable model offered in the UI."""

    name: str  # the exact tag to pull, e.g. "qwen2.5-coder:7b"
    role: ModelRole
    label: str
    publisher: str
    parameter_size: str
    approx_size_bytes: int
    context_window: int | None = None
    description: str
    strengths: list[str] = []
    # Rough amount of system/GPU memory to run this comfortably. Advisory
    # only — the UI shows it so a user doesn't pull 20GB onto an 8GB laptop.
    min_ram_gb: int
    recommended: bool = False
    # Embedding models only: output vector dimensionality. Switching between
    # models with different dimensions invalidates existing indexes.
    dimensions: int | None = None


CHAT_MODELS: list[CatalogEntry] = [
    CatalogEntry(
        name="qwen2.5-coder:1.5b",
        role="chat",
        label="Qwen2.5 Coder 1.5B",
        publisher="Alibaba",
        parameter_size="1.5B",
        approx_size_bytes=986 * _MB,
        context_window=32768,
        description=(
            "The smallest coding model that still follows tool-calling instructions. "
            "Fast on CPU-only machines; expect it to need more iterations."
        ),
        strengths=["fastest", "low memory", "CPU friendly"],
        min_ram_gb=4,
    ),
    CatalogEntry(
        name="qwen2.5-coder:7b",
        role="chat",
        label="Qwen2.5 Coder 7B",
        publisher="Alibaba",
        parameter_size="7B",
        approx_size_bytes=4700 * _MB,
        context_window=32768,
        description=(
            "The default. Best balance of code reasoning, structured-output "
            "reliability, and speed for a 16GB machine."
        ),
        strengths=["balanced", "reliable JSON", "code-tuned"],
        min_ram_gb=8,
        recommended=True,
    ),
    CatalogEntry(
        name="qwen2.5-coder:14b",
        role="chat",
        label="Qwen2.5 Coder 14B",
        publisher="Alibaba",
        parameter_size="14B",
        approx_size_bytes=9 * _GB,
        context_window=32768,
        description="Noticeably better multi-step reasoning than 7B, at roughly half the speed.",
        strengths=["stronger reasoning", "code-tuned"],
        min_ram_gb=16,
    ),
    CatalogEntry(
        name="qwen2.5-coder:32b",
        role="chat",
        label="Qwen2.5 Coder 32B",
        publisher="Alibaba",
        parameter_size="32B",
        approx_size_bytes=20 * _GB,
        context_window=32768,
        description="The strongest local coder here. Needs a large GPU or a lot of patience.",
        strengths=["highest quality", "code-tuned"],
        min_ram_gb=32,
    ),
    CatalogEntry(
        name="qwen3:8b",
        role="chat",
        label="Qwen3 8B",
        publisher="Alibaba",
        parameter_size="8B",
        approx_size_bytes=5200 * _MB,
        context_window=40960,
        description="General-purpose successor to Qwen2.5 with a hybrid thinking mode.",
        strengths=["general purpose", "long context"],
        min_ram_gb=8,
    ),
    CatalogEntry(
        name="deepseek-coder-v2:16b",
        role="chat",
        label="DeepSeek Coder V2 16B",
        publisher="DeepSeek",
        parameter_size="16B (MoE)",
        approx_size_bytes=8900 * _MB,
        context_window=32768,
        description=(
            "Mixture-of-experts coder: 16B on disk but only ~2.4B active per token, "
            "so it runs faster than its size suggests."
        ),
        strengths=["fast for its size", "code-tuned"],
        min_ram_gb=16,
    ),
    CatalogEntry(
        name="devstral:24b",
        role="chat",
        label="Devstral 24B",
        publisher="Mistral AI",
        parameter_size="24B",
        approx_size_bytes=14 * _GB,
        context_window=131072,
        description="Trained specifically for agentic software engineering — exploring repos and editing files.",
        strengths=["agentic", "tool use", "long context"],
        min_ram_gb=32,
    ),
    CatalogEntry(
        name="codellama:7b",
        role="chat",
        label="Code Llama 7B",
        publisher="Meta",
        parameter_size="7B",
        approx_size_bytes=3800 * _MB,
        context_window=16384,
        description="Older but well-understood code model. Useful as a baseline when comparing models.",
        strengths=["baseline", "widely benchmarked"],
        min_ram_gb=8,
    ),
    CatalogEntry(
        name="llama3.1:8b",
        role="chat",
        label="Llama 3.1 8B",
        publisher="Meta",
        parameter_size="8B",
        approx_size_bytes=4900 * _MB,
        context_window=131072,
        description="Strong general-purpose instruction following with a very long context window.",
        strengths=["general purpose", "long context", "tool use"],
        min_ram_gb=8,
    ),
    CatalogEntry(
        name="llama3.2:3b",
        role="chat",
        label="Llama 3.2 3B",
        publisher="Meta",
        parameter_size="3B",
        approx_size_bytes=2 * _GB,
        context_window=131072,
        description="Small general model that still handles long context. Good on modest hardware.",
        strengths=["small", "long context"],
        min_ram_gb=4,
    ),
    CatalogEntry(
        name="mistral:7b",
        role="chat",
        label="Mistral 7B",
        publisher="Mistral AI",
        parameter_size="7B",
        approx_size_bytes=4100 * _MB,
        context_window=32768,
        description="Fast, general-purpose model. A reasonable fallback when a coder model is unavailable.",
        strengths=["fast", "general purpose"],
        min_ram_gb=8,
    ),
    CatalogEntry(
        name="gemma3:4b",
        role="chat",
        label="Gemma 3 4B",
        publisher="Google",
        parameter_size="4B",
        approx_size_bytes=3300 * _MB,
        context_window=131072,
        description="Compact Google model with a large context window and solid instruction following.",
        strengths=["small", "long context"],
        min_ram_gb=8,
    ),
    CatalogEntry(
        name="gemma3:12b",
        role="chat",
        label="Gemma 3 12B",
        publisher="Google",
        parameter_size="12B",
        approx_size_bytes=8100 * _MB,
        context_window=131072,
        description="Mid-size general model — a good non-Qwen second opinion when comparing runs.",
        strengths=["general purpose", "long context"],
        min_ram_gb=16,
    ),
    CatalogEntry(
        name="phi4:14b",
        role="chat",
        label="Phi-4 14B",
        publisher="Microsoft",
        parameter_size="14B",
        approx_size_bytes=9100 * _MB,
        context_window=16384,
        description="Reasoning-focused model that punches above its parameter count on logic-heavy tasks.",
        strengths=["reasoning", "math"],
        min_ram_gb=16,
    ),
]

EMBEDDING_MODELS: list[CatalogEntry] = [
    CatalogEntry(
        name="nomic-embed-text",
        role="embedding",
        label="Nomic Embed Text",
        publisher="Nomic AI",
        parameter_size="137M",
        approx_size_bytes=274 * _MB,
        context_window=8192,
        dimensions=768,
        description="The default. Long input window and good code-retrieval quality for its size.",
        strengths=["long input", "balanced"],
        min_ram_gb=2,
        recommended=True,
    ),
    CatalogEntry(
        name="mxbai-embed-large",
        role="embedding",
        label="MxBai Embed Large",
        publisher="Mixedbread AI",
        parameter_size="335M",
        approx_size_bytes=670 * _MB,
        context_window=512,
        dimensions=1024,
        description="Higher-dimensional vectors and strong retrieval scores, but a short input window.",
        strengths=["high quality", "1024-dim"],
        min_ram_gb=2,
    ),
    CatalogEntry(
        name="bge-m3",
        role="embedding",
        label="BGE-M3",
        publisher="BAAI",
        parameter_size="567M",
        approx_size_bytes=1200 * _MB,
        context_window=8192,
        dimensions=1024,
        description="Multilingual, long-input embedding model. The best choice for non-English codebases.",
        strengths=["multilingual", "long input", "1024-dim"],
        min_ram_gb=4,
    ),
    CatalogEntry(
        name="all-minilm",
        role="embedding",
        label="All-MiniLM",
        publisher="Sentence Transformers",
        parameter_size="23M",
        approx_size_bytes=46 * _MB,
        context_window=256,
        dimensions=384,
        description="Tiny and very fast. Lowest retrieval quality here — useful for indexing huge repos quickly.",
        strengths=["tiny", "fastest indexing"],
        min_ram_gb=1,
    ),
    CatalogEntry(
        name="snowflake-arctic-embed2",
        role="embedding",
        label="Snowflake Arctic Embed 2",
        publisher="Snowflake",
        parameter_size="568M",
        approx_size_bytes=1200 * _MB,
        context_window=8192,
        dimensions=1024,
        description="Retrieval-tuned multilingual embeddings with a long input window.",
        strengths=["retrieval-tuned", "multilingual"],
        min_ram_gb=4,
    ),
]

MODEL_CATALOG: list[CatalogEntry] = [*CHAT_MODELS, *EMBEDDING_MODELS]

_BY_NAME = {entry.name: entry for entry in MODEL_CATALOG}

# Substrings that identify an embedding model when it isn't in the catalog.
# Embedding models are named far more consistently than chat models, so a
# name check is reliable enough for a UI hint; anything unmatched is treated
# as a chat model, which is the safe default (it stays listed and usable).
_EMBEDDING_NAME_HINTS = ("embed", "bge-", "bge:", "gte-", "minilm", "e5-", "-e5")


def find_catalog_entry(name: str) -> CatalogEntry | None:
    """Look up a catalog entry by tag, tolerating a missing `:latest` on
    either side — Ollama reports `nomic-embed-text:latest` for what the
    catalog (and the user) calls `nomic-embed-text`."""
    if name in _BY_NAME:
        return _BY_NAME[name]
    base = name.removesuffix(":latest")
    return _BY_NAME.get(base) or _BY_NAME.get(f"{base}:latest")


def infer_role(name: str, family: str | None = None) -> ModelRole:
    """Best-effort role for a model that isn't in the catalog."""
    entry = find_catalog_entry(name)
    if entry is not None:
        return entry.role
    haystack = f"{name} {family or ''}".lower()
    return "embedding" if any(hint in haystack for hint in _EMBEDDING_NAME_HINTS) else "chat"


def names_match(a: str, b: str) -> bool:
    """True when two tags refer to the same model, ignoring an implicit
    `:latest`. Ollama's own listings are inconsistent about it."""
    return a.removesuffix(":latest") == b.removesuffix(":latest")
