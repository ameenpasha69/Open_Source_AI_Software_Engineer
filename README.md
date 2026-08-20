# Local AI Software Engineer

A local, autonomous AI software engineering agent. Given a Git repository and a
natural-language issue, it inspects the codebase, retrieves relevant code
semantically, plans a fix, applies patches, runs the test suite, analyzes
failures, and iterates — all using models running on your own machine.

**Everything runs locally.** No OpenAI/Anthropic/Gemini API keys, no hosted
vector databases, no paid services of any kind.

## Project status

This project is being built incrementally, one milestone at a time (see
`docs/milestones.md` — added as milestones land). **Currently implemented:**

- ✅ **Milestone 1** — Local LLM provider abstraction (Ollama backend) + FastAPI health endpoint
- ✅ **Milestone 2** — Repository indexing + language-aware chunking
- ✅ **Milestone 3** — Local embeddings + FAISS vector search
- ✅ **Milestone 4** — Code retrieval API (`POST /api/search`)

Everything below this line describes what exists *today*, not the end goal.
The full architecture (agent loop, retrieval, tool system, evaluation
framework, etc.) is documented as each milestone is built — see the task
description this project was scoped from for the complete roadmap.

## Why a provider abstraction instead of calling Ollama directly?

The rest of the application (agent runtime, tools, evaluation) is written
against `LLMProvider`, an interface with `generate`, `stream`, and
`health_check`. `OllamaProvider` is the only implementation today, but nothing
outside `backend/app/llm/` imports Ollama-specific code or hard-codes a model
name. Swapping inference backends later (llama.cpp, vLLM, LM Studio) means
adding one new class and registering it in `llm/factory.py`.

```python
class LLMProvider(ABC):
    async def generate(self, messages, *, temperature=None, max_tokens=None) -> LLMResponse: ...
    def stream(self, messages, *, temperature=None, max_tokens=None) -> AsyncIterator[str]: ...
    async def health_check(self) -> bool: ...
```

## Repository indexing and chunking

`POST /api/repositories/index` walks a local repository, ignores dependency
and build directories (`.git`, `node_modules`, `.venv`, `dist`, `__pycache__`,
etc.), and splits each supported source file into chunks with metadata:

```text
repository, file_path, language, chunk_id, start_line, end_line, symbol, content
```

**Chunking strategy** (`backend/app/retrieval/chunker.py`):

- **Python** — AST-based. One chunk per top-level function; classes are kept
  as a single chunk only if small (≤ `chunk_max_lines / 4`), otherwise split
  into one chunk per method (`ClassName.method_name`), so a class doesn't
  swallow multiple unrelated methods into one retrieval unit. Decorators stay
  attached to the function/class they decorate. Module-level code not inside
  any function/class (imports, constants, `if __name__ == "__main__"` blocks)
  is captured separately rather than dropped. Oversized functions/classes are
  further split by the generic windower so no chunk is unbounded.
- **Everything else** (JS/TS/Go/Java/Rust/...) — a fixed-size sliding window
  over lines with configurable overlap (`GenericChunker`). No syntax
  awareness yet; see Limitations.

Re-indexing is incremental: each file's SHA-256 hash is compared against the
last indexed hash, and unchanged files are skipped entirely — no re-parse, no
re-chunk. This is also what lets embedding sync (below) skip re-embedding
unchanged chunks. Files deleted from disk since the last run have their
chunks removed so the index stays accurate.

```bash
curl -X POST http://localhost:8000/api/repositories/index \
  -H "Content-Type: application/json" \
  -d '{"path": "/absolute/path/to/a/repo"}'

curl http://localhost:8000/api/repositories
```

## Embeddings and vector search

`POST /api/repositories/index` also embeds every chunk that doesn't have a
vector yet (`app/retrieval/embedding_pipeline.py`) and stores the vectors in
a per-repository FAISS index (`data/vector_indexes/{repository_id}.faiss`,
inner product over L2-normalized vectors — i.e. cosine similarity):

```json
{
  "chunks_created": 229,
  "embedding": { "chunks_embedded": 229, "chunks_removed": 0, "duration_seconds": 2.46, "error": null }
}
```

**Reconciliation, not blind re-embedding**: a `VectorRecord` row is the
source of truth for "this chunk is embedded." On re-index, only chunks
without a `VectorRecord` get embedded (new or changed files — an unchanged
file's chunks keep their ids, so they're skipped); `VectorRecord`s whose
chunk no longer exists (file changed or deleted) are removed from the FAISS
index. A repository with no changes triggers zero calls to the embedding
model.

**Degrades gracefully**: chunking is persisted to SQLite regardless of
whether the embedding backend is reachable. If Ollama's embedding model isn't
pulled or the server is down, indexing still succeeds and the response's
`embedding.error` reports what went wrong — you don't lose the chunking work.

`EmbeddingPipeline.search(repository_id, query_text, top_k)` embeds the query
and returns ranked chunks — this is the retrieval engine underneath
`POST /api/search`, below.

## Code retrieval API

```bash
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"repository_id": "<id>", "query": "remove a vector from the FAISS index when its file is deleted", "top_k": 5}'
```

```json
{
  "query": "remove a vector from the FAISS index when its file is deleted",
  "repository_id": "8407f954003e4488869678a9cb976a9f",
  "reranking_applied": false,
  "results": [
    {
      "file_path": "app/retrieval/embedding_pipeline.py",
      "symbol": null,
      "location": "app/retrieval/embedding_pipeline.py:18-23",
      "score": 0.702,
      "content": "..."
    }
  ]
}
```

Verified live against this repository's own `backend/` (above is the actual
top hit for that query — the docstring explaining exactly that mechanism).
Each result carries a `location` field (`file_path:start_line-end_line`) so
results are directly citeable, per the project's source-location convention.

**Validation**: `top_k` is bounded (`1`–`100`, HTTP 422 outside that range);
an unknown `repository_id` returns 404 rather than an empty result set, so a
typo'd repository id fails loudly instead of silently returning nothing.

**Reranking** (`app/retrieval/reranker.py`) is optional and off by default
(`SEARCH_RERANKING_ENABLED=false`), overridable per-request via
`{"rerank": true}` for evaluation-framework experiments without a server
restart. The current `KeywordOverlapReranker` is a cheap, explainable signal
— it blends vector similarity with lexical overlap between the query and each
chunk's symbol/file path, so a query mentioning an exact identifier is
guaranteed to favor a chunk actually named that, without needing a
cross-encoder model. A `NoopReranker` (identity) is the default.

## Hardware / model defaults

Defaults were chosen for a machine with 16GB+ RAM and no dedicated GPU
requirement (tested on Apple Silicon, 36GB RAM):

- `LLM_MODEL=qwen2.5-coder:7b` — good code/tool-calling quality at a size that
  keeps per-iteration latency low, which matters because the agent calls the
  LLM many times per run. `qwen2.5-coder:14b` or `32b` are drop-in upgrades
  via `.env` if your hardware and patience allow — no code changes needed.
- `EMBEDDING_MODEL=nomic-embed-text` — served by the same local Ollama
  instance, avoiding an extra heavy dependency (torch/sentence-transformers)
  for embeddings alone.

## Installation

Requires: Python 3.11+, [Ollama](https://ollama.com) installed.

```bash
./scripts/setup.sh
```

This creates a virtualenv, installs dependencies, copies `.env.example` to
`.env`, starts Ollama if it isn't running, and pulls the configured LLM and
embedding models (a few GB — see model sizes above).

## Running

```bash
./scripts/start.sh
```

Then check:

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "ok",
  "app_name": "local-ai-software-engineer",
  "llm": { "provider": "ollama", "model": "qwen2.5-coder:7b", "reachable": true }
}
```

## Configuration

All configuration lives in `.env` (see `.env.example`), loaded via
`app/config/settings.py`. Nothing is hard-coded — model names, base URLs,
timeouts, and agent iteration limits are all environment-driven so different
models/settings can be compared without touching code.

## Testing

```bash
source .venv/bin/activate
pytest -q
```

Run from the project root — pytest's import machinery finds `backend/app`
automatically (no `PYTHONPATH` needed for tests). Tests use a fake
`LLMProvider` (`backend/tests/conftest.py`) and a temporary SQLite database
per test (`tmp_path`), so the suite is fast, deterministic, and needs neither
a live Ollama server nor a real repository.

Note: this project's editable pip install (`pip install -e .`) does not make
`app` importable via `.pth` on this environment's Python 3.14 build — a
`.pth`-processing quirk unrelated to this codebase. `scripts/start.sh` works
around it with uvicorn's `--app-dir backend`; pytest works around it via its
own import-path detection. Neither requires activating a workaround manually.

## Project structure

```text
backend/
  app/
    config/      # environment-driven settings
    llm/         # LLMProvider abstraction + Ollama backend
    database/    # SQLAlchemy models + session management (SQLite)
    embeddings/  # EmbeddingProvider abstraction + Ollama backend
    retrieval/   # repository walker, chunker, FAISS vector store, embedding pipeline
    api/routes/  # FastAPI routers
    schemas/     # Pydantic request/response models
    agents/ tools/ execution/
    verification/ evaluation/ models/   # scaffolded, empty — future milestones
  tests/
scripts/         # setup.sh, start.sh
data/            # local vector index, SQLite DB (gitignored)
evals/           # benchmark tasks (future milestone)
docs/
```

## Limitations (current milestone)

- Only Ollama is implemented as an LLM/embedding provider; the interfaces
  support others but none are built yet.
- No agent or tool system exists yet — search is a standalone API, not
  something an agent calls autonomously as part of investigating an issue.
- Reranking is a simple lexical-overlap heuristic, not a cross-encoder model
  — it corrects obvious cases (exact identifier match) but isn't a learned
  relevance model.
- The FAISS index is a flat (exact) index, rebuilt in-place per repository.
  Fine at the scale a single local repository produces; would need an
  approximate index (IVF/HNSW) to scale to millions of chunks.
- Non-Python languages use a syntax-unaware sliding-window chunker; only
  Python gets function/method-precise chunks. Language support is currently
  strongest for Python, as expected from the project's scope.
- No `.gitignore` awareness — the indexer uses its own fixed ignore-directory
  list, so a repo-specific ignore rule (e.g. a custom build output dir) won't
  be respected until this is added.
- Local model quality varies by hardware and chosen model size.
