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
- ✅ **Milestone 5** — Agent tool registry (read-only investigation tools)
- ✅ **Milestone 6** — Basic agent loop (investigation and diagnosis)
- ✅ **Milestone 7** — Code modification (`apply_patch`)

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

## Tool registry

The agent (Milestone 6+) won't get raw shell access — it calls typed tools
through a registry (`app/tools/`). Every tool declares a name, description,
a Pydantic input schema, and an output schema; `ToolExecutor` validates input
against that schema, enforces a per-tool timeout, and converts *every*
failure mode — bad input, an unknown tool, a timeout, an expected `ToolError`,
or even an unanticipated exception — into a `ToolResult(success=False, ...)`
rather than letting it propagate. A misbehaving tool degrades one step of
agent reasoning; it can't crash the run.

```python
class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[type[BaseModel]]
    output_schema: ClassVar[type[BaseModel]]
    timeout_seconds: ClassVar[float] = 30.0

    async def run(self, input_data: BaseModel) -> BaseModel: ...
```

**Available today** (`GET /api/tools`, `POST /api/tools/execute`):

| Tool | What it does |
|---|---|
| `list_files` | List a directory (optionally recursive), same ignore rules as indexing |
| `read_file` | Read a file, optionally a line range |
| `get_file_context` | Read a window of lines centered on a line number |
| `search_code` | Semantic search — the same engine behind `POST /api/search` |
| `find_symbol` | Find where a function/class/method is *defined*, by name |
| `find_references` | Find where a symbol is *mentioned* (lexical, word-boundary — not a real call-graph) |
| `get_git_status` | Branch + staged/unstaged/untracked files |
| `get_git_diff` | Working-tree or staged diff, optionally scoped to a path |
| `get_git_log` | Recent commit history, optionally scoped to a path |
| `apply_patch` | Replace an exact, unique excerpt of a file's content (Milestone 7 — see Code modification) |

**Path safety**: every file-path argument is resolved through
`resolve_safe_path()`, which rejects anything escaping the repository root —
`../../etc/passwd`, an absolute path elsewhere, a symlink pointing out.
Verified live: a `read_file` call with `path: "../../../../etc/passwd"`
against this repository comes back as `{"success": false, "error": "Path
'../../../../etc/passwd' escapes the repository root"}` — a failed tool
result, not a crash, not a file read.

**Git tools** run through `app/execution/subprocess_runner.py` — a fixed
argv list (never a shell string, so there's no injection surface), a hard
timeout, output truncation, and cwd pinned to the target repo. Verified live
against this project's own repo: `get_git_status` correctly reported this
session's actual uncommitted files, and `find_symbol` for `RepositoryIndexer`
correctly returned all four of its real methods.

This runner is deliberately minimal today (only fixed git subcommands use
it). It's the foundation the Milestone 8 sandboxing work extends —
allowlisting, environment isolation, and an optional Docker backend — for
`run_command`, `run_tests`, and `run_linter`/`run_formatter`, which execute
arbitrary or repo-defined commands and need a stronger boundary than a
fixed `git` subcommand does. `apply_patch` (Milestone 7, below) turned out
not to need this at all — it's a filesystem write, not a subprocess, so
`resolve_safe_path` plus its own denylist (env files, ignored directories)
is the complete safety story for it.

## Agent loop

`POST /api/agent/run` is the first version of the actual agent: given a
repository and a natural-language issue, it plans, then repeatedly decides
which tool to call, executes it, and updates its state — until it has enough
evidence to explain the root cause, or it runs out of iterations
(`MAX_AGENT_ITERATIONS`). It **cannot modify code or run tests yet**
(Milestones 7/8 add those tools) — "done" here means "produced a cited
diagnosis," not "fixed the bug."

```text
TASK → PLAN → (OBSERVE → SELECT TOOL → EXECUTE → UPDATE STATE)* → DONE
```

**Structured decisions, not native tool-calling.** I tried Ollama's native
`tools` parameter first — with `qwen2.5-coder:7b` it didn't populate
`message.tool_calls` at all; the model just echoed JSON into plain `content`
instead. Rather than depend on that, the agent uses Ollama's `format: "json"`
constrained decoding (`LLMProvider.generate(..., json_mode=True)`) with an
explicit schema in the prompt — verified far more reliable in practice. Each
turn the model returns exactly one of `action` (call a tool) or `finish`
(stop and answer), validated against a Pydantic model
(`AgentDecision`) that rejects a response setting both or neither.

**Explicit, structured state, not a raw chat transcript** (`AgentState`):
`task`, `repository_id`, `plan`, `observations`, `tool_calls`,
`modified_files` (empty until Milestone 7), `test_results` (empty until
Milestone 8), `iteration`, `status`. Persisted incrementally to SQLite
(`agent_runs`, `agent_events`, `tool_calls` tables) after every iteration, so
a run survives a process restart and is inspectable mid-flight —
`GET /api/agent/{run_id}` and `GET /api/agent/{run_id}/events`.

**The architecture is composed of named, independently testable pieces**,
not one large function:

| Component | File | Responsibility |
|---|---|---|
| `Planner` | `agents/planner.py` | One LLM call up front → a short investigation plan, with a hardcoded fallback plan if the response doesn't parse |
| `ContextManager` | `agents/context_manager.py` | Builds each turn's prompt from `AgentState` (not an ever-growing history); caps to the most recent N observations |
| Observation Handler | `context_manager.format_observation()` | Turns a raw `ToolResult` into a compact, tool-specific summary — this is what actually accumulates in memory, not the full JSON |
| `check_termination()` | `agents/termination.py` | One inspectable function deciding RUNNING vs. a terminal status — not `if`s scattered through the loop |
| `AgentRunner` | `agents/runner.py` | Orchestrates the above + `ToolExecutor` + DB persistence |

**repository_id is injected, never LLM-supplied.** Tool input schemas
require `repository_id` (so `/api/tools/execute` works standalone), but the
agent always overrides whatever the model puts there with the run's actual
repository before executing — the model can't accidentally or adversarially
point a tool call at a different repository than the one it was scoped to.

**Verified live** against this project's own `backend/` with real Ollama —
asked to find where a FAISS vector is removed when a chunk is deleted, the
agent's plan and tool sequence were genuinely sensible (`search_code` →
`read_file` → `find_symbol` → `get_file_context`), every tool call
succeeded, and it correctly converged on `FaissVectorStore.delete` in
`vector_store.py` — see Limitations for what it *didn't* do well.

## Code modification

The agent can now change code, via a new `apply_patch` tool
(`app/tools/patch_tools.py`) and `GET /api/agent/{run_id}/diff`.

**Search-and-replace with context validation, not a unified diff.** Spec
section 10 asks for "validate the expected surrounding context" before
patching — I read that as a design constraint, not just a checklist item.
Small local models are unreliable at producing correct line numbers and hunk
headers for a real unified diff, but are reasonably good at reproducing a
short, *exact* excerpt of code they just read. So `apply_patch` takes
`old_content` (the expected context) and `new_content`, and:

- **Zero matches** → `ToolError`: the expected context wasn't found (the
  model may be misremembering the file — it's told to re-read it).
- **More than one match** → `ToolError`: the patch is ambiguous about which
  occurrence to change; applying the wrong one would be worse than refusing.
- **Exactly one match** → applied, and a real unified diff (`difflib`) is
  generated *from the actual before/after content* — not trusted from the
  model — for the observation, the DB record, and the diff API.

**Protected regardless of repo boundary.** `resolve_safe_path` (Milestone 5)
guarantees a patch target is inside the repo, but "inside the repo" still
includes `.env` and `.git/` — `apply_patch` separately refuses env/secret
filenames and anything under an ignored directory (`.git`, `node_modules`,
etc.), verified in tests.

**Every modification is recorded independent of the tool call log**: a
successful `apply_patch` also writes a `ModifiedFile` row (path, diff,
+/- line counts) and appends to `AgentState.modified_files`, so
`GET /api/agent/{run_id}/diff` doesn't need the target to be a git
repository at all — it's built entirely from what the agent actually did,
not from shelling out to `git diff`.

**Always "unverified."** `AgentState.verification_status` is `"unverified"`
whenever `modified_files` is non-empty, `"not_applicable"` otherwise — there
is no test-running capability yet (Milestone 8), so a modification can never
be more than that. This is enforced in code, not just prompted for — the
field is computed from state, not something the model has to remember to say.

**Verified live, with a real bug**: I wrote a repo with a genuine bug
(`total = item.price` inside a loop — overwrites instead of accumulates —
so `calculate_total` returns only the last item's price). Given a task that
named the file, the agent read it, correctly diagnosed the root cause,
called `apply_patch`, and the file on disk was correctly fixed to
`total += item.price`:

```json
{
  "status": "done",
  "final_answer": "The bug in orders.py's calculate_total function has been fixed by changing 'total = item.price' to 'total += item.price'.",
  "root_cause": "The original code was incorrectly resetting the total price on each iteration instead of accumulating it.",
  "modified_files": ["orders.py"],
  "verification_status": "unverified"
}
```

```diff
--- a/orders.py
+++ b/orders.py
@@ -2,5 +2,5 @@
     """Sum up the price of every item in the order."""
     total = 0
     for item in items:
-        total = item.price
+        total += item.price
     return total
```

A vaguer version of the same task (not naming the file) hit
`MAX_AGENT_ITERATIONS` without patching anything — even though `search_code`
and `find_references` had already surfaced the correct file and the exact
buggy line by iteration 5. I confirmed this by inspecting the actual tool
output the model had seen (not just its stated reasoning): the correct path
and the buggy line were right there, and it still spent the next three
iterations guessing at a `src/orders.py` that doesn't exist instead of using
`orders.py`, which it had already been shown twice. Same finding as
Milestone 6, now with `apply_patch` too: this looks like an attention/
recall limitation under a long agentic loop, not a tooling bug — the tools
returned exactly the right information both times.

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
    tools/       # Tool/ToolRegistry/ToolExecutor + file, code-search, git, and patch tools
    execution/   # sandboxed subprocess runner (fixed argv, timeout, output caps)
    api/routes/  # FastAPI routers
    schemas/     # Pydantic request/response models
    agents/      # AgentRunner, Planner, ContextManager, state, termination logic
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
- **The agent can act but can't verify.** `apply_patch` exists, but
  `run_command`/`run_tests` don't yet (Milestone 8), so every modification
  is permanently "unverified" until then — the agent has no way to confirm
  a patch actually fixes anything, or even that it didn't break something else.
- **Small local models under-commit to finishing — and a bigger model isn't
  automatically the fix.** In live testing on the same investigation task
  (`qwen2.5-coder:7b`, default), the agent chose sensible tools, every call
  succeeded, and it correctly converged on the real answer
  (`FaissVectorStore.delete` in `vector_store.py`) — but never emitted
  `finish`, hitting `MAX_AGENT_ITERATIONS` after re-running a couple of
  near-duplicate searches instead of concluding. I re-ran the identical task
  against `qwen2.5-coder:14b` expecting better loop closure; instead it
  fixated on a plausible-but-wrong file (`indexer.py`, which handles SQLite
  chunk deletion, not FAISS vector deletion) and re-read it four times
  without escalating to a different tool, also hitting the iteration limit.
  The same pattern showed up again testing `apply_patch` (Milestone 7): given
  a vague task, the agent had the correct file path and the exact buggy line
  in its own tool output by iteration 5, and still spent the next three
  iterations guessing at a nonexistent `src/` path instead of using what it
  had already been shown — it needed the task to name the file explicitly
  before it would actually call `apply_patch`. One run each isn't a rigorous
  comparison — see the evaluation framework (Milestone 10) for making this
  kind of claim properly — but it's a consistent, real result I'm not going
  to paper over: this looks like an attention/recall limitation under a long
  agentic loop, not a tooling bug (the tools returned exactly the right
  information every time), and it isn't obviously solved by model size alone.
  The provider abstraction makes trying a different model a one-line `.env`
  change either way.
- **`apply_patch` requires an exact, unique text match** — no fuzzy or
  whitespace-tolerant matching, and it can only edit existing files, not
  create new ones. If the model's `old_content` doesn't match the file
  byte-for-byte (e.g. subtly wrong indentation), the patch is rejected
  rather than guessed at.
- No background execution, polling, or cancellation — `POST /api/agent/run`
  blocks until the run finishes (it doesn't block *other* requests, since
  the loop is `async`, but there's no way to check progress or cancel a
  run in flight). That arrives with Milestone 9's streaming work.
- No re-planning — the initial plan is fixed for the whole run, even if
  early observations contradict it. The agent still adapts moment-to-moment
  (each turn sees all prior observations), just not by rewriting the plan.
- `find_references` is lexical (word-boundary regex over indexed chunks),
  not a real cross-reference/call-graph index — it'll find usage sites but
  doesn't understand scoping, shadowing, or imports.
- No code-modification or command-execution tools yet (`apply_patch`,
  `run_command`, `run_tests`, `run_linter`, `run_formatter`) — those need the
  fuller sandboxing layer (Docker option, allowlisting, env isolation) that
  Milestones 7/8 build on top of today's minimal subprocess runner.
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
