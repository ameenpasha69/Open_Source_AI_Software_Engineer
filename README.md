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
cd backend && PYTHONPATH=. python -m pytest tests/ -v
```

Tests use a fake `LLMProvider` (`backend/tests/conftest.py`) for fast,
deterministic runs, plus targeted tests against `OllamaProvider`'s error
handling (connection errors, missing models, timeouts) using mocked HTTP
responses — no live Ollama server required to run the suite.

## Project structure

```text
backend/
  app/
    config/      # environment-driven settings
    llm/         # LLMProvider abstraction + Ollama backend
    api/routes/  # FastAPI routers
    schemas/     # Pydantic request/response models
    agents/ embeddings/ retrieval/ tools/ execution/
    verification/ evaluation/ database/ models/   # scaffolded, empty — future milestones
  tests/
scripts/         # setup.sh, start.sh
data/            # local vector index, SQLite DB (gitignored)
evals/           # benchmark tasks (future milestone)
docs/
```

## Limitations (current milestone)

- Only Ollama is implemented as a provider; the interface supports others but
  none are built yet.
- No agent, retrieval, or tool system exists yet — this milestone is
  infrastructure only.
- Local model quality varies by hardware and chosen model size.
