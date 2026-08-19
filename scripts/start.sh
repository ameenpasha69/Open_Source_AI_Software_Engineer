#!/usr/bin/env bash
# Start the backend API (and Ollama, if it isn't already running).
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v ollama >/dev/null 2>&1 && ! curl -s -o /dev/null http://localhost:11434/api/tags; then
  echo "==> Starting Ollama server in the background"
  (ollama serve > /tmp/ollama_serve.log 2>&1 &)
  sleep 2
fi

source .venv/bin/activate
# Run from the project root (not backend/) so relative paths like DATA_DIR
# resolve consistently with what's documented in the README. --app-dir tells
# uvicorn where to find the `app` package without changing the process cwd.
uvicorn app.main:app --reload --port 8000 --app-dir backend
