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
cd backend
uvicorn app.main:app --reload --port 8000
