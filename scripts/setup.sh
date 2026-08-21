#!/usr/bin/env bash
# One-time local setup: Python venv, dependencies, .env, and required Ollama models.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "==> Creating virtualenv (.venv) with $PYTHON_BIN"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate

echo "==> Installing backend dependencies"
pip install --upgrade pip >/dev/null
pip install -e ".[dev]"

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
fi

mkdir -p data

if ! command -v ollama >/dev/null 2>&1; then
  echo "WARNING: 'ollama' CLI not found. Install it from https://ollama.com before running the app."
  exit 0
fi

if ! curl -s -o /dev/null http://localhost:11434/api/tags; then
  echo "==> Starting Ollama server in the background"
  (ollama serve > /tmp/ollama_serve.log 2>&1 &)
  sleep 2
fi

LLM_MODEL="$(grep '^LLM_MODEL=' .env | cut -d= -f2)"
EMBEDDING_MODEL="$(grep '^EMBEDDING_MODEL=' .env | cut -d= -f2)"

for MODEL in "$LLM_MODEL" "$EMBEDDING_MODEL"; do
  if ! ollama list | awk '{print $1}' | grep -q "^${MODEL}$"; then
    echo "==> Pulling $MODEL (this may take a while / several GB)"
    ollama pull "$MODEL"
  else
    echo "==> $MODEL already present"
  fi
done

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "==> Building optional Docker sandbox image (local-ai-softeng-sandbox:latest)"
  docker build -f docker/sandbox.Dockerfile -t local-ai-softeng-sandbox:latest . >/dev/null
  echo "    Set SANDBOX_BACKEND=docker in .env to use it (default is 'subprocess', no Docker required)."
else
  echo "==> Docker not available — skipping sandbox image build. SANDBOX_BACKEND stays 'subprocess' (the default)."
fi

echo "==> Setup complete. Run ./scripts/start.sh to launch the API."
