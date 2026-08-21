# Base image for the Docker sandbox execution backend (SANDBOX_BACKEND=docker).
#
# Sandboxed containers run with --network none, so packages can't be
# installed at container-start time — every tool the allowlisted commands
# need (pytest, ruff, black, flake8, mypy) has to be baked in ahead of time.
# Node/Go/Rust/JVM toolchains are intentionally not included: this project's
# real language support is Python-first (see README limitations), and the
# subprocess backend remains available for anything this image can't run.
FROM python:3.11-slim

RUN pip install --no-cache-dir pytest ruff black flake8 mypy
