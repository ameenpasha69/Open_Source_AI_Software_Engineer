# Backend API image.
#
# The docker CLI is installed because the agent's sandbox backend runs each
# command in a throwaway container. With SANDBOX_BACKEND=docker the compose
# file mounts the host's Docker socket, so this container talks to the host
# daemon and the sandboxes come up as siblings rather than children -- there is
# no docker-in-docker here.
#
# That socket is worth understanding rather than pasting: anything that can
# reach it can start a privileged container, so it is host-level access. The
# alternative is SANDBOX_BACKEND=subprocess, where commands run inside this
# container instead -- no socket, but then an agent-authored command shares a
# filesystem with the database and the auth token. The socket is the better
# trade for a deployment; the compose file makes it easy to drop.

FROM python:3.11-slim

# The client only, taken from the official image. Debian's docker.io package is
# not a substitute: on Debian 13 it installs the runtime and docker-init but no
# `docker` binary, so the sandbox fails at run time with "Docker is not
# installed or not on PATH" while dpkg reports the package as present.
# Nothing here needs a daemon -- it drives the host's over the mounted socket.
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer survives application edits.
COPY pyproject.toml ./
COPY backend ./backend
RUN pip install -e .

# The same tools docker/sandbox.Dockerfile bakes in. With SANDBOX_BACKEND=
# subprocess the agent's commands run in *this* container, so without them
# run_tests fails with "No module named pytest" -- reported as a test failure
# rather than a missing dependency, which sends you looking in the wrong place.
# Unused in docker mode, where the sandbox image supplies them; ~50 MB to make
# both backends work from one image.
RUN pip install --no-cache-dir pytest ruff black flake8 mypy

# Written at runtime; declared so a bind mount is not required for it to exist.
RUN mkdir -p /app/data

EXPOSE 8000

# --host 0.0.0.0 binds inside the container only. What is actually reachable is
# decided by the port mapping in the compose file.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
