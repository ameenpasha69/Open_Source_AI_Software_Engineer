# Deploying

A three-service stack — Ollama, the API, and the web UI — brought up with one
command. Everything below was run and verified rather than written from the
documentation; the caveats are the ones that actually bit.

## Quick start

```bash
cd deploy
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # put this in API_AUTH_TOKEN
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5-coder:7b
docker compose exec ollama ollama pull nomic-embed-text
```

Then open <http://localhost:3000>.

On Docker Desktop for Windows or macOS, add the override — see
[Why Windows and macOS differ](#why-windows-and-macos-differ):

```bash
docker compose -f docker-compose.yml -f docker-compose.windows.yml up -d --build
```

## What you need

- A host that can run a **4.7 GB model**. On CPU it works but takes minutes per
  request; a GPU takes seconds. Uncomment the `deploy.resources` block in
  `docker-compose.yml` and install the NVIDIA Container Toolkit to use one.
- ~8 GB disk for images and models.
- Docker with Compose v2.

## Things that will catch you out

**The token has to be set before you build.** `NEXT_PUBLIC_*` values are
compiled into the browser bundle, not read at runtime, so the frontend image is
built with `API_AUTH_TOKEN` baked in. Change the token and you must rebuild the
frontend, not just restart it:

```bash
docker compose up -d --build frontend
```

Leaving `API_AUTH_TOKEN` empty is not an option here. The backend would generate
its own on first start and the frontend, built with an empty value, would be
locked out of it.

It also means the token is readable by anyone who can load the page. It protects
the API from other processes and other machines, not from the person using it.

**`PUBLIC_API_URL` is what a browser must reach**, not a service name.
`http://backend:8000` resolves only inside the Compose network; the page runs on
someone's laptop. Deploying to a server means setting this to that server's
address, and `FRONTEND_ORIGIN` to match, or the browser blocks the calls.

**Ollama's port is deliberately not published.** Only the backend needs it, and
Ollama has no authentication of its own — publishing it would put an open
inference endpoint beside an authenticated API.

**Models live in a volume, not an image.** They are pulled once with the
commands above. An image carrying 5 GB of weights would have to be rebuilt to
change model.

## Why Windows and macOS differ

With `SANDBOX_BACKEND=docker`, the backend asks the **host** daemon to start
each sandbox with `-v <cwd>:/workspace`, and the host resolves that path in its
own filesystem. So the repositories must be at the *same absolute path* inside
the backend container as outside — which the base compose file does.

That only works on a Linux host. A Windows path (`D:\...`) cannot exist inside a
Linux container, and macOS paths are not what Docker's VM sees either.

The failure is quiet, which is what makes it worth knowing: a mismatched path
does not error. Docker creates an empty directory and mounts that, pytest
reports `collected 0 items`, and the run looks like a repository with no tests.

`docker-compose.windows.yml` sidesteps it by mounting at a plain `/repos` and
switching to `SANDBOX_BACKEND=subprocess`, which never leaves the container and
needs no translation. The trade: agent commands then run inside the backend
container, sharing a filesystem with the database and the auth token. Still a
boundary against the host, but one boundary instead of two. For a deployment
where that matters, use a Linux host and the base file.

## The socket mount

The base file mounts `/var/run/docker.sock` so the backend can start sandbox
containers as siblings on the host daemon. Anything that can reach that socket
can start a privileged container, so it is host-level access.

Drop the mount and set `SANDBOX_BACKEND=subprocess` if that trade is not one you
want. The Windows override already does.

## Verified

Run against this stack, not inferred from the configuration:

- `401` without a token, `200` with one, on every route including `/api/health`
- The backend reaches Ollama across the Compose network
- Indexing a mounted repository: 3 files, 7 chunks, 7 embedded, no error
- `run_tests` in the containerized backend: 3 tests, 2 passed, and it names
  `tests/test_stats.py::test_median_even_length` as the failure
- The UI loads, authenticates, and lists the indexed repository

Two bugs found by doing this rather than assuming: Debian 13's `docker.io`
package installs no `docker` binary (the CLI now comes from the `docker:cli`
image), and the backend image needed the same test tooling the sandbox image
bakes in, or `subprocess` mode fails with `No module named pytest`.
