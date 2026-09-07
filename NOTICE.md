# Notice

## Origin

This repository is a copy of
[habeebsait/Open_Source_AI_Software_Engineer](https://github.com/habeebsait/Open_Source_AI_Software_Engineer),
written by **MD Habeeb Sait** (`habeebsait24@gmail.com`), used here with his
permission.

The original 18 commits are preserved with their authorship intact, so `git
log` remains an accurate record of who wrote what. This copy is maintained by
[@ameenpasha69](https://github.com/ameenpasha69).

## Licensing status

The upstream project publishes **no LICENSE file**, which under default
copyright means the author retains all rights. The permission described above
was given to @ameenpasha69 for this copy; it is not a public license, so it
does not extend to anyone else reading this.

If you want to use, modify, or redistribute this code, ask MD Habeeb Sait. The
durable fix is a LICENSE file in the upstream repository — until that exists,
every copy of this project needs its own conversation.

## Changes made in this copy

### Embedding requests are batched

`OllamaEmbeddingProvider.embed_documents` sent every chunk to Ollama's
`/api/embed` endpoint in a single request. That request is served by a model
runner subprocess, and a large enough batch kills it. Because the parent server
survives, the failure came back as a bare `HTTP 400` whose body was a
connection error to the runner's own port — which reads like a malformed
request rather than a crashed child process.

Indexing this repository (1313 chunks) reproduced it every time on a 4 GiB
GTX 1650. Bisected: 200 chunks succeed, 400 kill the runner. Raising
`LLM_REQUEST_TIMEOUT_SECONDS` does not help; it only moves the failure from a
120 s timeout to the crash underneath it.

Batch size is now bounded (`EMBEDDING_BATCH_SIZE`, default 64), so the peak the
runner absorbs no longer scales with repository size. After the fix: 1319
chunks embedded in 168 s, and semantic search over the result returns the
expected files.

### CORS accepts a list of origins

`frontend_origin` took exactly one value, so the API could serve the UI from
one address or another but never both. It now accepts a comma-separated list.
With a single origin configured the behaviour is unchanged.

### Windows verification

The test suite is green apart from 23 assertions that hardcode POSIX path
separators (`app/main.py` vs `app\main.py`) or rely on symlink and subprocess
behaviour that differs on Windows. These are test-side assumptions, not
defects in the application code, and they are not fixed here.

## Upstream contribution

The embedding-batching fix applies cleanly to the upstream project and is
offered back from
[ameenpasha69/Open_Source_AI_Software_Engineer](https://github.com/ameenpasha69/Open_Source_AI_Software_Engineer),
a fork kept in step with upstream for that purpose.
