# Frontend

Next.js UI for [Local AI Software Engineer](../README.md) — a single-page
dashboard: pick or index a repository, describe an issue, and watch the
agent investigate, patch, and test in real time.

## Running

Requires the backend running on `http://localhost:8000` (see the [root
README](../README.md) — `../scripts/start.sh` from the project root).

```bash
npm install
npm run dev
```

Open http://localhost:3000. To point at a backend on a different host/port,
copy `.env.local.example` to `.env.local` and set `NEXT_PUBLIC_API_BASE_URL`.

## Structure

```text
app/page.tsx           # the whole app — repository selection, task input,
                        # tabs (Timeline/Code/Diff/Tests/Report)
lib/api.ts              # typed fetch wrappers over the backend API
lib/types.ts             # TypeScript types mirroring the backend's Pydantic schemas
lib/useAgentStream.ts    # SSE hook for GET /api/agent/{run_id}/stream
components/              # one component per panel
```

**Live updates**: `useAgentStream` opens an `EventSource` against the
backend's SSE endpoint. Regular events arrive as default (unnamed) SSE
messages — `event_type` travels inside the JSON body, so one `onmessage`
handler covers every event type without registering a listener per type
ahead of time. The stream's only named event, `run_completed`, closes the
connection. The Code/Diff/Tests panels refetch their (richer) detail
endpoints whenever a `tool_completed` or `finished` event arrives, rather
than trying to reconstruct that detail from the stream's deliberately
compact payloads.
