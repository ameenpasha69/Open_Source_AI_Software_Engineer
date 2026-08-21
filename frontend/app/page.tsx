"use client";

import { useCallback, useEffect, useState } from "react";
import { AgentTimeline } from "@/components/AgentTimeline";
import { CodePanel } from "@/components/CodePanel";
import { DiffPanel } from "@/components/DiffPanel";
import { FinalReport } from "@/components/FinalReport";
import { RepositoryPanel } from "@/components/RepositoryPanel";
import { StatusBadge } from "@/components/StatusBadge";
import { TaskPanel } from "@/components/TaskPanel";
import { TestPanel } from "@/components/TestPanel";
import { api } from "@/lib/api";
import type { AgentDiffResponse, AgentRunSummary, RepositorySummary, TestRunOut, ToolCallOut } from "@/lib/types";
import { useAgentStream } from "@/lib/useAgentStream";

type Tab = "timeline" | "code" | "diff" | "tests" | "report";

export default function Home() {
  const [repositories, setRepositories] = useState<RepositorySummary[]>([]);
  const [selectedRepoId, setSelectedRepoId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<AgentRunSummary | null>(null);
  const [toolCalls, setToolCalls] = useState<ToolCallOut[]>([]);
  const [diff, setDiff] = useState<AgentDiffResponse | null>(null);
  const [testRuns, setTestRuns] = useState<TestRunOut[]>([]);
  const [tab, setTab] = useState<Tab>("timeline");
  const [loadError, setLoadError] = useState<string | null>(null);

  const { events, closed } = useAgentStream(runId);
  const running = run?.status === "running";

  const refreshRepositories = useCallback(async () => {
    try {
      const repos = await api.listRepositories();
      setRepositories(repos);
    } catch {
      setLoadError("Could not reach the API — is the backend running on :8000?");
    }
  }, []);

  useEffect(() => {
    // Standard fetch-on-mount — react-hooks/set-state-in-effect flags this
    // pattern generally, but there's no external-store equivalent for "load
    // the repository list once when the page opens" that isn't more complex
    // for no real benefit here.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refreshRepositories();
  }, [refreshRepositories]);

  // Refresh the detail panels whenever a new event lands (tool results, or
  // the run finishing) — keeps Code/Diff/Tests reasonably live without
  // polling on a timer.
  useEffect(() => {
    if (!runId) return;
    const latest = events.at(-1);
    if (!latest) return;
    if (latest.event_type === "tool_completed" || latest.event_type === "finished") {
      void Promise.all([
        api.getAgentDiff(runId).then(setDiff),
        api.getAgentTests(runId).then(setTestRuns),
        api.getAgentToolCalls(runId).then(setToolCalls),
      ]);
    }
  }, [events, runId]);

  useEffect(() => {
    if (!runId || !closed) return;
    void api.getAgentRun(runId).then(setRun);
  }, [runId, closed]);

  async function handleRun(task: string) {
    if (!selectedRepoId) return;
    setDiff(null);
    setTestRuns([]);
    setToolCalls([]);
    setTab("timeline");
    const summary = await api.runAgent(selectedRepoId, task);
    setRun(summary);
    setRunId(summary.id);
  }

  async function handleCancel() {
    if (!runId) return;
    const summary = await api.cancelAgentRun(runId);
    setRun(summary);
  }

  const selectedRepo = repositories.find((r) => r.id === selectedRepoId) ?? null;

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-border bg-surface px-6 py-3">
        <div className="flex items-center gap-2">
          <h1 className="text-base font-semibold">Local AI Software Engineer</h1>
          <span className="text-xs text-muted">100% local — no cloud APIs</span>
        </div>
        {run && <StatusBadge status={run.status} />}
      </header>

      {loadError && <div className="bg-danger/10 px-6 py-2 text-sm text-danger">{loadError}</div>}

      <div className="flex flex-1 overflow-hidden">
        <aside className="w-72 shrink-0 overflow-hidden border-r border-border bg-surface">
          <RepositoryPanel
            repositories={repositories}
            selectedId={selectedRepoId}
            onSelect={setSelectedRepoId}
            onIndexed={refreshRepositories}
          />
        </aside>

        <main className="flex flex-1 flex-col gap-4 overflow-hidden p-4">
          {selectedRepo && (
            <div className="rounded-lg border border-border bg-surface px-4 py-2 text-xs text-muted">
              <span className="font-medium text-foreground">{selectedRepo.name}</span> · {selectedRepo.path} ·{" "}
              {selectedRepo.chunk_count} chunks indexed
              {!selectedRepo.test_command && (
                <span className="ml-2 text-warning">no test command configured</span>
              )}
            </div>
          )}

          <TaskPanel disabled={!selectedRepoId} running={running} onRun={handleRun} onCancel={handleCancel} />

          {run && (
            <div className="flex flex-1 flex-col overflow-hidden">
              <nav className="mb-3 flex gap-1 border-b border-border">
                {(["timeline", "code", "diff", "tests", "report"] as Tab[]).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTab(t)}
                    className={`px-3 py-1.5 text-sm capitalize ${
                      tab === t
                        ? "border-b-2 border-accent font-medium text-accent"
                        : "text-muted hover:text-foreground"
                    }`}
                  >
                    {t}
                  </button>
                ))}
              </nav>

              <div className="flex-1 overflow-hidden">
                {tab === "timeline" && <AgentTimeline events={events} live={running} />}
                {tab === "code" && <CodePanel toolCalls={toolCalls} />}
                {tab === "diff" && <DiffPanel diff={diff} />}
                {tab === "tests" && <TestPanel testRuns={testRuns} />}
                {tab === "report" &&
                  (running ? (
                    <div className="flex h-full items-center justify-center text-sm text-muted">
                      The report is available once the run finishes.
                    </div>
                  ) : (
                    <FinalReport run={run} testRuns={testRuns} />
                  ))}
              </div>
            </div>
          )}

          {!run && (
            <div className="flex flex-1 items-center justify-center text-sm text-muted">
              {selectedRepoId ? "Describe an issue above to start the agent." : "Select or index a repository to begin."}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
