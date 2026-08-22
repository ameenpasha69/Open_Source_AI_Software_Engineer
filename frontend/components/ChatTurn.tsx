"use client";

import { useEffect, useState } from "react";
import { AgentTimeline } from "@/components/AgentTimeline";
import { CodePanel } from "@/components/CodePanel";
import { DiffPanel } from "@/components/DiffPanel";
import { FinalReport } from "@/components/FinalReport";
import { StatusBadge } from "@/components/StatusBadge";
import { TestPanel } from "@/components/TestPanel";
import { formatTokens } from "@/components/UsagePopover";
import { api } from "@/lib/api";
import type { AgentDiffResponse, AgentEventOut, ChatMessage, TestRunOut, ToolCallOut } from "@/lib/types";

type DetailTab = "timeline" | "code" | "diff" | "tests" | "report";

const TABS: { value: DetailTab; label: string }[] = [
  { value: "timeline", label: "Timeline" },
  { value: "code", label: "Code" },
  { value: "diff", label: "Diff" },
  { value: "tests", label: "Tests" },
  { value: "report", label: "Report" },
];

export function UserTurn({ message }: { message: ChatMessage }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-sm whitespace-pre-wrap">
        {message.content}
      </div>
    </div>
  );
}

/** An assistant turn is the answer plus everything behind it. The answer is
 * always visible; the evidence (timeline, retrieved code, diff, tests) is one
 * click away rather than gone — a claim you can't check is worth less. */
export function AssistantTurn({
  message,
  liveEvents,
  live,
}: {
  message: ChatMessage;
  liveEvents?: AgentEventOut[];
  live?: boolean;
}) {
  const [expanded, setExpanded] = useState(Boolean(live));
  const [tab, setTab] = useState<DetailTab>("timeline");
  const [events, setEvents] = useState<AgentEventOut[]>([]);
  const [toolCalls, setToolCalls] = useState<ToolCallOut[]>([]);
  const [diff, setDiff] = useState<AgentDiffResponse | null>(null);
  const [testRuns, setTestRuns] = useState<TestRunOut[]>([]);
  const runId = message.run_id;

  // Detail is fetched only when someone opens it — a long transcript would
  // otherwise fire four requests per turn on load for panels nobody looked at.
  useEffect(() => {
    if (!expanded || !runId || live) return;
    let cancelled = false;
    void Promise.all([
      api.getAgentEvents(runId),
      api.getAgentToolCalls(runId),
      api.getAgentDiff(runId),
      api.getAgentTests(runId),
    ]).then(([e, t, d, r]) => {
      if (cancelled) return;
      setEvents(e);
      setToolCalls(t);
      setDiff(d);
      setTestRuns(r);
    });
    return () => {
      cancelled = true;
    };
  }, [expanded, runId, live]);

  const shownEvents = live ? (liveEvents ?? []) : events;
  const run = message.run;
  // A turn still in flight has no run_id yet (nothing's been written for it),
  // but its evidence already exists as the live SSE stream — detail must not
  // be gated on the id a REST fetch would need, or a running turn can never
  // show its own timeline.
  const hasDetail = live || runId != null;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-start gap-2.5">
        <span
          aria-hidden
          className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-md bg-surface-hover text-[11px] font-bold"
        >
          ⌥
        </span>
        <div className="min-w-0 flex-1">
          {message.content ? (
            <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            <p className="text-sm text-muted italic">{latestThought(shownEvents) ?? "Working…"}</p>
          )}

          {run?.root_cause && (
            <p className="mt-2 rounded-lg border border-border bg-surface px-3 py-2 text-xs leading-relaxed text-muted">
              <span className="font-medium text-foreground">Root cause:</span> {run.root_cause}
            </p>
          )}

          <div className="mt-2 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[11px] text-muted-subtle">
            {run && <StatusBadge status={run.status} />}
            {run && run.modified_files.length > 0 && (
              <span>
                {run.modified_files.length} file{run.modified_files.length === 1 ? "" : "s"} changed
              </span>
            )}
            {run && run.iteration_count > 0 && <span>{run.iteration_count} iterations</span>}
            {message.usage && message.usage.total_tokens > 0 && (
              <span title={`${message.usage.prompt_tokens} prompt + ${message.usage.completion_tokens} completion tokens across ${message.usage.llm_calls} model calls`}>
                {formatTokens(message.usage.total_tokens)} tokens
              </span>
            )}
            {run?.model && <span className="font-mono">{run.model}</span>}
            {hasDetail && (
              <button
                onClick={() => setExpanded((value) => !value)}
                className="text-accent transition-colors hover:underline"
              >
                {expanded ? "Hide detail" : "Show detail"}
              </button>
            )}
          </div>
        </div>
      </div>

      {expanded && hasDetail && (
        <div className="ml-8 flex flex-col overflow-hidden rounded-xl border border-border bg-surface">
          <nav className="flex shrink-0 gap-1 border-b border-border px-2" aria-label="Run detail">
            {TABS.map(({ value, label }) => (
              <button
                key={value}
                aria-current={tab === value ? "page" : undefined}
                onClick={() => setTab(value)}
                className={`-mb-px border-b-2 px-2.5 py-1.5 text-xs transition-colors ${
                  tab === value
                    ? "border-accent font-medium text-accent"
                    : "border-transparent text-muted hover:text-foreground"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <div className="h-80">
            {tab === "timeline" && <AgentTimeline events={shownEvents} live={Boolean(live)} />}
            {tab === "code" && <CodePanel toolCalls={toolCalls} />}
            {tab === "diff" && <DiffPanel diff={diff} />}
            {tab === "tests" && <TestPanel testRuns={testRuns} />}
            {tab === "report" &&
              (run ? (
                <FinalReport run={run} testRuns={testRuns} />
              ) : (
                <div className="flex h-full items-center justify-center rounded-xl border border-border bg-surface text-sm text-muted">
                  The report is available once this turn finishes.
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** The most recent "thought" event, so the headline reads as what the agent
 * is actually doing right now instead of a static "Working…" for the entire
 * run — the timeline below already shows every step, this is just the
 * one-line version above it. */
function latestThought(events: AgentEventOut[]): string | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const event = events[i];
    if (event.event_type === "thought") {
      const thought = String((event.payload as Record<string, unknown>).thought ?? "");
      return thought || null;
    }
  }
  return null;
}
