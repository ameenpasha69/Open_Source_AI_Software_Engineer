"use client";

import { useEffect, useRef } from "react";
import type { AgentEventOut } from "@/lib/types";

function summarizeInput(input: unknown): string {
  if (!input || typeof input !== "object") return "";
  const entries = Object.entries(input as Record<string, unknown>);
  if (entries.length === 0) return "";
  return entries.map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
}

function TimelineRow({ event }: { event: AgentEventOut }) {
  const p = event.payload as Record<string, unknown>;

  switch (event.event_type) {
    case "plan_created": {
      const steps = (p.plan as string[]) ?? [];
      return (
        <div className="flex flex-col gap-1">
          <Row icon="✓" tone="success" text={`Plan created (${steps.length} steps)`} />
          <ol className="ml-6 list-decimal space-y-0.5 text-xs text-muted">
            {steps.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ol>
        </div>
      );
    }
    case "thought":
      return <Row icon="…" tone="muted" text={String(p.thought ?? "")} italic />;
    case "tool_called":
      return (
        <Row
          icon="→"
          tone="accent"
          text={`calling ${p.tool}(${summarizeInput(p.input)})`}
          mono
        />
      );
    case "tool_completed":
      return (
        <Row
          icon={p.success ? "✓" : "✗"}
          tone={p.success ? "success" : "danger"}
          text={`${p.tool} ${p.success ? "succeeded" : "failed"}`}
          mono
        />
      );
    case "invalid_decision":
      return <Row icon="⚠" tone="warning" text="Model response was invalid — retrying" />;
    case "finished":
      return <Row icon="✓" tone="success" text={`Finished: ${p.answer}`} bold />;
    case "iteration_started":
      return (
        <div className="mt-2 border-t border-border pt-2 text-[11px] font-medium tracking-wide text-muted uppercase">
          Iteration {event.iteration}
        </div>
      );
    default:
      return <Row icon="•" tone="muted" text={event.event_type} />;
  }
}

function Row({
  icon,
  tone,
  text,
  italic,
  mono,
  bold,
}: {
  icon: string;
  tone: "success" | "danger" | "warning" | "accent" | "muted";
  text: string;
  italic?: boolean;
  mono?: boolean;
  bold?: boolean;
}) {
  const toneClass = {
    success: "text-success",
    danger: "text-danger",
    warning: "text-warning",
    accent: "text-accent",
    muted: "text-muted",
  }[tone];
  return (
    <div className="flex items-start gap-2 text-sm">
      <span className={`shrink-0 ${toneClass}`}>{icon}</span>
      <span
        className={`${italic ? "italic" : ""} ${mono ? "font-mono text-xs" : ""} ${bold ? "font-semibold" : ""} text-foreground`}
      >
        {text}
      </span>
    </div>
  );
}

export function AgentTimeline({ events, live }: { events: AgentEventOut[]; live: boolean }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [events.length]);

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">Agent timeline</h3>
        {live && (
          <span className="flex items-center gap-1.5 text-xs text-accent">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" /> live
          </span>
        )}
      </div>
      <div className="flex-1 space-y-1.5 overflow-y-auto p-4">
        {events.length === 0 && <p className="text-xs text-muted">No activity yet.</p>}
        {events.map((event, i) => (
          <TimelineRow key={i} event={event} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
