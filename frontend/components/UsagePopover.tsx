"use client";

import { useState } from "react";
import type { ContextWindow, TokenUsage } from "@/lib/types";

/** Compact token counts: 305800 -> "305.8k". Raw digits at this magnitude are
 * hard to compare at a glance, which is the only thing this number is for. */
export function formatTokens(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(tokens >= 10_000_000 ? 0 : 1)}M`;
  if (tokens >= 1000) return `${(tokens / 1000).toFixed(tokens >= 100_000 ? 0 : 1)}k`;
  return String(tokens);
}

function Meter({ percent, tone }: { percent: number; tone: "accent" | "warning" | "danger" }) {
  const color = { accent: "bg-accent", warning: "bg-warning", danger: "bg-danger" }[tone];
  return (
    <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-border">
      <div
        className={`h-full rounded-full transition-[width] duration-500 ${color}`}
        style={{ width: `${Math.max(percent, percent > 0 ? 1.5 : 0)}%` }}
        role="progressbar"
        aria-valuenow={Math.round(percent)}
        aria-valuemin={0}
        aria-valuemax={100}
      />
    </div>
  );
}

function toneFor(percent: number): "accent" | "warning" | "danger" {
  if (percent >= 90) return "danger";
  if (percent >= 70) return "warning";
  return "accent";
}

interface Props {
  usage: TokenUsage;
  contextWindow: ContextWindow | null;
}

export function UsagePopover({ usage, contextWindow }: Props) {
  const [open, setOpen] = useState(false);
  const percent = contextWindow?.percent ?? null;

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((value) => !value)}
        aria-expanded={open}
        title="Token usage and context window"
        className="flex items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-[11px] transition-colors hover:border-border-strong"
      >
        <span className="text-muted">Context</span>
        <span className="font-mono tabular-nums">
          {percent != null ? `${Math.round(percent)}%` : formatTokens(contextWindow?.used_tokens ?? 0)}
        </span>
        {percent != null && (
          <span aria-hidden className="h-1 w-8 overflow-hidden rounded-full bg-border">
            <span
              className={`block h-full rounded-full ${
                { accent: "bg-accent", warning: "bg-warning", danger: "bg-danger" }[toneFor(percent)]
              }`}
              style={{ width: `${Math.max(percent, 2)}%` }}
            />
          </span>
        )}
      </button>

      {open && (
        <>
          <button
            aria-label="Close usage panel"
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-30 cursor-default"
          />
          <div // Opens downward: this lives in the header, so an upward panel would be
            // clipped off the top of the viewport.
            className="animate-enter absolute top-full right-0 z-40 mt-2 w-80 rounded-xl border border-border bg-surface-raised p-3.5 shadow-lg">
            <section>
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="text-xs font-medium">Context window</h3>
                <span className="font-mono text-[11px] tabular-nums text-muted">
                  {contextWindow?.limit_tokens
                    ? `${formatTokens(contextWindow.used_tokens)} / ${formatTokens(contextWindow.limit_tokens)}`
                    : `${formatTokens(contextWindow?.used_tokens ?? 0)} used`}
                  {percent != null && ` (${Math.round(percent)}%)`}
                </span>
              </div>
              {percent != null ? (
                <Meter percent={percent} tone={toneFor(percent)} />
              ) : (
                <p className="mt-1 text-[11px] text-muted-subtle">
                  This model&apos;s window size isn&apos;t reported by the backend or listed in the catalog, so
                  there&apos;s no percentage to show.
                </p>
              )}
              <p className="mt-1.5 text-[11px] leading-relaxed text-muted-subtle">
                The largest single prompt sent in this session, against{" "}
                <span className="font-mono">{contextWindow?.model ?? "the active model"}</span>. Totals across a
                session exceed the window many times over without any one request being near it.
              </p>
            </section>

            <div className="my-3 border-t border-border" />

            <section>
              <h3 className="mb-1.5 text-xs font-medium">Session token usage</h3>
              <dl className="space-y-1 text-[11px]">
                <Row label="Prompt" value={formatTokens(usage.prompt_tokens)} />
                <Row label="Completion" value={formatTokens(usage.completion_tokens)} />
                <Row label="Total" value={formatTokens(usage.total_tokens)} strong />
                <Row label="Model calls" value={String(usage.llm_calls)} />
              </dl>
            </section>

            <div className="my-3 border-t border-border" />

            <section className="flex items-baseline justify-between gap-2">
              <h3 className="text-xs font-medium">Cost</h3>
              <span className="font-mono text-[11px] text-success">$0.00 — local inference</span>
            </section>
            <p className="mt-1 text-[11px] text-muted-subtle">
              Every token above was generated on this machine. Nothing was billed and nothing left the device.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <dt className="text-muted">{label}</dt>
      <dd className={`font-mono tabular-nums ${strong ? "font-medium" : "text-muted"}`}>{value}</dd>
    </div>
  );
}
