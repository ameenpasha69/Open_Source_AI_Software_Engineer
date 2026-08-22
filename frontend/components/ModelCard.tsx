"use client";

import { displayModelName, formatBytes, formatRelativeTime, formatTokens } from "@/lib/format";
import type { CatalogEntryOut, InstalledModel, PullProgress } from "@/lib/types";
import { PullProgressBar } from "./PullProgressBar";

function Tag({ children, tone = "muted" }: { children: React.ReactNode; tone?: "muted" | "accent" | "success" }) {
  const styles = {
    muted: "border-border text-muted",
    accent: "border-accent/40 bg-accent-soft text-accent",
    success: "border-success/40 bg-success-soft text-success",
  }[tone];
  return (
    <span className={`rounded-md border px-1.5 py-0.5 text-[11px] leading-4 font-medium ${styles}`}>
      {children}
    </span>
  );
}

function RoleTag({ role }: { role: "chat" | "embedding" }) {
  return <Tag>{role === "chat" ? "Reasoning" : "Embedding"}</Tag>;
}

export function InstalledModelCard({
  model,
  onActivate,
  onRemove,
  busy,
}: {
  model: InstalledModel;
  onActivate: () => void;
  onRemove: () => void;
  busy: boolean;
}) {
  return (
    <div
      className={`flex flex-col gap-3 rounded-xl border p-4 transition-colors ${
        model.active ? "border-accent/50 bg-accent-soft/40" : "border-border bg-surface hover:border-border-strong"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="truncate font-mono text-sm font-medium">{displayModelName(model.name)}</h3>
            {model.active && <Tag tone="accent">In use</Tag>}
          </div>
          {model.catalog && <p className="mt-1 text-xs text-muted">{model.catalog.description}</p>}
        </div>
        {model.loaded && (
          <span
            title="Loaded in memory — responds without a cold start"
            className="flex shrink-0 items-center gap-1.5 text-[11px] text-success"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-success" />
            warm
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <RoleTag role={model.role} />
        {model.parameter_size && <Tag>{model.parameter_size}</Tag>}
        {model.quantization && <Tag>{model.quantization}</Tag>}
        <Tag>{formatBytes(model.size_bytes)}</Tag>
        {model.catalog?.context_window && <Tag>{formatTokens(model.catalog.context_window)} ctx</Tag>}
        {model.catalog?.dimensions && <Tag>{model.catalog.dimensions}-dim</Tag>}
      </div>

      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-[11px] text-muted-subtle">
          {model.modified_at ? `Pulled ${formatRelativeTime(model.modified_at)}` : model.family}
        </span>
        <div className="flex shrink-0 gap-1.5">
          {!model.active && (
            <button
              onClick={onActivate}
              disabled={busy}
              className="rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-accent-foreground transition-colors hover:bg-accent-hover disabled:opacity-50"
            >
              Use this model
            </button>
          )}
          <button
            onClick={onRemove}
            disabled={busy || model.active}
            title={model.active ? "Switch to another model before removing this one" : "Remove from disk"}
            className="rounded-lg border border-border px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:border-danger/50 hover:text-danger disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-muted"
          >
            Remove
          </button>
        </div>
      </div>
    </div>
  );
}

export function CatalogModelCard({
  entry,
  progress,
  onPull,
  onDismissProgress,
}: {
  entry: CatalogEntryOut;
  progress: PullProgress | undefined;
  onPull: () => void;
  onDismissProgress: () => void;
}) {
  const downloading = Boolean(progress) && !progress!.done;

  return (
    <div
      className={`flex flex-col gap-3 rounded-xl border p-4 transition-colors ${
        entry.installed ? "border-border bg-surface" : "border-border bg-surface hover:border-border-strong"
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="text-sm font-semibold">{entry.label}</h3>
            {entry.recommended && <Tag tone="accent">Recommended</Tag>}
            {entry.active && <Tag tone="success">In use</Tag>}
          </div>
          <p className="mt-0.5 font-mono text-[11px] text-muted-subtle">{entry.name}</p>
        </div>
        <span className="shrink-0 text-[11px] text-muted-subtle">{entry.publisher}</span>
      </div>

      <p className="text-xs leading-relaxed text-muted">{entry.description}</p>

      <div className="flex flex-wrap items-center gap-1.5">
        <RoleTag role={entry.role} />
        <Tag>{entry.parameter_size}</Tag>
        <Tag>~{formatBytes(entry.approx_size_bytes)}</Tag>
        {entry.context_window && <Tag>{formatTokens(entry.context_window)} ctx</Tag>}
        {entry.dimensions && <Tag>{entry.dimensions}-dim</Tag>}
        <Tag>{entry.min_ram_gb} GB RAM</Tag>
      </div>

      {entry.strengths.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {entry.strengths.map((strength) => (
            <span key={strength} className="text-[11px] text-muted-subtle">
              {strength}
            </span>
          ))}
        </div>
      )}

      {progress ? (
        <PullProgressBar progress={progress} onDismiss={onDismissProgress} />
      ) : entry.installed ? (
        <div className="flex items-center gap-1.5 text-xs text-success">
          <span aria-hidden>✓</span> Installed
        </div>
      ) : (
        <button
          onClick={onPull}
          disabled={downloading}
          className="w-full rounded-lg border border-accent/40 bg-accent-soft px-3 py-1.5 text-xs font-medium text-accent transition-colors hover:bg-accent hover:text-accent-foreground disabled:opacity-50"
        >
          Download · ~{formatBytes(entry.approx_size_bytes)}
        </button>
      )}
    </div>
  );
}
