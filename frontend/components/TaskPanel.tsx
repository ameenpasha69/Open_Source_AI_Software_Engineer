"use client";

import { useState } from "react";
import { ApiError } from "@/lib/api";

interface Props {
  disabled: boolean;
  running: boolean;
  onRun: (task: string) => Promise<void>;
  onCancel: () => Promise<void>;
}

export function TaskPanel({ disabled, running, onRun, onCancel }: Props) {
  const [task, setTask] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!task.trim() || disabled) return;
    setSubmitting(true);
    setError(null);
    try {
      await onRun(task.trim());
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to start the agent");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-4">
      <label className="text-sm font-semibold text-muted uppercase tracking-wide">Describe the issue</label>
      <textarea
        value={task}
        onChange={(e) => setTask(e.target.value)}
        disabled={disabled || running}
        placeholder="e.g. calculate_total(items) returns the wrong value for orders with multiple items…"
        rows={3}
        className="resize-none rounded-md border border-border bg-background px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-60"
      />
      <div className="flex items-center gap-2">
        {!running ? (
          <button
            type="submit"
            disabled={disabled || submitting || !task.trim()}
            className="rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-accent-foreground disabled:opacity-50"
          >
            {submitting ? "Starting…" : "Run agent"}
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void onCancel()}
            className="rounded-md border border-danger px-4 py-1.5 text-sm font-medium text-danger hover:bg-danger/10"
          >
            Cancel run
          </button>
        )}
        {disabled && <span className="text-xs text-muted">Select a repository first</span>}
        {error && <span className="text-xs text-danger">{error}</span>}
      </div>
    </form>
  );
}
