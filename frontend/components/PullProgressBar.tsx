"use client";

import { formatBytes } from "@/lib/format";
import type { PullProgress } from "@/lib/types";

/** A download's live state. Ollama reports byte counts only while layer data
 * is moving; the surrounding phases ("pulling manifest", "verifying sha256
 * digest") have no percentage, so the bar falls back to an indeterminate
 * shimmer rather than sitting frozen at whatever the last layer reached. */
export function PullProgressBar({ progress, onDismiss }: { progress: PullProgress; onDismiss?: () => void }) {
  const failed = Boolean(progress.error);
  const succeeded = progress.done && !failed;
  const percent = progress.percent ?? (succeeded ? 100 : null);

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 text-xs">
        <span className={failed ? "text-danger" : succeeded ? "text-success" : "text-muted"}>
          {failed ? progress.error : succeeded ? "Installed" : progress.status || "Starting…"}
        </span>
        <div className="flex shrink-0 items-center gap-2">
          {progress.total != null && progress.completed != null && !succeeded && !failed && (
            <span className="font-mono text-muted-subtle">
              {formatBytes(progress.completed)} / {formatBytes(progress.total)}
            </span>
          )}
          {percent != null && !failed && (
            <span className="font-mono tabular-nums text-muted">{Math.round(percent)}%</span>
          )}
          {(succeeded || failed) && onDismiss && (
            <button
              onClick={onDismiss}
              aria-label="Dismiss download status"
              className="text-muted-subtle transition-colors hover:text-foreground"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-background-subtle">
        <div
          className={`h-full rounded-full transition-[width] duration-300 ${
            failed ? "bg-danger" : succeeded ? "bg-success" : percent != null ? "bar-active bg-accent" : "bar-active bg-accent/30"
          }`}
          style={{ width: percent != null ? `${percent}%` : "100%" }}
          role="progressbar"
          aria-valuenow={percent ?? undefined}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}
