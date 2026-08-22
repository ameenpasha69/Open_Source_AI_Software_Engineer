"use client";

import type { AgentDiffResponse } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

function DiffLine({ line }: { line: string }) {
  let tone = "text-foreground";
  if (line.startsWith("+") && !line.startsWith("+++")) tone = "bg-diff-add text-success";
  else if (line.startsWith("-") && !line.startsWith("---")) tone = "bg-diff-remove text-danger";
  else if (line.startsWith("@@")) tone = "text-accent";
  else if (line.startsWith("---") || line.startsWith("+++")) tone = "text-muted";
  return <div className={`whitespace-pre px-2 ${tone}`}>{line || " "}</div>;
}

export function DiffPanel({ diff }: { diff: AgentDiffResponse | null }) {
  return (
    <div className="flex h-full flex-col overflow-hidden rounded-xl border border-border bg-surface">
      <div className="flex shrink-0 items-center justify-between border-b border-border px-4 py-2.5">
        <h3 className="text-xs font-semibold tracking-wide text-muted uppercase">
          Diff {diff && `(${diff.modified_files.length} file${diff.modified_files.length === 1 ? "" : "s"})`}
        </h3>
        {diff && <StatusBadge status={diff.verification_status} />}
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {(!diff || diff.modified_files.length === 0) && (
          <p className="text-xs text-muted">No files modified yet.</p>
        )}
        {diff?.modified_files.map((file) => (
          <div key={file.path} className="rounded-lg border border-border">
            <div className="flex items-center justify-between border-b border-border bg-code-bg px-3 py-1.5 text-xs">
              <span className="font-mono text-foreground">{file.path}</span>
              <span className="font-mono text-muted">
                <span className="text-success">+{file.lines_added}</span>{" "}
                <span className="text-danger">-{file.lines_removed}</span>
              </span>
            </div>
            <div className="overflow-x-auto py-2 font-mono text-xs leading-relaxed">
              {file.diff.split("\n").map((line, i) => (
                <DiffLine key={i} line={line} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
