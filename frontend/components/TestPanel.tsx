"use client";

import { useState } from "react";
import type { TestRunOut } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

function TestRunCard({ run }: { run: TestRunOut }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="rounded-md border border-border">
      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm"
      >
        <div className="flex items-center gap-2">
          <StatusBadge status={run.passed ? "passed" : "failed"} />
          <span className="font-mono text-xs text-muted">{run.command}</span>
          <span className="rounded border border-border px-1.5 py-0.5 text-[10px] text-muted uppercase">
            {run.scope}
          </span>
        </div>
        <span className="text-xs text-muted">{run.duration_seconds.toFixed(2)}s</span>
      </button>

      <div className="flex flex-wrap gap-x-4 gap-y-1 border-t border-border px-3 py-1.5 text-xs text-muted">
        {run.total_tests !== null && (
          <span>
            {run.passed_tests}/{run.total_tests} passed
          </span>
        )}
        {!run.passed && <span className="text-warning">category: {run.failure_category}</span>}
        {run.failed_tests.length > 0 && <span>{run.failed_tests.length} failing</span>}
      </div>

      {run.failed_tests.length > 0 && (
        <ul className="border-t border-border px-3 py-1.5 font-mono text-xs text-danger">
          {run.failed_tests.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
      )}

      {expanded && (
        <div className="border-t border-border p-3">
          {run.stdout && (
            <>
              <div className="mb-1 text-[10px] font-medium text-muted uppercase">stdout</div>
              <pre className="mb-2 max-h-64 overflow-auto rounded bg-code-bg p-2 text-xs">{run.stdout}</pre>
            </>
          )}
          {run.stderr && (
            <>
              <div className="mb-1 text-[10px] font-medium text-muted uppercase">stderr</div>
              <pre className="max-h-64 overflow-auto rounded bg-code-bg p-2 text-xs text-danger">{run.stderr}</pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export function TestPanel({ testRuns }: { testRuns: TestRunOut[] }) {
  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">
          Tests ({testRuns.length} run{testRuns.length === 1 ? "" : "s"})
        </h3>
      </div>
      <div className="flex-1 space-y-2 overflow-y-auto p-4">
        {testRuns.length === 0 && <p className="text-xs text-muted">No tests run yet.</p>}
        {testRuns.map((run, i) => (
          <TestRunCard key={i} run={run} />
        ))}
      </div>
    </div>
  );
}
