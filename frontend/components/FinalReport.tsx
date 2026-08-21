"use client";

import type { AgentRunSummary, TestRunOut } from "@/lib/types";
import { StatusBadge } from "./StatusBadge";

const RISK_NOTES: Record<string, string> = {
  not_applicable: "No code was changed — this run was investigation-only.",
  unverified: "Code was changed but never tested. Treat this fix as unconfirmed.",
  failed: "The most recent test run failed. This fix does not work as-is.",
  partially_verified: "Only specific tests were run, not the full suite — untested code paths may still be affected.",
  verified: "The full test suite passed after the change. No further verification was possible beyond that.",
};

export function FinalReport({ run, testRuns }: { run: AgentRunSummary; testRuns: TestRunOut[] }) {
  const lastTest = testRuns.at(-1);

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-surface">
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">Final report</h3>
        <StatusBadge status={run.status} />
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-4 text-sm">
        <Section title="Summary">
          <p>{run.final_answer ?? "The agent did not reach a conclusion within the iteration budget."}</p>
        </Section>

        {run.root_cause && (
          <Section title="Root cause">
            <p>{run.root_cause}</p>
          </Section>
        )}

        <Section title="Changes made">
          {run.modified_files.length === 0 ? (
            <p className="text-muted">No files were modified.</p>
          ) : (
            <ul className="list-inside list-disc font-mono text-xs">
              {run.modified_files.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          )}
        </Section>

        <Section title="Tests">
          {testRuns.length === 0 ? (
            <p className="text-muted">No tests were run.</p>
          ) : (
            <p>
              {lastTest?.passed ? "Passed" : "Failed"} ({lastTest?.scope} run
              {lastTest?.total_tests != null ? `, ${lastTest.passed_tests}/${lastTest.total_tests} passed` : ""}) —{" "}
              {testRuns.length} test invocation{testRuns.length === 1 ? "" : "s"} total this run.
            </p>
          )}
        </Section>

        <Section title="Verification status">
          <div className="flex items-center gap-2">
            <StatusBadge status={run.verification_status} />
          </div>
          <p className="mt-1 text-muted">{RISK_NOTES[run.verification_status]}</p>
        </Section>

        <Section title="Remaining risks">
          <ul className="list-inside list-disc text-muted">
            {run.status === "max_iterations_reached" && (
              <li>The agent hit its iteration limit without finishing — its work may be incomplete.</li>
            )}
            {run.status === "cancelled" && <li>This run was cancelled before it could finish.</li>}
            {run.verification_status !== "verified" && run.modified_files.length > 0 && (
              <li>This fix has not been confirmed against the full test suite.</li>
            )}
            {run.modified_files.length === 0 && <li>No changes were made — the underlying issue is unresolved.</li>}
          </ul>
        </Section>

        <Section title="Files modified">
          <p className="text-muted">{run.modified_files.length}</p>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h4 className="mb-1 text-xs font-semibold text-muted uppercase tracking-wide">{title}</h4>
      {children}
    </div>
  );
}
