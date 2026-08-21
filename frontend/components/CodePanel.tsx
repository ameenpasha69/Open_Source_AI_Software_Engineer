"use client";

import type { ToolCallOut } from "@/lib/types";

interface Snippet {
  filePath: string;
  symbol: string | null;
  startLine: number;
  endLine: number;
  content: string;
  source: string;
}

function extractSnippets(toolCalls: ToolCallOut[]): Snippet[] {
  const snippets: Snippet[] = [];
  for (const call of toolCalls) {
    if (!call.success || !call.output) continue;
    const out = call.output;

    if (call.tool_name === "search_code" && Array.isArray(out.results)) {
      for (const r of out.results as Record<string, unknown>[]) {
        snippets.push({
          filePath: String(r.file_path),
          symbol: (r.symbol as string) ?? null,
          startLine: Number(r.start_line),
          endLine: Number(r.end_line),
          content: String(r.content),
          source: "search_code",
        });
      }
    } else if (call.tool_name === "find_symbol" && Array.isArray(out.matches)) {
      for (const m of out.matches as Record<string, unknown>[]) {
        snippets.push({
          filePath: String(m.file_path),
          symbol: (m.symbol as string) ?? null,
          startLine: Number(m.start_line),
          endLine: Number(m.end_line),
          content: String(m.content),
          source: "find_symbol",
        });
      }
    } else if ((call.tool_name === "read_file" || call.tool_name === "get_file_context") && out.content) {
      snippets.push({
        filePath: String(out.path),
        symbol: null,
        startLine: Number(out.start_line),
        endLine: Number(out.end_line),
        content: String(out.content),
        source: call.tool_name,
      });
    }
  }
  return snippets;
}

export function CodePanel({ toolCalls }: { toolCalls: ToolCallOut[] }) {
  const snippets = extractSnippets(toolCalls);

  return (
    <div className="flex h-full flex-col rounded-lg border border-border bg-surface">
      <div className="border-b border-border px-4 py-2">
        <h3 className="text-sm font-semibold text-muted uppercase tracking-wide">
          Retrieved code ({snippets.length})
        </h3>
      </div>
      <div className="flex-1 space-y-3 overflow-y-auto p-4">
        {snippets.length === 0 && <p className="text-xs text-muted">Nothing retrieved yet.</p>}
        {snippets.map((s, i) => (
          <div key={i} className="rounded-md border border-border">
            <div className="flex items-center justify-between border-b border-border bg-code-bg px-3 py-1.5 text-xs">
              <span className="font-mono text-foreground">
                {s.filePath}:{s.startLine}-{s.endLine}
                {s.symbol && <span className="text-muted"> · {s.symbol}</span>}
              </span>
              <span className="text-muted">{s.source}</span>
            </div>
            <pre className="overflow-x-auto p-3 text-xs leading-relaxed">
              <code className="font-mono">{s.content}</code>
            </pre>
          </div>
        ))}
      </div>
    </div>
  );
}
