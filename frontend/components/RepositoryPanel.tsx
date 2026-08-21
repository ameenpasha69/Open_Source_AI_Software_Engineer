"use client";

import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { RepositorySummary } from "@/lib/types";

interface Props {
  repositories: RepositorySummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onIndexed: () => void;
}

export function RepositoryPanel({ repositories, selectedId, onSelect, onIndexed }: Props) {
  const [path, setPath] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);

  async function handleIndex(e: React.FormEvent) {
    e.preventDefault();
    if (!path.trim()) return;
    setIndexing(true);
    setError(null);
    setLastResult(null);
    try {
      const result = await api.indexRepository(path.trim());
      setLastResult(
        `Indexed ${result.files_indexed} file(s), ${result.chunks_created} chunk(s) in ${result.duration_seconds}s`,
      );
      setPath("");
      onIndexed();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to index repository");
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <h2 className="px-4 pt-4 text-sm font-semibold tracking-wide text-muted uppercase">Repository</h2>

      <form onSubmit={handleIndex} className="flex flex-col gap-2 px-4 pt-3">
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="/absolute/path/to/a/repo"
          className="rounded-md border border-border bg-surface px-3 py-1.5 text-sm outline-none focus:border-accent"
        />
        <button
          type="submit"
          disabled={indexing || !path.trim()}
          className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-foreground disabled:opacity-50"
        >
          {indexing ? "Indexing…" : "Index repository"}
        </button>
        {error && <p className="text-xs text-danger">{error}</p>}
        {lastResult && <p className="text-xs text-success">{lastResult}</p>}
      </form>

      <div className="mt-4 flex-1 overflow-y-auto px-2 pb-4">
        {repositories.length === 0 && (
          <p className="px-2 text-xs text-muted">No repositories indexed yet.</p>
        )}
        <ul className="flex flex-col gap-1">
          {repositories.map((repo) => (
            <li key={repo.id}>
              <button
                onClick={() => onSelect(repo.id)}
                className={`w-full rounded-md border px-3 py-2 text-left text-sm transition-colors ${
                  repo.id === selectedId
                    ? "border-accent bg-accent/5"
                    : "border-transparent hover:border-border hover:bg-surface"
                }`}
              >
                <div className="truncate font-medium">{repo.name}</div>
                <div className="truncate text-xs text-muted">{repo.path}</div>
                <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-muted">
                  <span>{repo.indexed_file_count} files</span>
                  <span>{repo.chunk_count} chunks</span>
                  <span>
                    {repo.embedded_chunk_count}/{repo.chunk_count} embedded
                  </span>
                  {repo.test_command && <span className="text-success">tests configured</span>}
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
