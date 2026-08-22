"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";

interface Props {
  onIndexed: (repositoryId: string) => void;
  onCancel: () => void;
}

/** A distinct "pick a new repository" flow, separate from the "+" on an
 * already-indexed project (which just opens a new session — no path needed).
 * This is the one place a filesystem path has to be typed, since a browser
 * can't hand back an absolute host path for a native subprocess to index. */
export function IndexRepositoryDialog({ onIndexed, onCancel }: Props) {
  const [path, setPath] = useState("");
  const [name, setName] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!path.trim() || indexing) return;
    setIndexing(true);
    setError(null);
    try {
      const result = await api.indexRepository(path.trim(), name.trim() || undefined);
      onIndexed(result.repository_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not index that path");
    } finally {
      setIndexing(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label="Add a repository"
        onClick={(event) => event.stopPropagation()}
        className="animate-enter w-full max-w-md rounded-2xl border border-border bg-surface-raised p-5 shadow-lg"
      >
        <h2 className="text-base font-semibold">Add a repository</h2>
        <p className="mt-1 text-sm leading-relaxed text-muted">
          Point at a local Git repository. It&apos;s indexed and embedded on this machine — nothing is uploaded.
        </p>

        <form onSubmit={handleSubmit} className="mt-4 flex flex-col gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted">Path</span>
            <input
              ref={inputRef}
              value={path}
              onChange={(event) => setPath(event.target.value)}
              placeholder="/absolute/path/to/a/repo"
              className="rounded-lg border border-border bg-background px-3 py-1.5 font-mono text-sm outline-none transition-colors focus:border-accent"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted">Name (optional)</span>
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Defaults to the folder name"
              className="rounded-lg border border-border bg-background px-3 py-1.5 text-sm outline-none transition-colors focus:border-accent"
            />
          </label>

          {error && <p className="text-xs text-danger">{error}</p>}

          <div className="mt-1 flex justify-end gap-2">
            <button
              type="button"
              onClick={onCancel}
              className="rounded-lg border border-border px-3.5 py-1.5 text-sm font-medium transition-colors hover:bg-surface-hover"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={indexing || !path.trim()}
              className="rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent-hover disabled:opacity-50"
            >
              {indexing ? "Indexing…" : "Add & index"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
