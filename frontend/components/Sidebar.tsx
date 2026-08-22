"use client";

import { useMemo, useState } from "react";
import { formatRelativeTime } from "@/lib/format";
import type { RepositorySummary, SessionSummary } from "@/lib/types";
import { IndexRepositoryDialog } from "./IndexRepositoryDialog";
import { useSessions } from "./SessionsProvider";
import { ThemeToggle } from "./ThemeToggle";
import { useToast } from "./Toast";

type View = "chat" | "models";

interface Props {
  view: View;
  onViewChange: (view: View) => void;
  runningSessionId: string | null;
}

export function Sidebar({ view, onViewChange, runningSessionId }: Props) {
  const { repositories, sessions, activeSessionId, selectSession, createSession, deleteSession, refreshRepositories } =
    useSessions();
  const toast = useToast();
  const [query, setQuery] = useState("");
  const [showNewRepository, setShowNewRepository] = useState(false);

  const groups = useMemo(() => groupSessions(repositories, sessions, query), [repositories, sessions, query]);

  async function handleIndexed(repositoryId: string) {
    await refreshRepositories();
    setShowNewRepository(false);
    toast("Repository added.", "success");
    await createSession(repositoryId);
    onViewChange("chat");
  }

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-border bg-background-subtle">
      <div className="flex items-center gap-2 px-3 pt-3 pb-2">
        <span
          aria-hidden
          className="grid h-6 w-6 shrink-0 place-items-center rounded-md bg-accent text-[11px] font-bold text-accent-foreground"
        >
          ⌥
        </span>
        <span className="truncate text-xs font-semibold">Local AI Software Engineer</span>
      </div>

      <div className="px-3 pb-2">
        <label className="relative block">
          <svg
            aria-hidden
            viewBox="0 0 16 16"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="absolute top-1/2 left-2.5 h-3.5 w-3.5 -translate-y-1/2 text-muted-subtle"
          >
            <circle cx="7" cy="7" r="4.5" />
            <path d="m10.5 10.5 3 3" strokeLinecap="round" />
          </svg>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search sessions"
            aria-label="Search sessions"
            className="w-full rounded-lg border border-border bg-surface py-1.5 pr-2 pl-8 text-xs outline-none transition-colors focus:border-accent"
          />
        </label>
      </div>

      <nav aria-label="Primary" className="mx-3 mb-2 flex gap-0.5 rounded-lg border border-border bg-surface p-0.5">
        {(["chat", "models"] as View[]).map((value) => (
          <button
            key={value}
            aria-current={view === value ? "page" : undefined}
            onClick={() => onViewChange(value)}
            className={`flex-1 rounded-md px-2 py-1 text-xs font-medium capitalize transition-colors ${
              view === value ? "bg-accent text-accent-foreground" : "text-muted hover:text-foreground"
            }`}
          >
            {value}
          </button>
        ))}
      </nav>

      <button
        onClick={() => setShowNewRepository(true)}
        className="mx-3 mb-3 flex items-center justify-center gap-1.5 rounded-lg border border-dashed border-border-strong px-2 py-1.5 text-xs font-medium text-muted transition-colors hover:border-accent/50 hover:text-accent"
      >
        <span aria-hidden className="text-sm leading-none">
          +
        </span>
        New repository
      </button>

      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {repositories.length === 0 && (
          <p className="px-2 py-3 text-xs text-muted">
            No repositories yet — use &ldquo;New repository&rdquo; above to index one.
          </p>
        )}

        {groups.map(({ repository, sessions: repoSessions }) => (
          <section key={repository.id} className="mb-3">
            <div className="group flex items-center justify-between gap-1 px-2 py-1">
              <span className="truncate text-[11px] font-medium text-muted-subtle" title={repository.path}>
                {repository.name}
              </span>
              <button
                onClick={() => void createSession(repository.id)}
                aria-label={`New session in ${repository.name}`}
                title={`New session in ${repository.name}`}
                className="shrink-0 rounded px-1 text-sm leading-none text-muted-subtle transition-colors hover:text-foreground"
              >
                +
              </button>
            </div>

            {repoSessions.length === 0 && (
              <p className="px-2 pb-1 text-[11px] text-muted-subtle">No sessions yet</p>
            )}

            <ul>
              {repoSessions.map((session) => (
                <li key={session.id} className="group/item relative">
                  <button
                    onClick={() => {
                      selectSession(session.id);
                      onViewChange("chat");
                    }}
                    aria-current={session.id === activeSessionId ? "true" : undefined}
                    className={`flex w-full items-center gap-2 rounded-lg py-1.5 pr-7 pl-2 text-left transition-colors ${
                      session.id === activeSessionId ? "bg-surface-hover" : "hover:bg-surface-hover/60"
                    }`}
                  >
                    <span
                      aria-hidden
                      className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                        session.id === runningSessionId
                          ? "animate-pulse bg-accent"
                          : session.id === activeSessionId
                            ? "bg-foreground"
                            : "border border-border-strong"
                      }`}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-xs">{session.title}</span>
                    </span>
                  </button>
                  <button
                    onClick={() => void deleteSession(session.id)}
                    aria-label={`Delete session ${session.title}`}
                    title={`Delete "${session.title}" (${session.message_count} message(s), last active ${formatRelativeTime(session.updated_at)})`}
                    className="absolute top-1/2 right-1 -translate-y-1/2 rounded px-1 text-xs text-muted-subtle opacity-0 transition-opacity group-hover/item:opacity-100 hover:text-danger focus-visible:opacity-100"
                  >
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>

      <div className="flex items-center justify-between gap-2 border-t border-border px-3 py-2">
        <span className="truncate text-[11px] text-muted-subtle">
          {repositories.length} repositor{repositories.length === 1 ? "y" : "ies"}
        </span>
        <ThemeToggle />
      </div>

      {showNewRepository && (
        <IndexRepositoryDialog onIndexed={handleIndexed} onCancel={() => setShowNewRepository(false)} />
      )}
    </aside>
  );
}

interface Group {
  repository: RepositorySummary;
  sessions: SessionSummary[];
}

function groupSessions(
  repositories: RepositorySummary[],
  sessions: SessionSummary[],
  query: string,
): Group[] {
  const needle = query.trim().toLowerCase();
  const matches = needle
    ? sessions.filter((s) => s.title.toLowerCase().includes(needle))
    : sessions;

  const groups = repositories.map((repository) => ({
    repository,
    sessions: matches.filter((session) => session.repository_id === repository.id),
  }));

  // While searching, an empty project is noise — hide it. Otherwise keep it,
  // so a freshly indexed repository is visibly there to start a session in.
  return needle ? groups.filter((group) => group.sessions.length > 0) : groups;
}
