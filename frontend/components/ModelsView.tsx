"use client";

import { useMemo, useState } from "react";
import { ApiError } from "@/lib/api";
import { displayModelName } from "@/lib/format";
import type { InstalledModel, ModelRole } from "@/lib/types";
import { ConfirmDialog } from "./ConfirmDialog";
import { CatalogModelCard, InstalledModelCard } from "./ModelCard";
import { useModels } from "./ModelsProvider";
import { useToast } from "./Toast";

type RoleFilter = "all" | ModelRole;

const ROLE_FILTERS: { value: RoleFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "chat", label: "Reasoning" },
  { value: "embedding", label: "Embedding" },
];

/** A switch the backend refused until the caller opts into the consequences,
 * held here while the user reads them. */
interface PendingSwitch {
  role: ModelRole;
  name: string;
  detail: string;
}

export function ModelsView() {
  const { data, loading, error, pulls, refresh, pull, stopWatchingPull, activate, remove } = useModels();
  const toast = useToast();

  const [query, setQuery] = useState("");
  const [roleFilter, setRoleFilter] = useState<RoleFilter>("all");
  const [busy, setBusy] = useState(false);
  const [pendingSwitch, setPendingSwitch] = useState<PendingSwitch | null>(null);
  const [pendingRemoval, setPendingRemoval] = useState<InstalledModel | null>(null);

  const installed = useMemo(
    () => filterModels(data?.installed ?? [], query, roleFilter, (m) => [m.name, m.family ?? "", m.catalog?.label ?? ""]),
    [data, query, roleFilter],
  );
  const catalog = useMemo(
    () =>
      filterModels(data?.catalog ?? [], query, roleFilter, (entry) => [
        entry.name,
        entry.label,
        entry.publisher,
        entry.description,
        ...entry.strengths,
      ]),
    [data, query, roleFilter],
  );
  const notInstalled = catalog.filter((entry) => !entry.installed);

  async function handleActivate(role: ModelRole, name: string, confirmReindex = false) {
    setBusy(true);
    try {
      const result = await activate(role, name, confirmReindex);
      setPendingSwitch(null);
      toast(result.message, "success");
    } catch (err) {
      // 409 is the backend asking for informed consent, not a failure —
      // it carries the exact consequence text to show the user.
      if (err instanceof ApiError && err.status === 409 && !confirmReindex) {
        setPendingSwitch({ role, name, detail: err.message });
      } else {
        toast(err instanceof ApiError ? err.message : `Could not switch to ${name}`, "error");
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove(model: InstalledModel) {
    setBusy(true);
    try {
      await remove(model.name);
      toast(`Removed ${displayModelName(model.name)}.`, "success");
    } catch (err) {
      toast(err instanceof ApiError ? err.message : `Could not remove ${model.name}`, "error");
    } finally {
      setBusy(false);
      setPendingRemoval(null);
    }
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 py-6">
      <ActiveModelsSummary />

      {error && (
        <Banner tone="danger">
          {error}{" "}
          <button onClick={() => void refresh()} className="underline underline-offset-2">
            Retry
          </button>
        </Banner>
      )}

      {data && !data.backend.reachable && (
        <Banner tone="warning">
          <span className="font-medium">
            Can&apos;t reach the {data.backend.provider} server at {data.backend.base_url}.
          </span>{" "}
          Start it with <code className="rounded bg-code-bg px-1 py-0.5 font-mono">ollama serve</code>, then{" "}
          <button onClick={() => void refresh()} className="underline underline-offset-2">
            refresh
          </button>
          . The catalog below shows what you can install once it&apos;s running.
        </Banner>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <div className="relative min-w-56 flex-1">
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
            placeholder="Search models by name, publisher, or strength…"
            aria-label="Search models"
            className="w-full rounded-lg border border-border bg-surface py-2 pr-3 pl-8 text-sm outline-none transition-colors focus:border-accent"
          />
        </div>
        <div role="tablist" aria-label="Filter by role" className="flex gap-0.5 rounded-lg border border-border bg-surface p-0.5">
          {ROLE_FILTERS.map((filter) => (
            <button
              key={filter.value}
              role="tab"
              aria-selected={roleFilter === filter.value}
              onClick={() => setRoleFilter(filter.value)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                roleFilter === filter.value
                  ? "bg-accent text-accent-foreground"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      <Section
        title="Installed"
        count={installed.length}
        subtitle="Switch between these at any time — a change applies to the next run, never to one already in flight."
      >
        {loading ? (
          <SkeletonGrid />
        ) : installed.length === 0 ? (
          <EmptyState>
            {data?.installed.length
              ? "No installed model matches that search."
              : "Nothing installed yet. Pick a model below to get started."}
          </EmptyState>
        ) : (
          <Grid>
            {installed.map((model) => (
              <InstalledModelCard
                key={model.name}
                model={model}
                busy={busy}
                onActivate={() => void handleActivate(model.role, model.name)}
                onRemove={() => setPendingRemoval(model)}
              />
            ))}
          </Grid>
        )}
      </Section>

      <Section
        title="Available to install"
        count={notInstalled.length}
        subtitle="Downloads run on your machine and stay there. Nothing is sent to a cloud API."
      >
        {notInstalled.length === 0 ? (
          <EmptyState>
            {catalog.length > 0
              ? "Everything in the catalog matching this filter is already installed."
              : "No catalog model matches that search."}
          </EmptyState>
        ) : (
          <Grid>
            {notInstalled.map((entry) => (
              <CatalogModelCard
                key={entry.name}
                entry={entry}
                progress={pulls[entry.name]}
                onPull={() => void pull(entry.name)}
                onDismissProgress={() => stopWatchingPull(entry.name)}
              />
            ))}
          </Grid>
        )}
      </Section>

      {pendingSwitch && (
        <ConfirmDialog
          title="Switch the embedding model?"
          body={pendingSwitch.detail}
          confirmLabel="Switch and clear embeddings"
          tone="danger"
          busy={busy}
          onConfirm={() => void handleActivate(pendingSwitch.role, pendingSwitch.name, true)}
          onCancel={() => setPendingSwitch(null)}
        />
      )}

      {pendingRemoval && (
        <ConfirmDialog
          title={`Remove ${displayModelName(pendingRemoval.name)}?`}
          body={`This deletes the model from disk. You can install it again later, but it will have to be downloaded from scratch.`}
          confirmLabel="Remove"
          tone="danger"
          busy={busy}
          onConfirm={() => void handleRemove(pendingRemoval)}
          onCancel={() => setPendingRemoval(null)}
        />
      )}
    </div>
  );
}

function ActiveModelsSummary() {
  const { data } = useModels();
  if (!data) return null;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <ActiveModelTile
        label="Reasoning model"
        name={data.active_chat_model}
        note="Plans the work, picks tools, and writes the patch."
      />
      <ActiveModelTile
        label="Embedding model"
        name={data.active_embedding_model}
        note="Turns your code into vectors so the agent can search it."
      />
    </div>
  );
}

function ActiveModelTile({ label, name, note }: { label: string; name: string; note: string }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-[11px] font-semibold tracking-wide text-muted uppercase">{label}</p>
      <p className="mt-1 truncate font-mono text-sm font-medium">{displayModelName(name)}</p>
      <p className="mt-1 text-xs text-muted">{note}</p>
    </div>
  );
}

function Section({
  title,
  count,
  subtitle,
  children,
}: {
  title: string;
  count: number;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex flex-col gap-3">
      <div>
        <h2 className="flex items-center gap-2 text-sm font-semibold">
          {title}
          <span className="rounded-md border border-border px-1.5 py-0.5 text-[11px] font-normal text-muted">
            {count}
          </span>
        </h2>
        <p className="mt-0.5 text-xs text-muted">{subtitle}</p>
      </div>
      {children}
    </section>
  );
}

function Grid({ children }: { children: React.ReactNode }) {
  return <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">{children}</div>;
}

function SkeletonGrid() {
  return (
    <Grid>
      {[0, 1, 2].map((i) => (
        <div key={i} className="h-40 animate-pulse rounded-xl border border-border bg-surface-hover" />
      ))}
    </Grid>
  );
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-muted">
      {children}
    </div>
  );
}

function Banner({ tone, children }: { tone: "danger" | "warning"; children: React.ReactNode }) {
  const styles =
    tone === "danger"
      ? "border-danger/40 bg-danger-soft text-danger"
      : "border-warning/40 bg-warning-soft text-warning";
  return <div className={`rounded-xl border px-4 py-3 text-sm ${styles}`}>{children}</div>;
}

function filterModels<T extends { role: ModelRole }>(
  items: T[],
  query: string,
  roleFilter: RoleFilter,
  searchableFields: (item: T) => string[],
): T[] {
  const needle = query.trim().toLowerCase();
  return items.filter((item) => {
    if (roleFilter !== "all" && item.role !== roleFilter) return false;
    if (!needle) return true;
    return searchableFields(item).some((field) => field.toLowerCase().includes(needle));
  });
}
