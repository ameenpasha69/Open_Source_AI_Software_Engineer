"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError } from "@/lib/api";
import { displayModelName, formatBytes } from "@/lib/format";
import { useModels } from "./ModelsProvider";
import { useToast } from "./Toast";

/** The header's quick switch: the reasoning models already installed, one
 * click each. Anything that needs a decision — installing something new,
 * changing the embedding model — belongs on the Models page instead, so
 * this stays a menu you can use without reading. */
export function ModelSwitcher({ onManageModels }: { onManageModels: () => void }) {
  const { data, activate } = useModels();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [switching, setSwitching] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    window.addEventListener("mousedown", onPointerDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onPointerDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const chatModels = (data?.installed ?? []).filter((model) => model.role === "chat");
  const active = data?.active_chat_model ?? "…";
  const reachable = data?.backend.reachable ?? false;

  async function handleSelect(name: string) {
    setSwitching(true);
    try {
      const result = await activate("chat", name);
      toast(result.message, "success");
      setOpen(false);
    } catch (err) {
      toast(err instanceof ApiError ? err.message : `Could not switch to ${name}`, "error");
    } finally {
      setSwitching(false);
    }
  }

  return (
    <div ref={containerRef} className="relative">
      <button
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex max-w-64 items-center gap-2 rounded-lg border border-border bg-surface px-2.5 py-1.5 text-xs transition-colors hover:border-border-strong"
      >
        <span
          aria-hidden
          title={reachable ? "Model backend reachable" : "Model backend unreachable"}
          className={`h-1.5 w-1.5 shrink-0 rounded-full ${reachable ? "bg-success" : "bg-danger"}`}
        />
        <span className="truncate font-mono font-medium">{displayModelName(active)}</span>
        <span aria-hidden className="shrink-0 text-muted-subtle">
          ▾
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className="animate-enter absolute top-full right-0 z-40 mt-1.5 w-80 overflow-hidden rounded-xl border border-border bg-surface-raised shadow-lg"
        >
          <div className="border-b border-border px-3 py-2">
            <p className="text-[11px] font-semibold tracking-wide text-muted uppercase">Reasoning model</p>
            <p className="mt-0.5 text-[11px] text-muted-subtle">
              Applies to the next run. A run already in flight keeps its model.
            </p>
          </div>

          <div className="max-h-72 overflow-y-auto p-1">
            {chatModels.length === 0 && (
              <p className="px-2 py-3 text-xs text-muted">
                {reachable ? "No reasoning models installed yet." : "Model backend unreachable."}
              </p>
            )}
            {chatModels.map((model) => {
              const isActive = model.active;
              return (
                <button
                  key={model.name}
                  role="menuitem"
                  disabled={switching || isActive}
                  onClick={() => void handleSelect(model.name)}
                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-2 text-left transition-colors ${
                    isActive ? "bg-accent-soft" : "hover:bg-surface-hover"
                  } disabled:cursor-default`}
                >
                  <span className={`w-3 shrink-0 text-xs ${isActive ? "text-accent" : "text-transparent"}`}>✓</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate font-mono text-xs font-medium">
                      {displayModelName(model.name)}
                    </span>
                    <span className="block truncate text-[11px] text-muted">
                      {[model.parameter_size, formatBytes(model.size_bytes), model.loaded ? "warm" : null]
                        .filter(Boolean)
                        .join(" · ")}
                    </span>
                  </span>
                </button>
              );
            })}
          </div>

          <button
            role="menuitem"
            onClick={() => {
              setOpen(false);
              onManageModels();
            }}
            className="w-full border-t border-border px-3 py-2 text-left text-xs font-medium text-accent transition-colors hover:bg-surface-hover"
          >
            Manage & install models →
          </button>
        </div>
      )}
    </div>
  );
}
