"use client";

import { useEffect, useRef } from "react";

interface Props {
  title: string;
  body: React.ReactNode;
  confirmLabel: string;
  tone?: "accent" | "danger";
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Blocking confirmation for the two model actions that destroy something:
 * switching embedding models (throws away every indexed vector) and removing
 * a model from disk. */
export function ConfirmDialog({ title, body, confirmLabel, tone = "accent", busy, onConfirm, onCancel }: Props) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    confirmRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4 backdrop-blur-[2px]"
      onClick={onCancel}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(event) => event.stopPropagation()}
        className="animate-enter w-full max-w-md rounded-2xl border border-border bg-surface-raised p-5 shadow-lg"
      >
        <h2 className="text-base font-semibold">{title}</h2>
        <div className="mt-2 text-sm leading-relaxed text-muted">{body}</div>
        <div className="mt-5 flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="rounded-lg border border-border px-3.5 py-1.5 text-sm font-medium transition-colors hover:bg-surface-hover"
          >
            Cancel
          </button>
          <button
            ref={confirmRef}
            onClick={onConfirm}
            disabled={busy}
            className={`rounded-lg px-3.5 py-1.5 text-sm font-medium text-accent-foreground transition-colors disabled:opacity-50 ${
              tone === "danger" ? "bg-danger hover:opacity-90" : "bg-accent hover:bg-accent-hover"
            }`}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
