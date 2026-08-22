"use client";

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

type ToastTone = "success" | "error" | "info";

interface Toast {
  id: number;
  tone: ToastTone;
  message: string;
}

const ToastContext = createContext<(message: string, tone?: ToastTone) => void>(() => {});

/** Confirmation for actions whose result isn't visible where the user
 * clicked — switching models from the header changes what the *next* run
 * uses, with nothing on screen to show for it otherwise. */
export function useToast() {
  return useContext(ToastContext);
}

const DISMISS_AFTER_MS = 5000;

const TONE_STYLES: Record<ToastTone, string> = {
  success: "border-success/40 bg-success-soft text-success",
  error: "border-danger/40 bg-danger-soft text-danger",
  info: "border-border bg-surface-raised text-foreground",
};

const TONE_ICONS: Record<ToastTone, string> = { success: "✓", error: "✗", info: "•" };

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const nextId = useRef(0);

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    (message: string, tone: ToastTone = "info") => {
      const id = nextId.current++;
      setToasts((current) => [...current, { id, tone, message }]);
      setTimeout(() => dismiss(id), DISMISS_AFTER_MS);
    },
    [dismiss],
  );

  const value = useMemo(() => push, [push]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed bottom-4 left-1/2 z-50 flex w-full max-w-md -translate-x-1/2 flex-col gap-2 px-4"
      >
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`animate-enter pointer-events-auto flex items-start gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm shadow-lg ${TONE_STYLES[toast.tone]}`}
          >
            <span aria-hidden className="mt-px shrink-0 font-semibold">
              {TONE_ICONS[toast.tone]}
            </span>
            <span className="flex-1 leading-snug">{toast.message}</span>
            <button
              onClick={() => dismiss(toast.id)}
              aria-label="Dismiss"
              className="shrink-0 opacity-60 transition-opacity hover:opacity-100"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}
