"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { ActiveModelsResponse, ModelRole, ModelsResponse, PullProgress } from "@/lib/types";

interface ModelsContextValue {
  data: ModelsResponse | null;
  loading: boolean;
  /** Only set when the models list itself couldn't be fetched — a backend
   * that's up but has no models is not an error. */
  error: string | null;
  /** In-flight and just-finished downloads, keyed by model name. */
  pulls: Record<string, PullProgress>;
  refresh: () => Promise<void>;
  pull: (name: string) => Promise<void>;
  stopWatchingPull: (name: string) => void;
  activate: (role: ModelRole, name: string, confirmReindex?: boolean) => Promise<ActiveModelsResponse>;
  remove: (name: string) => Promise<void>;
}

const ModelsContext = createContext<ModelsContextValue | null>(null);

export function useModels(): ModelsContextValue {
  const context = useContext(ModelsContext);
  if (!context) throw new Error("useModels must be used inside a ModelsProvider");
  return context;
}

export function ModelsProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pulls, setPulls] = useState<Record<string, PullProgress>>({});
  const abortControllers = useRef<Record<string, AbortController>>({});

  const refresh = useCallback(async () => {
    try {
      setData(await api.listModels());
      setError(null);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Could not reach the API — is the backend running on :8000?",
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Abort every open progress stream on unmount. The downloads themselves
  // keep running on the backend — this only drops our interest in watching.
  useEffect(() => {
    const controllers = abortControllers.current;
    return () => {
      for (const controller of Object.values(controllers)) controller.abort();
    };
  }, []);

  const pull = useCallback(
    async (name: string) => {
      if (abortControllers.current[name]) return; // already downloading
      const controller = new AbortController();
      abortControllers.current[name] = controller;
      setPulls((current) => ({ ...current, [name]: { ...EMPTY_PROGRESS, status: "starting" } }));

      try {
        await api.pullModel(
          name,
          (progress) => setPulls((current) => ({ ...current, [name]: progress })),
          controller.signal,
        );
      } catch (err) {
        if (controller.signal.aborted) return;
        setPulls((current) => ({
          ...current,
          [name]: {
            ...EMPTY_PROGRESS,
            status: "failed",
            done: true,
            error: err instanceof ApiError ? err.message : `Could not download ${name}`,
          },
        }));
      } finally {
        delete abortControllers.current[name];
      }
      await refresh();
    },
    [refresh],
  );

  /** Clears a finished download from the progress list. Deliberately does
   * not abort a running one — a half-downloaded model is the one thing a
   * user can't easily recover from by clicking again. */
  const stopWatchingPull = useCallback((name: string) => {
    setPulls((current) => {
      const next = { ...current };
      delete next[name];
      return next;
    });
  }, []);

  const activate = useCallback(
    async (role: ModelRole, name: string, confirmReindex = false) => {
      const result = await api.setActiveModel(role, name, confirmReindex);
      await refresh();
      return result;
    },
    [refresh],
  );

  const remove = useCallback(
    async (name: string) => {
      await api.deleteModel(name);
      await refresh();
    },
    [refresh],
  );

  const value = useMemo(
    () => ({ data, loading, error, pulls, refresh, pull, stopWatchingPull, activate, remove }),
    [data, loading, error, pulls, refresh, pull, stopWatchingPull, activate, remove],
  );

  return <ModelsContext.Provider value={value}>{children}</ModelsContext.Provider>;
}

const EMPTY_PROGRESS: PullProgress = {
  status: "",
  digest: null,
  completed: null,
  total: null,
  percent: null,
  done: false,
  error: null,
};
