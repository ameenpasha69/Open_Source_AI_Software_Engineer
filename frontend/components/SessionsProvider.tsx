"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "@/lib/api";
import type { AgentRunSummary, ChatMessage, RepositorySummary, SessionDetail, SessionSummary } from "@/lib/types";

interface SessionsContextValue {
  repositories: RepositorySummary[];
  sessions: SessionSummary[];
  activeSessionId: string | null;
  activeSession: SessionDetail | null;
  /** The run answering the newest turn, while it's still going. */
  activeRunId: string | null;
  /** A turn typed but not yet answered — shown immediately so the transcript
   * never appears to swallow what someone just sent. */
  pendingUserMessage: ChatMessage | null;
  loading: boolean;
  error: string | null;
  selectSession: (sessionId: string | null) => void;
  createSession: (repositoryId: string) => Promise<SessionSummary>;
  renameSession: (sessionId: string, title: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  sendMessage: (content: string) => Promise<AgentRunSummary>;
  onRunSettled: () => Promise<void>;
  refreshRepositories: () => Promise<void>;
}

const SessionsContext = createContext<SessionsContextValue | null>(null);

export function useSessions(): SessionsContextValue {
  const context = useContext(SessionsContext);
  if (!context) throw new Error("useSessions must be used inside a SessionsProvider");
  return context;
}

export function SessionsProvider({ children }: { children: React.ReactNode }) {
  const [repositories, setRepositories] = useState<RepositorySummary[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeSession, setActiveSession] = useState<SessionDetail | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [pendingUserMessage, setPendingUserMessage] = useState<ChatMessage | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshRepositories = useCallback(async () => {
    setRepositories(await api.listRepositories());
  }, []);

  const refreshSessions = useCallback(async () => {
    setSessions(await api.listSessions());
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        await Promise.all([refreshRepositories(), refreshSessions()]);
        setError(null);
      } catch (err) {
        setError(
          err instanceof ApiError ? err.message : "Could not reach the API — is the backend running on :8000?",
        );
      } finally {
        setLoading(false);
      }
    })();
  }, [refreshRepositories, refreshSessions]);

  // Load the transcript whenever the selected session changes.
  useEffect(() => {
    if (!activeSessionId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setActiveSession(null);
      return;
    }
    let cancelled = false;
    void api.getSession(activeSessionId).then((detail) => {
      if (!cancelled) setActiveSession(detail);
    });
    return () => {
      cancelled = true;
    };
  }, [activeSessionId]);

  const selectSession = useCallback((sessionId: string | null) => {
    setActiveSessionId(sessionId);
    setActiveRunId(null);
    setPendingUserMessage(null);
  }, []);

  const createSession = useCallback(
    async (repositoryId: string) => {
      const created = await api.createSession(repositoryId);
      await refreshSessions();
      selectSession(created.id);
      return created;
    },
    [refreshSessions, selectSession],
  );

  const renameSession = useCallback(
    async (sessionId: string, title: string) => {
      await api.renameSession(sessionId, title);
      await refreshSessions();
      if (sessionId === activeSessionId) setActiveSession(await api.getSession(sessionId));
    },
    [refreshSessions, activeSessionId],
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      await api.deleteSession(sessionId);
      await refreshSessions();
      if (sessionId === activeSessionId) selectSession(null);
    },
    [refreshSessions, activeSessionId, selectSession],
  );

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeSessionId) throw new Error("No session selected");
      const { message, run } = await api.postMessage(activeSessionId, content);
      setPendingUserMessage(message);
      setActiveRunId(run.id);
      // The first turn renames the session, so the sidebar needs to catch up.
      await refreshSessions();
      return run;
    },
    [activeSessionId, refreshSessions],
  );

  /** Called once the run's event stream closes: the assistant's turn and the
   * session's token totals are only written when the run reaches a terminal
   * status, so this is the moment the server has something new to say. */
  const onRunSettled = useCallback(async () => {
    if (!activeSessionId) return;
    setActiveSession(await api.getSession(activeSessionId));
    setPendingUserMessage(null);
    setActiveRunId(null);
    await refreshSessions();
  }, [activeSessionId, refreshSessions]);

  const value = useMemo(
    () => ({
      repositories,
      sessions,
      activeSessionId,
      activeSession,
      activeRunId,
      pendingUserMessage,
      loading,
      error,
      selectSession,
      createSession,
      renameSession,
      deleteSession,
      sendMessage,
      onRunSettled,
      refreshRepositories,
    }),
    [
      repositories,
      sessions,
      activeSessionId,
      activeSession,
      activeRunId,
      pendingUserMessage,
      loading,
      error,
      selectSession,
      createSession,
      renameSession,
      deleteSession,
      sendMessage,
      onRunSettled,
      refreshRepositories,
    ],
  );

  return <SessionsContext.Provider value={value}>{children}</SessionsContext.Provider>;
}
