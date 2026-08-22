"use client";

import { useState } from "react";
import { ChatView } from "@/components/ChatView";
import { ModelsProvider } from "@/components/ModelsProvider";
import { ModelsView } from "@/components/ModelsView";
import { SessionsProvider, useSessions } from "@/components/SessionsProvider";
import { Sidebar } from "@/components/Sidebar";
import { ToastProvider } from "@/components/Toast";

type View = "chat" | "models";

export default function Home() {
  return (
    <ToastProvider>
      <ModelsProvider>
        <SessionsProvider>
          <Workbench />
        </SessionsProvider>
      </ModelsProvider>
    </ToastProvider>
  );
}

function Workbench() {
  const [view, setView] = useState<View>("chat");
  const { error, loading, activeSessionId, activeRunId } = useSessions();

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <Sidebar
        view={view}
        onViewChange={setView}
        runningSessionId={activeRunId ? activeSessionId : null}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        {error && (
          <div className="shrink-0 border-b border-danger/30 bg-danger-soft px-5 py-2 text-sm text-danger">
            {error}
          </div>
        )}

        {view === "models" ? (
          <div className="flex-1 overflow-y-auto">
            <ModelsView />
          </div>
        ) : loading ? (
          <div className="flex flex-1 items-center justify-center text-sm text-muted">Loading…</div>
        ) : (
          <ChatView onManageModels={() => setView("models")} />
        )}
      </main>
    </div>
  );
}
