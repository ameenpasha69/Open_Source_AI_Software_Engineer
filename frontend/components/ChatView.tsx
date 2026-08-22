"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { displayModelName } from "@/lib/format";
import type { ChatMessage } from "@/lib/types";
import { useAgentStream } from "@/lib/useAgentStream";
import { AssistantTurn, UserTurn } from "./ChatTurn";
import { useModels } from "./ModelsProvider";
import { useSessions } from "./SessionsProvider";
import { useToast } from "./Toast";
import { UsagePopover } from "./UsagePopover";

const EMPTY_USAGE = { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0, llm_calls: 0 };

export function ChatView({ onManageModels }: { onManageModels: () => void }) {
  const {
    repositories,
    sessions,
    activeSession,
    activeSessionId,
    activeRunId,
    pendingUserMessage,
    sendMessage,
    onRunSettled,
    createSession,
  } = useSessions();
  const { data: models } = useModels();
  const toast = useToast();
  const { events, closed } = useAgentStream(activeRunId);
  const [cancelling, setCancelling] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!closed || !activeRunId) return;
    void onRunSettled();
  }, [closed, activeRunId, onRunSettled]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [activeSession?.messages.length, events.length, pendingUserMessage]);

  async function handleCancel() {
    if (!activeRunId) return;
    setCancelling(true);
    try {
      await api.cancelAgentRun(activeRunId);
    } catch (err) {
      toast(err instanceof ApiError ? err.message : "Could not cancel the run", "error");
    } finally {
      setCancelling(false);
    }
  }

  if (!activeSessionId) {
    return <NoSession repositoryCount={repositories.length} onStart={() => void startFirst()} />;
  }

  async function startFirst() {
    if (repositories.length === 0) return;
    await createSession(repositories[0].id);
  }

  const summary = sessions.find((s) => s.id === activeSessionId) ?? null;
  const title = summary?.title ?? activeSession?.title ?? "Session";
  const repositoryId = summary?.repository_id ?? activeSession?.repository_id;
  const repository = repositories.find((r) => r.id === repositoryId) ?? null;
  const messages: ChatMessage[] = activeSession?.messages ?? [];
  const showPending =
    pendingUserMessage != null && !messages.some((m) => m.id === pendingUserMessage.id);

  return (
    <div className="flex h-full flex-col">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-border px-5 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          {/* Title from the sidebar list, not the loaded detail: the first
              message renames the session, and the list refreshes on send while
              the detail only refreshes once the run finishes. */}
          <h1 className="truncate text-sm font-medium">{title}</h1>
          {repository && (
            <span
              title={repository.path}
              className="shrink-0 rounded-md border border-border bg-surface px-1.5 py-0.5 font-mono text-[11px] text-muted"
            >
              {repository.name}
            </span>
          )}
        </div>
        <UsagePopover usage={activeSession?.usage ?? EMPTY_USAGE} contextWindow={activeSession?.context_window ?? null} />
      </header>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-6 px-5 py-6">
          {messages.length === 0 && !showPending && (
            <EmptySession repositoryName={repository?.name} chunkCount={repository?.chunk_count ?? 0} />
          )}

          {messages.map((message) =>
            message.role === "user" ? (
              <UserTurn key={message.id} message={message} />
            ) : (
              <AssistantTurn
                key={message.id}
                message={message}
                live={message.run_id != null && message.run_id === activeRunId}
                liveEvents={events}
              />
            ),
          )}

          {showPending && pendingUserMessage && (
            <>
              <UserTurn message={pendingUserMessage} />
              <AssistantTurn message={PENDING_ASSISTANT} live liveEvents={events} />
            </>
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <Composer
        repositoryLabel={repository ? `${repository.name} · ${repository.chunk_count.toLocaleString()} chunks` : ""}
        modelLabel={models ? displayModelName(models.active_chat_model) : "…"}
        running={activeRunId != null}
        cancelling={cancelling}
        onSend={sendMessage}
        onCancel={handleCancel}
        onManageModels={onManageModels}
      />
    </div>
  );
}

/** Placeholder turn shown while the run that will fill it is still going —
 * `run_id` is null so it never tries to fetch detail for a run with no rows
 * written yet; the live event stream supplies the timeline instead. */
const PENDING_ASSISTANT: ChatMessage = {
  id: "__pending__",
  role: "assistant",
  content: "",
  run_id: null,
  created_at: new Date().toISOString(),
  run: null,
  usage: null,
};

function Composer({
  repositoryLabel,
  modelLabel,
  running,
  cancelling,
  onSend,
  onCancel,
  onManageModels,
}: {
  repositoryLabel: string;
  modelLabel: string;
  running: boolean;
  cancelling: boolean;
  onSend: (content: string) => Promise<unknown>;
  onCancel: () => void;
  onManageModels: () => void;
}) {
  const [value, setValue] = useState("");
  const [sending, setSending] = useState(false);
  const toast = useToast();

  async function submit() {
    const content = value.trim();
    if (!content || sending || running) return;
    setSending(true);
    try {
      await onSend(content);
      setValue("");
    } catch (err) {
      toast(err instanceof ApiError ? err.message : "Could not send that message", "error");
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="shrink-0 px-5 pb-4">
      <div className="mx-auto max-w-3xl">
        {repositoryLabel && (
          <div className="mb-1.5 px-1 font-mono text-[11px] text-muted-subtle">{repositoryLabel}</div>
        )}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
          className="rounded-2xl border border-border bg-surface p-2.5 transition-colors focus-within:border-border-strong"
        >
          <textarea
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends; Shift+Enter is a newline. This is a chat box, and
              // most turns are one line.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void submit();
              }
            }}
            disabled={running}
            rows={2}
            placeholder={running ? "Waiting for the current turn to finish…" : "Describe a bug, or ask for a change…"}
            aria-label="Message"
            className="w-full resize-none bg-transparent px-1.5 py-1 text-sm outline-none disabled:opacity-60"
          />
          <div className="flex items-center justify-between gap-2 pt-1">
            <button
              type="button"
              onClick={onManageModels}
              title="Change the reasoning model"
              className="flex items-center gap-1.5 rounded-lg border border-border px-2 py-1 text-[11px] text-muted transition-colors hover:text-foreground"
            >
              <span className="font-mono">{modelLabel}</span>
              <span aria-hidden className="text-muted-subtle">
                ▾
              </span>
            </button>

            {running ? (
              <button
                type="button"
                onClick={onCancel}
                disabled={cancelling}
                className="rounded-lg border border-danger px-3 py-1 text-xs font-medium text-danger transition-colors hover:bg-danger-soft disabled:opacity-50"
              >
                {cancelling ? "Cancelling…" : "Stop"}
              </button>
            ) : (
              <button
                type="submit"
                disabled={sending || !value.trim()}
                className="rounded-lg bg-accent px-3 py-1 text-xs font-medium text-accent-foreground transition-colors hover:bg-accent-hover disabled:opacity-50"
              >
                {sending ? "Sending…" : "Send"}
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}

function NoSession({ repositoryCount, onStart }: { repositoryCount: number; onStart: () => void }) {
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-sm text-center">
        <p className="text-sm font-medium">
          {repositoryCount === 0 ? "Index a repository to begin" : "Pick a session, or start a new one"}
        </p>
        <p className="mt-1.5 text-sm text-muted">
          {repositoryCount === 0
            ? "Everything is read from disk and embedded locally. No code leaves your machine."
            : "Each session remembers its own conversation and searches only its own repository."}
        </p>
        {repositoryCount > 0 && (
          <button
            onClick={onStart}
            className="mt-4 rounded-lg bg-accent px-3.5 py-1.5 text-sm font-medium text-accent-foreground transition-colors hover:bg-accent-hover"
          >
            New session
          </button>
        )}
      </div>
    </div>
  );
}

function EmptySession({ repositoryName, chunkCount }: { repositoryName?: string; chunkCount: number }) {
  return (
    <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center">
      <p className="text-sm font-medium">Ready when you are</p>
      <p className="mx-auto mt-1.5 max-w-sm text-sm text-muted">
        The agent searches {chunkCount.toLocaleString()} indexed chunks in{" "}
        <span className="font-mono">{repositoryName ?? "this repository"}</span>, and remembers everything said
        in this session — so you can follow up without repeating yourself.
      </p>
    </div>
  );
}
