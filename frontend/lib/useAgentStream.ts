"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE_URL } from "./api";
import type { AgentEventOut, AgentStatus } from "./types";

interface AgentStreamState {
  events: AgentEventOut[];
  closed: boolean;
  finalStatus: AgentStatus | null;
}

/** Opens an EventSource against GET /api/agent/{runId}/stream and
 * accumulates events as they arrive. Regular events are unnamed SSE
 * messages (event_type travels inside the JSON body), so a single
 * `onmessage` handler covers every event_type without registering a
 * listener per type ahead of time; the stream's one named event,
 * "run_completed", closes the connection. */
export function useAgentStream(runId: string | null): AgentStreamState {
  const [events, setEvents] = useState<AgentEventOut[]>([]);
  const [closed, setClosed] = useState(false);
  const [finalStatus, setFinalStatus] = useState<AgentStatus | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    // Resetting state when `runId` changes, before opening the new
    // connection below — this effect *is* the subscription setup, so the
    // reset and the subscription are inherently one unit of work, not two.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setEvents([]);
    setClosed(false);
    setFinalStatus(null);
    if (!runId) return;

    const source = new EventSource(`${API_BASE_URL}/api/agent/${runId}/stream`);
    sourceRef.current = source;

    source.onmessage = (message) => {
      const payload = JSON.parse(message.data) as AgentEventOut;
      setEvents((prev) => [...prev, payload]);
    };

    source.addEventListener("run_completed", (message: MessageEvent) => {
      const payload = JSON.parse(message.data) as { status: AgentStatus };
      setFinalStatus(payload.status);
      setClosed(true);
      source.close();
    });

    source.onerror = () => {
      // A network hiccup, or the server closed the connection after
      // run_completed (which we've already handled above) — either way,
      // there's nothing useful to retry once the run itself is done.
      if (source.readyState === EventSource.CLOSED) {
        setClosed(true);
      }
    };

    return () => {
      source.close();
      sourceRef.current = null;
    };
  }, [runId]);

  return { events, closed, finalStatus };
}
