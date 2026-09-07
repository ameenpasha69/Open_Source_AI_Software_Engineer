import type {
  ActiveModelsResponse,
  AgentDiffResponse,
  AgentEventOut,
  AgentRunSummary,
  HealthResponse,
  IndexRunResult,
  ModelRole,
  ModelsResponse,
  PostMessageResponse,
  PullProgress,
  RepositorySummary,
  SessionDetail,
  SessionSummary,
  TestRunOut,
  ToolCallOut,
} from "./types";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** The API's token. The backend generates one on first start and logs it; put
 * it in NEXT_PUBLIC_API_AUTH_TOKEN. Empty means the backend is running with
 * AUTH_ENABLED=false, in which case no header is sent and nothing changes. */
export const API_AUTH_TOKEN = process.env.NEXT_PUBLIC_API_AUTH_TOKEN ?? "";

export function authHeaders(): Record<string, string> {
  return API_AUTH_TOKEN ? { Authorization: `Bearer ${API_AUTH_TOKEN}` } : {};
}

/** EventSource cannot set headers, so the stream carries the token in the URL
 * instead. Only for that case -- everything else uses the header. */
export function withToken(url: string): string {
  if (!API_AUTH_TOKEN) return url;
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}token=${encodeURIComponent(API_AUTH_TOKEN)}`;
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...authHeaders(), ...init?.headers },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail ?? `Request to ${path} failed with ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listRepositories: () => request<RepositorySummary[]>("/api/repositories"),

  getRepository: (repositoryId: string) => request<RepositorySummary>(`/api/repositories/${repositoryId}`),

  indexRepository: (path: string, name?: string) =>
    request<IndexRunResult>("/api/repositories/index", {
      method: "POST",
      body: JSON.stringify({ path, name }),
    }),

  runAgent: (repositoryId: string, task: string) =>
    request<AgentRunSummary>("/api/agent/run", {
      method: "POST",
      body: JSON.stringify({ repository_id: repositoryId, task }),
    }),

  listSessions: (repositoryId?: string) =>
    request<SessionSummary[]>(
      repositoryId ? `/api/sessions?repository_id=${encodeURIComponent(repositoryId)}` : "/api/sessions",
    ),

  createSession: (repositoryId: string, title?: string) =>
    request<SessionSummary>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ repository_id: repositoryId, title }),
    }),

  getSession: (sessionId: string) => request<SessionDetail>(`/api/sessions/${sessionId}`),

  renameSession: (sessionId: string, title: string) =>
    request<SessionSummary>(`/api/sessions/${sessionId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    }),

  deleteSession: async (sessionId: string): Promise<void> => {
    const res = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    if (!res.ok) throw new ApiError(res.status, `Could not delete session ${sessionId}`);
  },

  postMessage: (sessionId: string, content: string) =>
    request<PostMessageResponse>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),

  getAgentRun: (runId: string) => request<AgentRunSummary>(`/api/agent/${runId}`),

  getAgentEvents: (runId: string) => request<AgentEventOut[]>(`/api/agent/${runId}/events`),

  getAgentDiff: (runId: string) => request<AgentDiffResponse>(`/api/agent/${runId}/diff`),

  getAgentTests: (runId: string) => request<TestRunOut[]>(`/api/agent/${runId}/tests`),

  getAgentToolCalls: (runId: string) => request<ToolCallOut[]>(`/api/agent/${runId}/tool-calls`),

  cancelAgentRun: (runId: string) => request<AgentRunSummary>(`/api/agent/${runId}/cancel`, { method: "POST" }),

  health: () => request<HealthResponse>("/api/health"),

  listModels: () => request<ModelsResponse>("/api/models"),

  setActiveModel: (role: ModelRole, name: string, confirmReindex = false) =>
    request<ActiveModelsResponse>("/api/models/active", {
      method: "POST",
      body: JSON.stringify({ role, name, confirm_reindex: confirmReindex }),
    }),

  deleteModel: (name: string) =>
    request<ActiveModelsResponse>(`/api/models?name=${encodeURIComponent(name)}`, { method: "DELETE" }),

  /** Downloads a model, invoking `onProgress` for each update the backend
   * emits. Hand-rolled rather than EventSource: a pull is a POST, and
   * EventSource can only issue GETs. Aborting `signal` drops the client's
   * interest in the stream — the backend keeps downloading, and Ollama
   * resumes rather than restarting if the pull is re-opened later. */
  pullModel: async (
    name: string,
    onProgress: (progress: PullProgress) => void,
    signal?: AbortSignal,
  ): Promise<void> => {
    const res = await fetch(`${API_BASE_URL}/api/models/pull`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ name }),
      signal,
    });
    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, body.detail ?? `Could not start the download of ${name}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // A read can land mid-line, so the trailing fragment stays buffered
      // until the newline that completes it arrives.
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (line.trim()) onProgress(JSON.parse(line) as PullProgress);
      }
    }
    if (buffer.trim()) onProgress(JSON.parse(buffer) as PullProgress);
  },
};

export { ApiError };
