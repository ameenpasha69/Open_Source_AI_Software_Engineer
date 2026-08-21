import type {
  AgentDiffResponse,
  AgentEventOut,
  AgentRunSummary,
  IndexRunResult,
  RepositorySummary,
  TestRunOut,
  ToolCallOut,
} from "./types";

export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

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
    headers: { "Content-Type": "application/json", ...init?.headers },
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

  getAgentRun: (runId: string) => request<AgentRunSummary>(`/api/agent/${runId}`),

  getAgentEvents: (runId: string) => request<AgentEventOut[]>(`/api/agent/${runId}/events`),

  getAgentDiff: (runId: string) => request<AgentDiffResponse>(`/api/agent/${runId}/diff`),

  getAgentTests: (runId: string) => request<TestRunOut[]>(`/api/agent/${runId}/tests`),

  getAgentToolCalls: (runId: string) => request<ToolCallOut[]>(`/api/agent/${runId}/tool-calls`),

  cancelAgentRun: (runId: string) => request<AgentRunSummary>(`/api/agent/${runId}/cancel`, { method: "POST" }),
};

export { ApiError };
