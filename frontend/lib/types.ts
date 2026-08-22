export interface RepositorySummary {
  id: string;
  name: string;
  path: string;
  created_at: string;
  last_indexed_at: string | null;
  indexed_file_count: number;
  chunk_count: number;
  embedded_chunk_count: number;
  test_command: string[] | null;
  lint_command: string[] | null;
  format_command: string[] | null;
}

export interface EmbeddingSyncResult {
  chunks_embedded: number;
  chunks_removed: number;
  duration_seconds: number;
  error: string | null;
}

export interface IndexRunResult {
  index_run_id: string;
  repository_id: string;
  status: string;
  files_scanned: number;
  files_indexed: number;
  files_skipped_unchanged: number;
  files_ignored: number;
  files_deleted: number;
  chunks_created: number;
  duration_seconds: number;
  error: string | null;
  embedding: EmbeddingSyncResult | null;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  llm_calls: number;
}

export interface ContextWindow {
  model: string;
  /** null when neither the backend nor the catalog knows this model's window
   * — the UI shows the raw count rather than a made-up percentage. */
  limit_tokens: number | null;
  used_tokens: number;
  percent: number | null;
}

export interface SessionSummary {
  id: string;
  repository_id: string;
  repository_name: string;
  title: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  run_id: string | null;
  created_at: string;
  run: AgentRunSummary | null;
  usage: TokenUsage | null;
}

export interface SessionDetail extends SessionSummary {
  messages: ChatMessage[];
  usage: TokenUsage;
  context_window: ContextWindow;
}

export interface PostMessageResponse {
  message: ChatMessage;
  run: AgentRunSummary;
}

export type AgentStatus =
  | "running"
  | "done"
  | "failed"
  | "max_iterations_reached"
  | "no_progress"
  | "cancelled";

export type VerificationStatus = "not_applicable" | "unverified" | "failed" | "partially_verified" | "verified";

export interface AgentRunSummary {
  id: string;
  repository_id: string;
  session_id: string | null;
  task: string;
  status: AgentStatus;
  plan: string[];
  final_answer: string | null;
  root_cause: string | null;
  iteration_count: number;
  modified_files: string[];
  verification_status: VerificationStatus;
  started_at: string;
  finished_at: string | null;
  error: string | null;
  model: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  llm_call_count: number;
  peak_prompt_tokens: number;
}

export interface AgentEventOut {
  iteration: number;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface ModifiedFileOut {
  path: string;
  diff: string;
  lines_added: number;
  lines_removed: number;
}

export interface AgentDiffResponse {
  run_id: string;
  verification_status: VerificationStatus;
  modified_files: ModifiedFileOut[];
}

export interface TestRunOut {
  command: string;
  scope: "full" | "targeted";
  passed: boolean;
  total_tests: number | null;
  passed_tests: number | null;
  failed_tests: string[];
  failure_category: string;
  duration_seconds: number;
  stdout: string;
  stderr: string;
  created_at: string;
}

export interface ToolCallOut {
  iteration: number;
  tool_name: string;
  input: Record<string, unknown>;
  success: boolean;
  output: Record<string, unknown> | null;
  error: string | null;
  duration_seconds: number;
  created_at: string;
}

export type ModelRole = "chat" | "embedding";

export interface CatalogEntry {
  name: string;
  role: ModelRole;
  label: string;
  publisher: string;
  parameter_size: string;
  approx_size_bytes: number;
  context_window: number | null;
  description: string;
  strengths: string[];
  min_ram_gb: number;
  recommended: boolean;
  dimensions: number | null;
}

export interface CatalogEntryOut extends CatalogEntry {
  installed: boolean;
  active: boolean;
}

export interface InstalledModel {
  name: string;
  role: ModelRole;
  size_bytes: number;
  parameter_size: string | null;
  quantization: string | null;
  family: string | null;
  modified_at: string | null;
  context_length: number | null;
  active: boolean;
  loaded: boolean;
  catalog: CatalogEntry | null;
}

export interface ModelBackendInfo {
  provider: string;
  base_url: string;
  reachable: boolean;
  error: string | null;
}

export interface ModelsResponse {
  backend: ModelBackendInfo;
  active_chat_model: string;
  active_embedding_model: string;
  installed: InstalledModel[];
  catalog: CatalogEntryOut[];
}

export interface ActiveModelsResponse {
  active_chat_model: string;
  active_embedding_model: string;
  repositories_to_reindex: string[];
  message: string;
}

/** One line of a model download's NDJSON progress stream. */
export interface PullProgress {
  status: string;
  digest: string | null;
  completed: number | null;
  total: number | null;
  percent: number | null;
  done: boolean;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  app_name: string;
  llm: { provider: string; model: string; reachable: boolean };
}
