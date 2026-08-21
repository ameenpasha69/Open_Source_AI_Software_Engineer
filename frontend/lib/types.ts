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

export type AgentStatus = "running" | "done" | "failed" | "max_iterations_reached" | "cancelled";

export type VerificationStatus = "not_applicable" | "unverified" | "failed" | "partially_verified" | "verified";

export interface AgentRunSummary {
  id: string;
  repository_id: string;
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
