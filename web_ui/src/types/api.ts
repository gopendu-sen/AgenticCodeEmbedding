export type Role = "user" | "assistant" | "system";

export interface SourceMetadata {
  repo_name?: string;
  file_path?: string;
  start_line?: number | string;
  end_line?: number | string;
  node_type?: string;
  language?: string;
  symbol?: string;
}

export interface SourceChunk {
  citation_index?: number;
  collection?: string;
  distance?: number;
  metadata?: SourceMetadata;
  document?: string;
}

export interface ChatMetaEvent {
  event: "meta";
  intent: string;
  repo_stores: string[];
  source_count: number;
  no_sources: boolean;
  sources: SourceChunk[];
}

export interface ChatTokenEvent {
  event: "token";
  token: string;
}

export interface ChatDoneEvent {
  event: "done";
  response: string;
  sources: SourceChunk[];
  intent: string;
  cited_indices: number[];
  no_sources: boolean;
  model?: string;
}

export interface ChatErrorEvent {
  event: "error";
  message: string;
}

export type ChatEvent = ChatMetaEvent | ChatTokenEvent | ChatDoneEvent | ChatErrorEvent;

export interface ChatRequestBody {
  session_id: string;
  message: string;
  store_names: string[];
  enable_summarisation?: boolean;
  enable_intent_tracking?: boolean;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  sources: SourceChunk[];
  citedIndices: number[];
  intent: string;
  createdAt: number;
}

export interface SessionSummary {
  session_id: string;
  summary: string;
  intents: string[];
  updated_at: number;
  last_message: string;
  message_count: number;
}

export interface SessionHistoryResponse {
  session_id: string;
  summary: string;
  intents: string[];
  messages: Array<{
    id: number;
    role: Role;
    content: string;
    sources: SourceChunk[];
    cited_indices: number[];
    intent: string;
    created_at: number;
  }>;
  updated_at: number;
}

export interface UIConfigResponse {
  title: string;
  subtitle: string;
  assistant_greeting: string;
  input_placeholder: string;
  spinner_text: string;
  show_sources: boolean;
  max_context_chunks: number;
  history_messages: number;
  chat_api_host?: string;
  chat_api_port?: number;
  embedding_api_base_url?: string;
  ui_host?: string;
  ui_port?: number;
}

export interface EmbeddingJob {
  job_id: string;
  status: string;
  repo_name: string;
  repo_path: string;
  config_path: string;
  created_at_utc: string;
  updated_at_utc: string;
  pid?: number | null;
  summary?: Record<string, unknown> | null;
  error?: string;
  report_path?: string;
  log_path?: string;
  partial?: boolean;
  file_status_counts?: Record<string, number>;
  files?: Array<{
    file_path?: string;
    status?: string;
    partial?: boolean;
    error?: string;
  }>;
}

export interface EvaluationRule {
  id: string;
  title: string;
  definition: string;
  strong_signals: string[];
  weak_signals: string[];
  false_positives: string[];
}

export interface EvaluationRulesPayload {
  version: number;
  updated_at_utc: string;
  path?: string;
  items: EvaluationRule[];
}

export interface EvaluationEvidence {
  citation_id: number;
  collection: string;
  distance: number;
  file_path: string;
  start_line: number | string;
  end_line: number | string;
  snippet: string;
  node_type: string;
  repo_name: string;
}

export interface EvaluationReportItem {
  id: string;
  title: string;
  definition: string;
  status: "Detected" | "Not Detected" | "Needs Review";
  reason: string;
  confidence: number;
  matched_strong_signals: string[];
  matched_weak_signals: string[];
  false_positive_risks: string[];
  evidences: EvaluationEvidence[];
  decision_policy?: "llm" | "deterministic_fallback" | "recall_override_strong" | "recall_override_weak";
  deterministic_signal_counts?: {
    strong: number;
    weak: number;
    false_positive: number;
  };
}

export interface EvaluationReportSummary {
  report_type: "evaluation_report";
  job_id?: string;
  repo_name: string;
  repo_path: string;
  generated_at_utc: string;
  rules_version: number;
  rules_hash: string;
  summary: {
    detected: number;
    not_detected: number;
    needs_review: number;
  };
  items: EvaluationReportItem[];
}

export interface EvaluationJob {
  job_id: string;
  status: string;
  repo_name: string;
  repo_path: string;
  config_path: string;
  created_at_utc: string;
  updated_at_utc: string;
  started_at_utc?: string;
  finished_at_utc?: string;
  status_counts?: {
    detected?: number;
    not_detected?: number;
    needs_review?: number;
  };
  rule_count?: number;
  report_html_path?: string;
  report_json_path?: string;
  log_path?: string;
  error?: string;
  partial?: boolean;
  pid?: number | null;
  summary?: Record<string, unknown> | null;
}
