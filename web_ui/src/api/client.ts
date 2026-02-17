import type {
  ChatEvent,
  ChatRequestBody,
  EmbeddingJob,
  EvaluationJob,
  EvaluationRulesPayload,
  SessionHistoryResponse,
  SessionSummary,
  UIConfigResponse
} from "../types/api";


function parseSSEChunk(chunk: string): ChatEvent | null {
  const lines = chunk
    .split("\n")
    .map((line) => line.trimEnd())
    .filter((line) => line.length > 0);
  if (!lines.length) {
    return null;
  }

  let eventName = "message";
  const dataLines: string[] = [];
  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }
  if (!dataLines.length) {
    return null;
  }

  const raw = dataLines.join("\n");
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return { event: "error", message: `Invalid SSE payload: ${raw}` };
  }
  return { event: eventName, ...payload } as ChatEvent;
}


class BaseApiClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    const normalized = baseUrl.trim().replace(/\/+$/, "");
    this.baseUrl = normalized;
  }

  protected url(path: string): string {
    if (!this.baseUrl) {
      return path;
    }
    return `${this.baseUrl}${path}`;
  }

  protected async request(path: string, init: RequestInit = {}, tag = "request"): Promise<Response> {
    const method = (init.method ?? "GET").toUpperCase();
    const url = this.url(path);
    const body = typeof init.body === "string" ? init.body : "";
    const started = performance.now();
    console.info("[REST][start]", {
      tag,
      method,
      url,
      hasBody: body.length > 0,
      bodyChars: body.length
    });
    try {
      const response = await fetch(url, init);
      const elapsedMs = performance.now() - started;
      console.info("[REST][done]", {
        tag,
        method,
        url,
        status: response.status,
        ok: response.ok,
        elapsedMs: Number(elapsedMs.toFixed(2))
      });
      return response;
    } catch (error) {
      const elapsedMs = performance.now() - started;
      console.error("[REST][error]", {
        tag,
        method,
        url,
        elapsedMs: Number(elapsedMs.toFixed(2)),
        error
      });
      throw error;
    }
  }
}


export class ChatApiClient extends BaseApiClient {
  async getUIConfig(): Promise<UIConfigResponse> {
    const response = await this.request("/ui-config", {}, "ui_config");
    if (!response.ok) {
      throw new Error(`Failed to load UI config: ${response.status}`);
    }
    return (await response.json()) as UIConfigResponse;
  }

  async health(): Promise<void> {
    const response = await this.request("/health", {}, "health");
    if (!response.ok) {
      throw new Error(`Backend health check failed: ${response.status}`);
    }
  }

  async listStores(): Promise<string[]> {
    const response = await this.request("/stores", {}, "stores_list");
    if (!response.ok) {
      throw new Error(`Failed to list stores: ${response.status}`);
    }
    const payload = (await response.json()) as { stores?: string[] };
    return payload.stores ?? [];
  }

  async listSessions(limit = 100): Promise<SessionSummary[]> {
    const response = await this.request(`/sessions?limit=${limit}`, {}, "sessions_list");
    if (!response.ok) {
      throw new Error(`Failed to list sessions: ${response.status}`);
    }
    const payload = (await response.json()) as { sessions?: SessionSummary[] };
    return payload.sessions ?? [];
  }

  async getHistory(sessionId: string): Promise<SessionHistoryResponse> {
    const response = await this.request(`/history/${encodeURIComponent(sessionId)}`, {}, "history_get");
    if (!response.ok) {
      throw new Error(`Failed to load history: ${response.status}`);
    }
    return (await response.json()) as SessionHistoryResponse;
  }

  async deleteSession(sessionId: string): Promise<void> {
    const response = await this.request(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "DELETE"
    }, "session_delete");
    if (!response.ok) {
      throw new Error(`Failed to delete session: ${response.status}`);
    }
  }

  async streamChat(
    body: ChatRequestBody,
    onEvent: (event: ChatEvent) => void
  ): Promise<void> {
    const streamStarted = performance.now();
    console.info("[REST][stream_start]", {
      tag: "chat_stream",
      sessionId: body.session_id,
      stores: body.store_names,
      messageChars: body.message.length
    });
    const response = await this.request("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }, "chat_stream_open");
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Chat request failed: ${detail || response.status}`);
    }
    if (!response.body) {
      throw new Error("Chat stream unavailable: missing response body");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let eventCount = 0;
    let tokenEventCount = 0;
    let tokenChars = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let splitIndex = buffer.indexOf("\n\n");
      while (splitIndex >= 0) {
        const chunk = buffer.slice(0, splitIndex);
        buffer = buffer.slice(splitIndex + 2);
        const event = parseSSEChunk(chunk);
        if (event) {
          eventCount += 1;
          if (event.event === "token") {
            tokenEventCount += 1;
            tokenChars += event.token.length;
          }
          console.debug("[REST][stream_event]", {
            tag: "chat_stream",
            event: event.event,
            eventCount,
            tokenEventCount,
            tokenChars
          });
          onEvent(event);
        }
        splitIndex = buffer.indexOf("\n\n");
      }
    }

    const tail = buffer.trim();
    if (tail) {
      const event = parseSSEChunk(tail);
      if (event) {
        eventCount += 1;
        if (event.event === "token") {
          tokenEventCount += 1;
          tokenChars += event.token.length;
        }
        onEvent(event);
      }
    }
    const elapsedMs = performance.now() - streamStarted;
    console.info("[REST][stream_done]", {
      tag: "chat_stream",
      sessionId: body.session_id,
      eventCount,
      tokenEventCount,
      tokenChars,
      elapsedMs: Number(elapsedMs.toFixed(2))
    });
  }
}


export class OpsApiClient extends BaseApiClient {
  async startEmbeddingJob(repoPath: string, repoName: string): Promise<EmbeddingJob> {
    const response = await this.request("/embedding/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_path: repoPath, repo_name: repoName })
    }, "embedding_job_start");
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Failed to start embedding job: ${detail || response.status}`);
    }
    return (await response.json()) as EmbeddingJob;
  }

  async listEmbeddingJobs(limit = 20): Promise<EmbeddingJob[]> {
    const response = await this.request(`/embedding/jobs?limit=${limit}`, {}, "embedding_jobs_list");
    if (!response.ok) {
      throw new Error(`Failed to list embedding jobs: ${response.status}`);
    }
    const payload = (await response.json()) as { jobs?: EmbeddingJob[] };
    return payload.jobs ?? [];
  }

  async getEmbeddingJob(jobId: string): Promise<EmbeddingJob> {
    const response = await this.request(`/embedding/jobs/${encodeURIComponent(jobId)}`, {}, "embedding_job_get");
    if (!response.ok) {
      throw new Error(`Failed to fetch embedding job ${jobId}: ${response.status}`);
    }
    return (await response.json()) as EmbeddingJob;
  }

  async getEvaluationRules(): Promise<EvaluationRulesPayload> {
    const response = await this.request("/evaluation/rules", {}, "evaluation_rules_get");
    if (!response.ok) {
      throw new Error(`Failed to load evaluation rules: ${response.status}`);
    }
    return (await response.json()) as EvaluationRulesPayload;
  }

  async saveEvaluationRules(payload: EvaluationRulesPayload): Promise<EvaluationRulesPayload> {
    const response = await this.request("/evaluation/rules", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }, "evaluation_rules_put");
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Failed to save evaluation rules: ${detail || response.status}`);
    }
    return (await response.json()) as EvaluationRulesPayload;
  }

  async startEvaluationJob(repoName: string, repoPath: string): Promise<EvaluationJob> {
    const response = await this.request("/evaluation/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_name: repoName, repo_path: repoPath })
    }, "evaluation_job_start");
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Failed to start evaluation job: ${detail || response.status}`);
    }
    return (await response.json()) as EvaluationJob;
  }

  async listEvaluationJobs(limit = 20): Promise<EvaluationJob[]> {
    const response = await this.request(`/evaluation/jobs?limit=${limit}`, {}, "evaluation_jobs_list");
    if (!response.ok) {
      throw new Error(`Failed to list evaluation jobs: ${response.status}`);
    }
    const payload = (await response.json()) as { jobs?: EvaluationJob[] };
    return payload.jobs ?? [];
  }

  async getEvaluationJob(jobId: string): Promise<EvaluationJob> {
    const response = await this.request(`/evaluation/jobs/${encodeURIComponent(jobId)}`, {}, "evaluation_job_get");
    if (!response.ok) {
      throw new Error(`Failed to fetch evaluation job ${jobId}: ${response.status}`);
    }
    return (await response.json()) as EvaluationJob;
  }

  getEvaluationReportHtmlUrl(jobId: string): string {
    return this.url(`/evaluation/jobs/${encodeURIComponent(jobId)}/html`);
  }

  getEvaluationReportJsonUrl(jobId: string): string {
    return this.url(`/evaluation/jobs/${encodeURIComponent(jobId)}/json`);
  }
}
