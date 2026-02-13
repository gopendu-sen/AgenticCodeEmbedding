# Code RAG Agentic RAG

Agentic code-ingestion and retrieval system for multi-repo RAG.
It indexes source code into Chroma with repo tags, then serves citation-grounded chat through FastAPI + React.

## What Is In This Repo
- `agentic_rag/`: ingestion pipeline (parser, orchestrator, embedding, config).
- `retreiving_module/`: designated retrieval layer and embedding-job orchestration (`StoreRetriever`).
- `chat_module/`: FastAPI API layer + persistent chat session memory (SQLite).
- `web_ui/`: React + Vite + TypeScript frontend.

## Core Capabilities
- Deterministic parsing across common enterprise stacks (Python, JS/TS, Java, Kotlin, Scala, C#, Go, C/C++, SQL, COBOL, R, Markdown, Jupyter notebooks).
- Unconditional parser-error fallback path to LLM tool-plan parsing.
- Security/audit node tagging (`auth_guard`, `policy_check`, `audit_log`, `sensitive_op`).
- Repo-tagged embeddings (`metadata.repo_name`) for multi-repo isolation.
- Retrieval intent routing across split collections:
  - `code_symbols`, `code_routes`, `code_usage`, `configs`, `docs`, `security_tags`, `flows`.
- Chat responses with strict source citations (`[1]`, `[2]`, ...).
- Persistent chat memory in SQLite.
- Background embedding jobs with JSON status + logs.
- Background evaluation jobs with HTML + JSON audit reports.
- Retrieval and ingestion audit logs in `reports/`.

## Runtime Architecture
```text
config.yml
  -> agentic_rag.core.config_loader.load_agentic_rag_config
  -> chat_module.api (FastAPI)
       -> retreiving_module.StoreRetriever
            -> ChromaStore + EmbeddingClient + LLMClient
       -> SessionService/SessionStore (SQLite memory)
  -> web_ui (React SPA)
       -> REST + SSE calls to FastAPI
```

## Data Outputs
- Metadata DB: `paths.sqlite_path`
- Chroma vectors: `paths.chroma_dir`
- Ingestion report JSON: `paths.reports_dir/ingestion_report_<timestamp>.json`
- Embedding jobs:
  - status: `paths.reports_dir/embedding_jobs/<job_id>.json`
  - worker log: `paths.reports_dir/embedding_jobs/<job_id>.log`
  - per-file outcomes:
    - `files[]` with status values: `success`, `success_with_llm`, `error`, `failed`, `skipped_unchanged`
    - `partial=true` when file had partial processing (for example deterministic success with LLM/security errors)
    - `file_status_counts` rollup and top-level job status `completed` or `completed_partial`
- Retrieval log: `paths.reports_dir/retrieve_log.jsonl`
- Evaluation jobs:
  - status: `evaluation.jobs_dir/<job_id>.json`
  - worker log: `evaluation.jobs_dir/<job_id>.log`
- Evaluation reports:
  - HTML: `evaluation.reports_dir/evaluation_report_<repo>_<timestamp>.html`
  - JSON: `evaluation.reports_dir/evaluation_report_<repo>_<timestamp>.json`
- Evaluation run log: `evaluation.log_jsonl_path`
- Chat session memory DB: `chat.memory.sqlite_path`

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Frontend dependencies:
```bash
cd web_ui
npm install
cd ..
```

## Run (Step by Step)
1. Configure `config.yml`:
- `paths.repo_path`, `paths.chroma_dir`, `paths.reports_dir`
- `llm.base_url`, `llm.model`
- `embedding.base_url`, `embedding.model`
- `evaluation.rules_json_path`, `evaluation.jobs_dir`, `evaluation.reports_dir`, `evaluation.log_jsonl_path`
- `chat.api.host`, `chat.api.port`, `chat.api.cors_allowed_origins`
- `chat.memory.sqlite_path`

2. Start backend:
```bash
python3 -m chat_module.api --config config.yml
```
`chat_module` loads endpoint/runtime settings from `config.yml` (`chat.api.*`, `llm.*`, `embedding.*`, `chat.memory.*`).

3. Start frontend:
```bash
cd web_ui
npm run dev
```
Vite proxy target defaults to `http://127.0.0.1:8005`.
You can still override proxy target explicitly:
```bash
VITE_PROXY_TARGET=http://127.0.0.1:8005 npm run dev
```

4. Open UI:
- `http://localhost:5173`

5. Start embedding from GUI:
- In **Embedding Jobs** panel:
  - set `Repo path` (absolute/local path to code repo)
  - set `Repo store tag` (repo name used for retrieval filtering)
  - click `Start Embedding Job`
- Job is asynchronous; UI does not block.
- Click `Refresh` in **Embedding Jobs** to check progress.
- After completion, click `Refresh` in **Repo Stores**.

6. Start evaluation report from GUI:
- In **Evaluation Reports** panel:
  - select one `Repo store` (or type one when discovery is empty)
  - set required `Repo path`
  - click `Start Evaluation Job`
- Job runs asynchronously.
- Click `Refresh` in **Evaluation Reports** to update status.
- For completed jobs, use:
  - `Open HTML` for auditor-readable report
  - `Download JSON` for machine-readable evidence output
- In **Evaluation Rules JSON**, use:
  - `Load` to fetch current global rules
  - `Save` to persist edits
  - `Import`/`Export` for JSON file exchange

7. Chat:
- Select at least one repo store in **Repo Stores** (or manual fallback if discovery is empty).
- Ask your question.
- Sources panel shows citation-to-source mapping (repo, file, line range, node type, collection, distance).

## Optional CLI Ingestion
Use this if you want direct ingestion outside UI jobs:
```bash
python3 ingest.py --config config.yml --repo-name <repo_store_tag>
```

`--repo-name` is required and becomes `metadata.repo_name` on all embedded records.

## Retrieval and Citation Contract
`StoreRetriever` is the single retrieval engine used by API/UI.

Per chat turn:
1. Embed query once.
2. Detect intent via `chat.retrieval.intent_keywords`.
3. Compute weighted budgets from `chat.retrieval.base_top_k` and `chat.retrieval.intent_weights`.
4. Query configured collections with repo filter:
- `where={"repo_name": <selected_store>}`
5. Merge + rank by distance, cap by `chat.max_context_chunks`.
6. Build LLM prompt with retrieved context and citation rules.
7. Stream response tokens through SSE.
8. Enforce citations if model omits them.
9. Persist structured retrieve log event.

No-source behavior:
- If no chunks are retrieved, backend returns deterministic no-source guidance and sets `no_sources=true`.
- This prevents generic “I cannot access repo” model replies.

## FastAPI Endpoints
- `GET /health`
- `GET /ui-config`
- `GET /stores`
- `POST /chat` (SSE `text/event-stream`)
- `GET /sessions`
- `GET /history/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /embedding/jobs`
- `GET /embedding/jobs?limit=N`
- `GET /embedding/jobs/{job_id}`
- `GET /evaluation/rules`
- `PUT /evaluation/rules`
- `POST /evaluation/jobs`
- `GET /evaluation/jobs?limit=N`
- `GET /evaluation/jobs/{job_id}`
- `GET /evaluation/jobs/{job_id}/html`
- `GET /evaluation/jobs/{job_id}/json`

### `/chat` SSE Events
- `meta`: intent, selected stores, initial sources
- `token`: token chunk
- `done`: final response + citations + sources
- `error`: safe failure message

## Session Memory Model (SQLite)
Tables:
- `chat_sessions(session_id, summary, intents_json, created_at, updated_at)`
- `chat_messages(id, session_id, role, content, sources_json, cited_indices_json, intent, created_at)`

Behavior:
- User and assistant turns are persisted.
- History survives backend restarts.
- Optional summarization/intent tracking is controlled by:
  - `chat.memory.enable_summarisation`
  - `chat.memory.enable_intent_tracking`

## Embedding Node Routing
- `code_symbols`: `function`, `class`, `component`, `table`, `view`, `query`, `notebook_code`
- `code_routes`: `route`, `entrypoint`
- `code_usage`: `call_edge`, `import_usage`
- `configs`: `config`
- `docs`: `doc`, `doc_code`, `comment_line`, `comment_block`, `comment_doc`, `comment_other`
- `security_tags`: `auth_guard`, `policy_check`, `audit_log`, `sensitive_op`
- `flows`: `flow_chain` (reserved/phase-2)

Each vector metadata includes:
- `repo_name`, `file_path`, `start_line`, `end_line`, `node_type`, `language`, `symbol`, `confidence`

## Logging and Reports
- Retrieval logs: `reports/retrieve_log.jsonl`
  - includes `session_id`, query, intent, selected stores, source list, cited indices, latency.
- Embedding job logs/status:
  - `reports/embedding_jobs/*.log`
  - `reports/embedding_jobs/*.json`
- Evaluation logs/status:
  - `reports/evaluation_jobs/*.log`
  - `reports/evaluation_jobs/*.json`
- Evaluation artifacts:
  - `reports/evaluations/*.html`
  - `reports/evaluations/*.json`
- Evaluation run log:
  - `reports/evaluation_log.jsonl`
- REST/API logging is verbose by design:
  - backend logs inbound REST request start/end, status, latency, and key request metadata
  - backend logs outbound LLM/embedding REST call start/end, status, latency, and token usage when available
  - frontend logs REST and SSE stream lifecycle in browser devtools console
- Ingestion report JSON includes summary metrics, collection totals, node-type totals, and token usage sections.

## Config Rules
- Source of truth: `config.yml`
- Env overrides: `AGENTIC_RAG__...` keys only
- Relative `paths.*`, `chat.memory.sqlite_path`, and `evaluation.*` path fields resolve relative to config file directory

## Migration Notes
- Streamlit was removed from runtime.
- New runtime is FastAPI + React.
- If your vector store contains old records without `repo_name`, run a clean reindex for strict repo filtering.

## Troubleshooting
1. UI says no sources:
- check embedding job status is `completed` or `completed_partial`
- refresh repo stores
- confirm selected store tag matches embedding job `repo_name`
- inspect latest line in `reports/retrieve_log.jsonl`

2. `ModuleNotFoundError: fastapi`:
- install Python dependencies: `pip install -r requirements.txt`
- ensure you are in the same Python env used to run backend

3. CORS errors from browser:
- add frontend origin to `chat.api.cors_allowed_origins` in `config.yml`

4. Old/generic answers without citations:
- verify retrieval source count in `retrieve_log.jsonl`
- if `source_count=0`, re-embed repo and retry

5. Vite proxy `ECONNREFUSED` (for `/ui-config`, `/stores`, `/sessions`, `/embedding/jobs`, `/evaluation/*`):
- backend is not reachable on `127.0.0.1:8005`
- start backend first:
```bash
python3 -m chat_module.api --config config.yml
```
- then start/restart frontend:
```bash
cd web_ui
npm run dev
```

6. Evaluation job fails quickly:
- verify selected repo store exists in indexed metadata
- confirm repo path is accessible from backend process
- inspect:
  - `reports/evaluation_jobs/<job_id>.json`
  - `reports/evaluation_jobs/<job_id>.log`
  - `reports/evaluation_log.jsonl`
