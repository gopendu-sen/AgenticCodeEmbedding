# Vyom

Agentic code-ingestion and retrieval system for multi-repo RAG.
It indexes source code into Chroma with repo tags, then serves citation-grounded chat through FastAPI + React.

## What Is In This Repo
- `agentic_rag/`: ingestion pipeline (parser, orchestrator, embedding, config).
- `retreiving_module/`: designated retrieval layer + job orchestration (`StoreRetriever`).
- `chat_module/`: chat/session FastAPI API layer + persistent chat session memory (SQLite).
- `ops_module/`: embedding/evaluation FastAPI API layer + embedding proxy (`/v1/embeddings`).
- `web_ui/`: React + Vite + TypeScript frontend.

## Core Capabilities
- Deterministic parsing across common enterprise stacks (Python, JS/TS, Java, Kotlin, Scala, C#, Go, C/C++, SQL, COBOL, R, Markdown, Jupyter notebooks).
- Unconditional parser-error fallback path to LLM tool-plan parsing.
- Security/audit node tagging (`auth_guard`, `policy_check`, `audit_log`, `sensitive_op`).
- Repo-tagged embeddings (`metadata.repo_name`) for multi-repo isolation.
- Retrieval intent routing across split collections:
  - Base stores: `code_symbols`, `code_routes`, `code_usage`, `configs`, `docs`, `security_tags`, `flows`
  - Audit-dimension stores: `audit_identity_profile`, `audit_auth_controls`, `audit_money_movement`,
    `audit_payee_recipient`, `audit_docs_disclosures`, `audit_limits_access`
- Chat responses with strict source citations (`[1]`, `[2]`, ...).
- Persistent chat memory in SQLite.
- Background embedding jobs with JSON status + logs.
- Background evaluation jobs with HTML + JSON audit reports.
- Retrieval and ingestion audit logs in `reports/`.
- Built-in retry for LLM REST failures (timeouts/connection errors/retryable HTTP status codes).

## Runtime Architecture
```text
config.chat.yml
  -> chat_module.api (FastAPI :8025)
       -> retreiving_module.StoreRetriever (retrieval + store discovery)
            -> ChromaStore + EmbeddingClient(base_url=http://127.0.0.1:8026/v1)
       -> SessionService/SessionStore (SQLite memory)
config.embedding.yml
  -> ops_module.api (FastAPI :8026)
       -> retreiving_module.StoreRetriever (embedding/evaluation jobs + rules)
       -> /v1/embeddings passthrough to upstream embedding provider
  -> web_ui (React SPA)
       -> chat REST + SSE to chat service
       -> embedding/evaluation REST to ops service
```

## Architecture Details
- Chat service (`chat_module`) owns conversational APIs only:
  - `/chat` SSE streaming
  - session/history lifecycle
  - UI config and store discovery for the UI
- Ops service (`ops_module`) owns operational APIs:
  - embedding jobs
  - evaluation rules + evaluation jobs + reports
  - OpenAI-compatible embeddings proxy (`POST /v1/embeddings`)
- Data stores are shared between services:
  - Chroma (`paths.chroma_dir`) for vector data
  - SQLite (`paths.sqlite_path`) for file/node metadata
  - `reports/` for ingestion/evaluation/job status artifacts
- Embedding flow split:
  - chat retrieval embeds the query through `EmbeddingClient -> ops /v1/embeddings`
  - ops `/v1/embeddings` forwards to upstream embedding provider
  - no direct chat fallback to upstream provider if ops is down (fail-fast)

## How It Works
1. You run an embedding job from ops UI/API.
2. Orchestrator scans selected repository files and parses them into nodes.
3. Before re-indexing a changed file, stale vectors and stale SQLite nodes for that file are deleted.
4. Nodes are routed to one or many embedding stores (base + audit dimensions).
5. Vectors are upserted to Chroma with `repo_name` and file metadata.
6. Chat queries run against weighted stores filtered by selected repo tags.
7. Retrieved evidence is sent to the LLM with citation constraints and streamed back via SSE.
8. Evaluation jobs use the same retrieval system but rule-aware dimension preferences first.

## Working Flow (Runtime)
1. User starts embedding job (`/embedding/jobs`) from UI or API.
2. Worker process updates `reports/embedding_jobs/<job_id>.json` through stages:
   - `init` -> `config_loaded` -> `orchestrator_running` -> terminal (`completed`, `completed_partial`, `failed`)
3. Orchestrator writes ingestion report with per-store summary:
   - `summary.embedding_store_types`
   - `summary.embedding_store_summary[]`
   - `summary.parser_index_version`
4. User opens chat; chat service retrieves with intent weights and repo filtering.
5. If chat/eval LLM service call fails due retryable transport/server issue, client retries automatically.
6. User can run evaluation job; report JSON/HTML is generated under `reports/evaluations/`.

## Data Outputs
- Metadata DB: `paths.sqlite_path`
- Chroma vectors: `paths.chroma_dir`
- Ingestion report JSON: `paths.reports_dir/ingestion_report_<timestamp>.json`
- Embedding jobs:
  - status: `paths.reports_dir/embedding_jobs/<job_id>.json`
  - worker log: `paths.reports_dir/embedding_jobs/<job_id>.log`
  - status fields:
    - `status`: `starting | running | completed | completed_partial | failed`
    - `stage`: `init | config_loaded | orchestrator_running | completed | failed`
    - terminal timestamps are always written (`started_at_utc`, `finished_at_utc`, `updated_at_utc`) even on failures
  - per-file outcomes:
    - `files[]` with status values: `success`, `success_with_llm`, `error`, `failed`, `skipped_unchanged`
    - `partial=true` when file had partial processing (for example deterministic success with LLM/security errors)
    - `file_status_counts` rollup and top-level job status `completed` or `completed_partial`
  - report summary includes per-store outcome metrics:
    - `embedding_store_summary[]` (`store_key`, `store_type`, queued/upserted/deleted/final counts, status, error)
    - `embedding_store_types` (`base_stores[]`, `audit_dimension_stores[]`)
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
1. Keep `config.yml` for backward compatibility, then use split configs:
- `config.chat.yml`: chat service config (with `embedding.base_url: "http://127.0.0.1:8026/v1"`).
  - Chat bind settings are driven by YAML: `chat.api.host` and `chat.api.port`.
  - UI dev-server bind settings are driven by YAML: `chat.ui.host` and `chat.ui.port`.
- `config.embedding.yml`: ops service config (with upstream embedding provider in `embedding.base_url`).

2. Start ops service (embedding/evaluation + `/v1/embeddings` proxy):
```bash
python3 -m ops_module.api --config config.embedding.yml
```

3. Start chat service:
```bash
python3 -m chat_module.api --config config.chat.yml
```

4. Start frontend:
```bash
cd web_ui
npm run dev
```
Vite reads proxy targets and GUI port from YAML:
- chat target: `config.chat.yml -> chat.api.host/chat.api.port` (default `127.0.0.1:8025`)
- ops target: `config.embedding.yml -> chat.api.host/chat.api.port` (default `127.0.0.1:8026`)
- GUI server: `config.chat.yml -> chat.ui.host/chat.ui.port` (default `0.0.0.0:5173`)

Optional overrides:
```bash
VITE_CHAT_PROXY_TARGET=http://127.0.0.1:8025 \
VITE_EMBEDDING_PROXY_TARGET=http://127.0.0.1:8026 \
npm run dev
```
Legacy `VITE_PROXY_TARGET` is still supported for chat proxy fallback.
Frontend API base envs (optional):
- `VITE_CHAT_API_BASE_URL` (fallback: `VITE_API_BASE_URL`)
- `VITE_EMBEDDING_API_BASE_URL` (fallback: `VITE_API_BASE_URL`)

5. Open UI:
- `http://localhost:5173`

Windows one-click startup:
```bat
start_apps.bat
```

6. Start embedding from GUI:
- In **Embedding Jobs** panel:
  - set `Repo path` (absolute/local path to code repo)
  - set `Repo store tag` (repo name used for retrieval filtering)
  - click `Start Embedding Job`
- Job is asynchronous; UI does not block.
- Click `Refresh` in **Embedding Jobs** to check progress.
- After completion, click `Refresh` in **Repo Stores**.

7. Start evaluation report from GUI:
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

8. Chat:
- Select at least one repo store in **Repo Stores** (or manual fallback if discovery is empty).
- Ask your question.
- Sources panel shows citation-to-source mapping (repo, file, line range, node type, collection, distance).

## Optional CLI Ingestion
Use this if you want direct ingestion outside UI jobs:
```bash
python3 ingest.py --config config.embedding.yml --repo-name <repo_store_tag>
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

## How Rule-Driven Audit Works
The audit engine is rule-first and evidence-driven.

### Rule Source
- Rules are loaded from `retreiving_module/default_evaluation_rules.json` (or configured rules path).
- Contract per rule item:
  - `id`, `title`, `definition`
  - `strong_signals[]`
  - `weak_signals[]`
  - `false_positives[]`

### Candidate Retrieval Strategy
For each rule:
1. Build three retrieval prompts:
   - title + definition
   - joined strong signals
   - joined weak signals
2. Query preferred audit dimensions first (rule-id mapping), then base dimensions fallback.
3. Deduplicate candidates by `(file_path,start_line,end_line,node_type)`.
4. Score candidates using:
   - strong-signal hit count
   - weak-signal hit count
   - vector distance
5. Keep top-N candidates and extract bounded evidence snippets.

### Decisioning
- Signal hits are recomputed on selected evidences.
- LLM adjudicates each rule into one status:
  - `Detected`
  - `Not Detected`
  - `Needs Review`
- If LLM decision fails, item is marked `Needs Review` with failure reason (partial run).

### Outputs
- Per-rule output includes:
  - status, reason, confidence
  - matched strong/weak/false-positive signal lists
  - evidence snippets with citation ids
- Job-level outputs include:
  - summary status counts
  - total candidates
  - JSON + HTML reports for auditor and machine workflows

## Chat FastAPI Endpoints
- `GET /health`
- `GET /ui-config`
- `GET /stores`
- `POST /chat` (SSE `text/event-stream`)
- `GET /sessions`
- `GET /history/{session_id}`
- `DELETE /sessions/{session_id}`

## Ops FastAPI Endpoints
- `GET /health`
- `GET /stores`
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
- `POST /v1/embeddings`

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

## Embedding Dimensions
This release uses 13 dimensions: 7 base stores + 6 audit-dimension stores.

### Base Stores
- `code_symbols`: symbols and definitions (`function`, `class`, `table`, `query`, `notebook_code`)
- `code_routes`: routes and entrypoints
- `code_usage`: call edges/import usage
- `configs`: config-like nodes (`.yml/.yaml/.json/.toml/.ini/.xml/.properties`)
- `docs`: docs/markdown/comments/doc-code chunks
- `security_tags`: `auth_guard`, `policy_check`, `audit_log`, `sensitive_op`
- `flows`: flow-chain nodes

### Audit-Dimension Stores
- `audit_identity_profile`: customer/profile/PII/identity/account-linking signals
- `audit_auth_controls`: password/PIN/MFA/device/unlock/block-auth controls
- `audit_money_movement`: transfer/payment rails/arrangements/collections movement signals
- `audit_payee_recipient`: payee/beneficiary/recipient lifecycle signals
- `audit_docs_disclosures`: e-sign/document/disclosure/statement signals
- `audit_limits_access`: block/unblock/freeze/limit-up-down/access restoration signals

### Multi-Destination Routing
- A single node can be embedded into multiple stores.
- Base routing is always applied when node type matches.
- Audit-dimension routing is lexical + metadata-driven for high recall.

### Embedded Metadata and Text Enrichment
Each vector metadata includes:
- `repo_name`, `file_path`, `start_line`, `end_line`, `node_type`, `language`, `symbol`, `confidence`, `store_key`

Embedding text payload is enriched with selected metadata when present:
- `framework`, `annotations`, `attributes`, `sql_kind`, `sql_op`, `import_kind`, `entrypoint_kind`
- document section metadata (`section_*`, `doc_title`, `doc_format`, `code_language`)
- comment classification (`comment_kind`)

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
  - backend logs outbound LLM/embedding REST call start/end, status, latency, retry attempts, and token usage when available
  - frontend logs REST and SSE stream lifecycle in browser devtools console
- Ingestion report JSON includes summary metrics, collection totals, node-type totals, and token usage sections.

## Config Rules
- Source configs:
  - `config.yml` (legacy single-service compatible)
  - `config.chat.yml` (chat service)
  - `config.embedding.yml` (ops service)
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
- ensure you are in the same Python env used to run both services

3. CORS errors from browser:
- add frontend origin to `chat.api.cors_allowed_origins` in both `config.chat.yml` and `config.embedding.yml`

4. Old/generic answers without citations:
- verify retrieval source count in `retrieve_log.jsonl`
- if `source_count=0`, re-embed repo and retry

5. Vite proxy `ECONNREFUSED`:
- chat routes (`/ui-config`, `/stores`, `/sessions`, `/chat`) require chat service on `127.0.0.1:8025`
- ops routes (`/embedding/*`, `/evaluation/*`, `/v1/embeddings`) require ops service on `127.0.0.1:8026`
- start both services:
```bash
python3 -m ops_module.api --config config.embedding.yml
python3 -m chat_module.api --config config.chat.yml
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

7. Chat/evaluation LLM call still fails after retries:
- check upstream LLM endpoint health and credentials
- check network reachability from backend process host
- check logs for `attempt=<n>/<total>` retry messages in backend output
- increase `llm.timeout_s` if responses are timing out under load
