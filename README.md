# Code RAG Agentic RAG

Deterministic + agentic ingestion pipeline for source repositories, with structured parsing, fallback LLM enrichment, vector indexing, and a Streamlit chat interface.

## Core Capabilities
- Incremental repository indexing (hash-based skip for unchanged files)
- Stack/framework detection (deterministic + optional LLM fallback)
- Multi-language parsing:
  - Python (AST)
  - JavaScript/TypeScript (including React/Angular patterns)
  - Java/Spring patterns
  - Kotlin/Kotlin-Spring/Ktor patterns
  - Scala patterns
  - C#/.NET patterns
  - Go patterns
  - COBOL program/section/paragraph patterns
  - R scripts
  - SQL scripts
  - C/C++ patterns
  - Jupyter notebooks (`.ipynb`)
  - Markdown docs (`.md`) with section-aware parsing
  - Config files and plain text docs
- Explicit node extraction for:
  - structural symbols (`class`, `function`, `component`, `route`, `entrypoint`)
  - data/repo symbols (`table`, `view`, `query`, `notebook_code`)
  - usage nodes (`call_edge`, `import_usage`)
  - security nodes (`auth_guard`, `policy_check`, `audit_log`, `sensitive_op`)
  - comments (`comment_line`, `comment_block`, `comment_doc`, `comment_other`)
  - markdown sections and fenced code blocks (`doc`, `doc_code`)
  - flow nodes (`flow_chain`, reserved for phase-2 materialization)
- Optional parser LLM fallback (tool-plan flow) when deterministic confidence is low or parser errors occur
- Storage:
  - SQLite for metadata and node snapshots
- Chroma for embeddings
- Streamlit chatbot with conversation memory and source-grounded retrieval
- Verbose embedding logs and timestamped JSON ingestion reports

## Package Architecture
```text
agentic_rag/
  agentic_ai/
    orchestrator.py      # top-level ingestion workflow
    stack_detect.py      # stack inference
    tool_plan_agent.py   # LLM tool-plan fallback
  code_parser/
    deterministic.py     # language parsers + markdown parser + call-edge extraction
    service.py           # parser router + parser error fallback node handling
    node_builder.py      # converts LLM-final JSON to CodeNode list
    types.py             # CodeNode model
  embedding/
    client.py            # embeddings client (OpenAI-compatible endpoint)
    chroma_store.py      # chroma wrapper
    service.py           # embedding text strategy + upsert routing
  core/
    config.py            # typed config schema
    config_loader.py     # YAML + env override loader
    repo_tools.py        # repository IO/tool utilities
    sqlite_store.py      # sqlite persistence
    llm_client.py        # LLM chat client
    utils.py             # hashing/helpers
retreiving_module/
  service.py             # designated retrieval/chat service (discovery, retrieval, citation, retrieve_log)
```

## Runtime Workflow (Ingestion DAG)
```text
config.yml + AGENTIC_RAG__* env
            |
            v
     load_agentic_rag_config
            |
            v
    AgenticRagOrchestrator.run
            |
            +--> StackDetector.detect/confidence
            |       |
            |       +--> optional llm_stack_fallback
            |
            +--> RepoTools.iter_files
                    |
                    +--> for each changed file (hash gate via SQLite)
                            |
                            +--> CodeParserService.parse_file
                            |       |
                            |       +--> deterministic parser path
                            |       +--> parser-error fallback node
                            |
                            +--> optional ToolPlanParseAgent fallback
                            |
                            +--> SQLiteStore.upsert_node
                            |
                            +--> EmbeddingService.upsert_nodes
                                    |
                                    +--> EmbeddingClient.embed
                                    +--> ChromaStore.upsert
                                    +--> verbose embedding logs
                            |
                            +--> write timestamped JSON report in reports/
```

## Design Patterns
- Orchestration pattern:
  - `AgenticRagOrchestrator` coordinates components, keeps business flow in one place.
- Strategy pattern:
  - parser routing by extension/language in `CodeParserService`.
  - embedding routing by node type in `EmbeddingService`.
- Adapter pattern:
  - `RepoTools`, `SQLiteStore`, `ChromaStore`, `LLMClient`, `EmbeddingClient` isolate external systems.
- Config-driven runtime:
  - no behavior should depend on ad hoc runtime constants; knobs come from `config.yml` (+ namespaced env overrides).

## Parsing Model
All extracted artifacts are stored as `CodeNode` with:
- `node_id`, `node_type`, `language`, `file_path`
- `start_line`, `end_line`, `symbol`
- `text`, `confidence`, `metadata`

### Node Types Produced
- Code structure: `file`, `class`, `function`, `component`, `route`, `entrypoint`
- Data/analytics/codeflow: `table`, `view`, `query`, `notebook_code`, `flow_chain`
- Usage graph: `call_edge`, `import_usage`
- Security: `auth_guard`, `policy_check`, `audit_log`, `sensitive_op`
- Documentation/context: `doc`, `doc_code`, `comment_line`, `comment_block`, `comment_doc`, `comment_other`, `config`

Security-tag nodes emitted by the LLM tagging pass carry explicit metadata:
- `source=llm_security_tagger`
- `why_relevant`
- `control_area`
- `severity`
- `evidence`

### Call Hierarchy / DAG Capture
- Python:
  - AST-based caller/callee extraction from function/method scopes.
- JS/TS, Java, C#, C/C++:
  - scope-constrained call token extraction within parsed symbols.
- Kotlin/Scala/R/Go:
  - deterministic symbol extraction + scoped call token edges.
- COBOL:
  - program/section/paragraph symbol extraction + `CALL` statement edges.
- SQL:
  - deterministic query/object extraction + reference edges (query/routine to tables/views/procedures).
- Output:
  - `call_edge` nodes with metadata `{relation: "calls", caller, callee}`.
- Config knob:
  - `parser.call_graph.max_callees_per_scope`

## Markdown (`.md`) Parsing and Embedding Strategy
Markdown is parsed with a dedicated strategy (not plain text chunking):

1. Heading-aware sectioning
- Extract heading hierarchy (`#`..`######`) and section path.
- Build section chunks using `parser.markdown.chunk_lines` and `parser.markdown.overlap_lines`.
- Preserve section metadata in each node:
  - `doc_title`, `section_title`, `section_path`, `section_level`, `doc_format=markdown`

2. Fenced code block extraction
- Extract fenced code blocks (triple-backtick fences or `~~~`) into `doc_code` nodes.
- Capture language hint in metadata: `code_language`.

3. Embedding strategy for markdown
- `doc` and `doc_code` nodes are routed to the docs collection.
- Embedding payload includes:
  - file path + lines
  - document title + section path/title
  - markdown format marker
  - code language (for fenced blocks)
  - chunk text
- This improves retrieval quality for:
  - architecture docs
  - runbooks
  - API guides
  - README-driven navigation

## Notebook (`.ipynb`) Parsing Strategy
- Notebook JSON is parsed cell-by-cell.
- Markdown cells:
  - chunked into `doc` nodes for retrieval.
- Code cells:
  - indexed as `notebook_code` nodes.
  - Python code cells also extract `function`/`class` symbols and `call_edge` relations.
- Notebook parse errors are captured in node metadata (`parse_error`) so fallback logic can still activate.

## Parser Fallback Behavior
- Deterministic parser errors generate explicit parse-error metadata.
- If LLM fallback is enabled, parser-error files always trigger fallback attempts.
- Low-confidence fallback remains supported via configured thresholds and importance heuristics.

## Configuration
Runtime config source of truth is `config.yml`.

### Loading Rules
- Default file: `./config.yml`
- Override file: `python3 ingest.py --config /path/to/config.yml --repo-name <repo_tag>`
- Relative `paths.*` values resolve relative to the config file directory.

### Environment Override Contract
Only namespaced keys are allowed:
- `AGENTIC_RAG__SECTION__KEY`
- `AGENTIC_RAG__SECTION__SUBSECTION__KEY`

Examples:
```bash
export AGENTIC_RAG__PATHS__REPO_PATH="/path/to/repo"
export AGENTIC_RAG__EMBEDDING__BASE_URL="http://localhost:11434/v1"
export AGENTIC_RAG__PARSER__CALL_GRAPH__MAX_CALLEES_PER_SCOPE="32"
export AGENTIC_RAG__PARSER__MARKDOWN__CHUNK_LINES="120"
export AGENTIC_RAG__CHAT__RETRIEVAL__BASE_TOP_K="4"
export AGENTIC_RAG__SECURITY_TAGGING__ENABLED="true"
```

Type casting behavior:
- `int`, `float`, `bool` cast to target field types
- `list` / `dict` overrides must be JSON strings
- strings kept as-is

### Logging/Report Settings
- `paths.reports_dir`: directory where ingestion JSON reports are written.
- `logging.level`: runtime logger level (`DEBUG`, `INFO`, `WARNING`, etc.).
- `logging.embedding_verbose_per_node`: when `true`, logs every node-to-collection embedding mapping.
- `chat.store_discovery.sample_size_per_collection`: metadata records sampled per collection to discover repo stores.
- `chat.store_discovery.max_store_names`: max unique repo store names returned to the UI.
- Orchestrator logs include file-by-file processing, parse confidence, LLM fallback decisions, security tagger attempts, embedding flushes, and final summary/report path.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## User Step-By-Step (From Scratch)
1. Prepare config:
- Open `config.yml` and confirm endpoints:
  - `llm.base_url`, `llm.model`
  - `embedding.base_url`, `embedding.model`
  - `paths.chroma_dir`, `paths.sqlite_path`, `paths.reports_dir`

2. Start the GUI:
```bash
streamlit run streamlit_app.py
```

3. Load runtime config in GUI:
- In sidebar, set `Config path` (default `config.yml`).

4. Embed a repository from GUI:
- In sidebar section `Embed Repository`:
  - set `Repo path for embedding` (local repo you want indexed)
  - set `Repo store tag` (for example `payments_repo`)
  - click `Start Embedding Job`
- This starts a background job and returns immediately (GUI does not block).
- Click `Refresh Embedding Jobs` to see current status (`running` / `completed` / `failed`).
- After job completion, click `Refresh Repo Stores` to load new repo tags for chat retrieval.

5. Select retrieval scope:
- In sidebar `Repo Stores`, choose one or more repo tags.
- If discovery is empty, enter tags manually in `Repo Stores (comma-separated)`.

6. Ask questions in chat:
- Type prompts in chat input.
- Responses include source citations like `[1]`, `[2]`.
- Open `Sources` to inspect cited files/lines and repo tags.

7. Check logs/reports:
- Retrieval logs: `reports/retrieve_log.jsonl`
- Ingestion report: `reports/ingestion_report_<timestamp>.json`
- Background embedding worker logs/status: `reports/embedding_jobs/<job_id>.log` and `reports/embedding_jobs/<job_id>.json`
- Token usage in ingestion report:
  - `summary.chat_token_usage` and `summary.embedding_token_usage`
  - `token_usage.chat`, `token_usage.chat_breakdown`, `token_usage.embedding`
  - `embedding.batch_events[*].embed_input_tokens|embed_output_tokens|embed_total_tokens`

## Troubleshooting Chat Answers
If the assistant gives generic model answers (for example, "I cannot access your repository"), check these first:

1. Verify retrieval actually returned sources:
- Open `reports/retrieve_log.jsonl`
- Confirm latest event has `source_count > 0`

2. Verify embedding finished successfully:
- Open `reports/embedding_jobs/<job_id>.json`
- Status must be `completed` (not `failed`)

3. If `source_count` is `0`:
- Re-run embedding for the same `repo_name` tag
- Click `Refresh Embedding Jobs` until completed
- Click `Refresh Repo Stores`
- Ask the question again

Current runtime behavior:
- When retrieval returns zero sources, chat now skips LLM answering and returns an explicit "no indexed context found" response with next steps.
- LLM parser fallback failures during ingestion no longer abort the entire run; deterministic nodes continue to index and the run completes with failure counters in the report.

## User Step-By-Step (CLI Ingestion + GUI Chat)
1. Run ingestion:
```bash
python3 ingest.py --config config.yml --repo-name payments_repo
```

2. Start GUI:
```bash
streamlit run streamlit_app.py
```

3. Select `payments_repo` from `Repo Stores` and start chatting.

## How To Run
### Ingestion
```bash
python3 ingest.py --repo-name repo_a
# or
python3 ingest.py --config /absolute/path/to/config.yml --repo-name repo_a
```

### Chat UI
```bash
streamlit run streamlit_app.py
```

In the sidebar:
- set config path if it is not `config.yml`
- optionally start embedding as background job from `Embed Repository`
- select repo stores for retrieval.

## How Chat Works
1. Streamlit initializes `StoreRetriever` from `retreiving_module` and only handles UI input/output.
2. `StoreRetriever.discover_store_names()` scans collection metadata for `repo_name`.
3. User selects one or more repo stores in the sidebar (manual input fallback is shown if discovery is empty).
4. Embedding jobs from GUI are started asynchronously by `StoreRetriever.start_embedding_job(...)` and monitored via `StoreRetriever.list_embedding_jobs(...)`.
5. `StoreRetriever.chat_turn(...)` runs end-to-end:
  - embed query
  - detect intent
  - query weighted collections with repo filter `where={"repo_name": <selected_store>}`
  - rank/cap context
  - build LLM prompt
  - enforce citation rules and fallback citation footer if model omits citations
  - append structured retrieval event to `retrieve_log`
6. Build prompt with:
  - configured system prompt
  - explicit citation rules (`[1]`, `[2]`, ...)
  - detected retrieval intent
  - active repo stores
  - retrieved snippets (with collection and repo provenance)
  - rolling conversation history (`chat.history_messages`)
7. Render answer + source panel (source ids match citation ids).

Conversation memory is maintained in Streamlit session state.

## Embedding Output Contract
Each parsed `CodeNode` is transformed into one Chroma record.

### Collection Routing
- `code_symbols` collection:
  - `function`, `class`, `component`, `table`, `view`, `query`, `notebook_code`
- `code_routes` collection:
  - `route`, `entrypoint`
- `code_usage` collection:
  - `call_edge`, `import_usage`
- `docs` collection:
  - `doc`, `doc_code`, `comment`, `comment_line`, `comment_block`, `comment_doc`, `comment_other`
- `configs` collection:
  - `config`
- `security_tags` collection:
  - `auth_guard`, `policy_check`, `audit_log`, `sensitive_op`
- `flows` collection:
  - `flow_chain` (phase-2 population target)

### Chroma Record Shape
For every upserted node, the payload is:
- `id`:
  - `node_id` (stable deterministic identifier)
- `embedding`:
  - vector from `EmbeddingClient.embed(embed_text)`
- `document`:
  - `node.text` truncated to `io_limits.embed_doc_max_chars`
- `metadata`:
  - `repo_name`
  - `file_path`
  - `start_line`
  - `end_line`
  - `node_type`
  - `language`
  - `symbol`
  - `confidence`

### What Gets Embedded (`embed_text`)
The embedded text is not raw code only. It is a structured prompt-like payload:
- Common header for all nodes:
  - `TYPE`, `LANG`, `FILE`, `LINES`, optional `SYMBOL`
- `doc` / `doc_code`:
  - includes `DOC_TITLE`, `SECTION_PATH`, `SECTION_TITLE`, `DOC_FORMAT`, `CODE_LANG`, then `DOC_CONTENT`
- `call_edge`:
  - includes `CALLER`, `CALLEE`, `RELATION`, then `CODE_CONTEXT`
- Other code-like nodes:
  - includes optional `WHY` metadata, then `CODE`

### Embedding Telemetry Output
Each ingestion run writes a report JSON at:
- `paths.reports_dir/ingestion_report_<UTC timestamp>.json`

The `embedding` section includes:
- `upsert_calls`
- `total_nodes_received`
- `total_nodes_upserted`
- `total_batches`
- `total_embed_seconds`
- `total_upsert_seconds`
- `total_elapsed_seconds`
- `collection_totals`
- `batch_events` with per-batch durations

## Retrieval Usage Guide
This is the exact retrieval path used by `streamlit_app.py`:

1. Query embedding:
- embed user prompt once with `EmbeddingClient`

2. Repo-store selection:
- repo stores are repo tags (`metadata.repo_name`)
- user selection is mandatory in Streamlit
- if discovery is empty, manual comma-separated store input is used

3. Multi-collection search:
- detect query intent from `chat.retrieval.intent_keywords`
- compute weighted per-collection top-k from `chat.retrieval.base_top_k` and `chat.retrieval.intent_weights`
- query weighted collections only: `code_symbols`, `code_routes`, `code_usage`, `configs`, `docs`, `security_tags`, `flows`
- apply metadata filter per selected store: `where={"repo_name": store_name}`

4. Rank and cap:
- merge all hits and sort by ascending distance
- keep first `chat.max_context_chunks`

5. Build model context:
- context block includes `repo_name`, `file_path`, line range, `node_type`, `language`, `symbol`, and retrieved `document`
- total context text is capped by `chat.max_context_chars`

6. LLM answer + citation enforcement:
- send system prompt + retrieved context + rolling history (`chat.history_messages`)
- enforce bracketed source citations (`[1]`, `[2]`, ...)
- if no citation appears but sources exist, append fallback `Citations: [1] ...`

7. Retrieve logging:
- each chat turn is appended to `paths.reports_dir/retrieve_log.jsonl`
- includes query, selected repo stores, intent, source list, cited indices, model, latency

### How To Use This Output For Better Retrieval
- Architecture and API flow questions:
  - prioritize `code_symbols`, `code_routes`, and `code_usage`
- Security enforcement questions:
  - prioritize `security_tags`, `code_routes`, and `code_usage`
- Runbook or design intent questions:
  - prioritize `docs` hits (`doc`, `doc_code`, comment nodes)
- Config and environment questions:
  - prioritize `configs` (and `docs` when runbooks describe config behavior)
- Improve recall:
  - increase `chat.retrieval.base_top_k` and/or intent weights
- Improve precision/latency:
  - reduce `chat.max_context_chunks` and `chat.max_context_chars`

## Outputs
- SQLite metadata DB: `paths.sqlite_path`
- Chroma persistent directory: `paths.chroma_dir`
- JSON report per ingestion run: `paths.reports_dir/ingestion_report_<UTC timestamp>.json`
- Retrieval log (JSONL): `paths.reports_dir/retrieve_log.jsonl`
- Report includes:
  - summary metrics
  - `summary.node_type_counts`
  - `summary.chroma_counts`
  - `summary.embedding_collection_totals_run`
  - `summary.security_tags_generated`
  - `summary.security_tagger_attempts`
  - `summary.security_tagger_failures`
  - per-batch embedding telemetry (embed/upsert durations, batch sizes, collection totals)
  - run timestamps and config snapshot

## Operational Notes
- LLM is mandatory for this runtime. `llm.enabled` must be `true`.
- Ingestion probes the configured LLM endpoint at startup and fails fast if unavailable.
- Ingestion requires explicit repo tag: `--repo-name <value>`.
- GUI embedding jobs are non-blocking and run in background worker processes.
- Use `Refresh Embedding Jobs` and `Refresh Repo Stores` in the sidebar to update visibility of newly indexed repos.
- LLM/embedding request timeouts are controlled by config (`llm.timeout_s`, `embedding.timeout_s`, `security_tagging.llm_timeout_s`) and apply inside background embedding workers too.
- Existing vectors without `metadata.repo_name` are not repo-store filterable.
- This schema migration requires a full reindex from a clean vector store path:
  - remove old `paths.chroma_dir` data, then run ingestion.
- Incremental indexing skips unchanged files by hash.
- Parsing quality for incomplete code is improved via parser-error fallback + comment extraction + call-edge heuristics.
