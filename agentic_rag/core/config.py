from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_path: str
    out_dir: str
    sqlite_path: str
    chroma_dir: str
    reports_dir: str


class FileSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_exts: List[str]
    exclude_dirs: List[str]
    max_file_size_bytes: int


class IOLimitsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    read_file_max_chars: int
    repo_tree_max_files: int
    tool_search_max_hits: int
    manifest_snippet_chars: int
    llm_header_preview_lines: int
    node_text_max_chars: int
    embed_doc_max_chars: int


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    base_url: Optional[str]
    model: Optional[str]
    timeout_s: int


class EmbeddingCollectionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code_symbols: str
    code_routes: str
    code_usage: str
    docs: str
    configs: str
    security_tags: str
    flows: str


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    model: str
    timeout_s: int
    batch_size: int
    collections: EmbeddingCollectionsConfig


class FallbackParseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    confidence_threshold: float
    allowed_exts: List[str]
    important_dir_hints: List[str]
    important_keywords: List[str]


class FallbackStackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    llm_min_confidence: float
    manifest_files: List[str]


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parse: FallbackParseConfig
    stack: FallbackStackConfig


class StackDetectionConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: float
    per_signal: float
    max: float


class StackDetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signals: Dict[str, List[str]]
    dotnet_project_suffixes: List[str]
    confidence: StackDetectionConfidenceConfig


class ParserChunkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_lines: int
    overlap_lines: int


class ParserGenericConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_confidence: float


class ParserMarkdownConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_lines: int
    overlap_lines: int


class ParserCallGraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_callees_per_scope: int


class ParserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    docs: ParserChunkConfig
    config: ParserChunkConfig
    markdown: ParserMarkdownConfig
    call_graph: ParserCallGraphConfig
    generic: ParserGenericConfig


class SecurityTaggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    supported_exts: List[str]
    llm_timeout_s: int
    max_header_lines: int

    @model_validator(mode="after")
    def _validate_security_tagging(self):
        if self.llm_timeout_s < 1:
            raise ValueError("security_tagging.llm_timeout_s must be >= 1")
        if self.max_header_lines < 1:
            raise ValueError("security_tagging.max_header_lines must be >= 1")
        if self.enabled and not self.supported_exts:
            raise ValueError("security_tagging.supported_exts must not be empty when security_tagging.enabled is true")
        return self


class ChatRetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_top_k: int
    intent_weights: Dict[str, Dict[str, float]]
    intent_keywords: Dict[str, List[str]]

    @model_validator(mode="after")
    def _validate_retrieval(self):
        if self.base_top_k < 1:
            raise ValueError("chat.retrieval.base_top_k must be >= 1")

        required_collections = {
            "code_symbols",
            "code_routes",
            "code_usage",
            "configs",
            "docs",
            "security_tags",
            "flows",
        }
        if not self.intent_weights:
            raise ValueError("chat.retrieval.intent_weights must define at least one intent profile")
        if "general" not in self.intent_weights:
            raise ValueError("chat.retrieval.intent_weights must define a 'general' profile")

        for intent, weights in self.intent_weights.items():
            missing = required_collections.difference(weights.keys())
            if missing:
                missing_csv = ", ".join(sorted(missing))
                raise ValueError(f"chat.retrieval.intent_weights.{intent} missing keys: {missing_csv}")
            for key, value in weights.items():
                if value < 0:
                    raise ValueError(f"chat.retrieval.intent_weights.{intent}.{key} must be >= 0")

        return self


class ChatStoreDiscoveryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_size_per_collection: int
    max_store_names: int

    @model_validator(mode="after")
    def _validate_store_discovery(self):
        if self.sample_size_per_collection < 1:
            raise ValueError("chat.store_discovery.sample_size_per_collection must be >= 1")
        if self.max_store_names < 1:
            raise ValueError("chat.store_discovery.max_store_names must be >= 1")
        return self


class ChatAPIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int
    cors_allowed_origins: List[str]

    @model_validator(mode="after")
    def _validate_api(self):
        if not self.host.strip():
            raise ValueError("chat.api.host must be a non-empty string")
        if self.port < 1 or self.port > 65535:
            raise ValueError("chat.api.port must be in range 1..65535")
        if not self.cors_allowed_origins:
            raise ValueError("chat.api.cors_allowed_origins must contain at least one origin")
        return self


class ChatMemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sqlite_path: str
    max_history_messages: int
    enable_summarisation: bool
    enable_intent_tracking: bool
    summarise_prompt: str
    intent_prompt: str

    @model_validator(mode="after")
    def _validate_memory(self):
        if not self.sqlite_path.strip():
            raise ValueError("chat.memory.sqlite_path must be a non-empty string")
        if self.max_history_messages < 1:
            raise ValueError("chat.memory.max_history_messages must be >= 1")
        if not self.summarise_prompt.strip():
            raise ValueError("chat.memory.summarise_prompt must be a non-empty string")
        if not self.intent_prompt.strip():
            raise ValueError("chat.memory.intent_prompt must be a non-empty string")
        return self


class ChatConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    subtitle: str
    assistant_greeting: str
    input_placeholder: str
    spinner_text: str
    system_prompt: str
    history_messages: int
    api: ChatAPIConfig
    memory: ChatMemoryConfig
    retrieval: ChatRetrievalConfig
    store_discovery: ChatStoreDiscoveryConfig
    max_context_chunks: int
    max_context_chars: int
    temperature: float
    show_sources: bool

    @model_validator(mode="after")
    def _validate_chat(self):
        if self.history_messages < 1:
            raise ValueError("chat.history_messages must be >= 1")
        if self.max_context_chunks < 1:
            raise ValueError("chat.max_context_chunks must be >= 1")
        if self.max_context_chars < 1:
            raise ValueError("chat.max_context_chars must be >= 1")
        return self


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str
    embedding_verbose_per_node: bool


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules_json_path: str
    jobs_dir: str
    reports_dir: str
    log_jsonl_path: str
    evidence_per_item: int
    max_candidates_per_item: int
    max_snippet_chars: int
    llm_timeout_s: int
    html_title: str

    @model_validator(mode="after")
    def _validate_evaluation(self):
        if not self.rules_json_path.strip():
            raise ValueError("evaluation.rules_json_path must be a non-empty string")
        if not self.jobs_dir.strip():
            raise ValueError("evaluation.jobs_dir must be a non-empty string")
        if not self.reports_dir.strip():
            raise ValueError("evaluation.reports_dir must be a non-empty string")
        if not self.log_jsonl_path.strip():
            raise ValueError("evaluation.log_jsonl_path must be a non-empty string")
        if self.evidence_per_item < 1:
            raise ValueError("evaluation.evidence_per_item must be >= 1")
        if self.max_candidates_per_item < 1:
            raise ValueError("evaluation.max_candidates_per_item must be >= 1")
        if self.max_snippet_chars < 1:
            raise ValueError("evaluation.max_snippet_chars must be >= 1")
        if self.llm_timeout_s < 1:
            raise ValueError("evaluation.llm_timeout_s must be >= 1")
        if not self.html_title.strip():
            raise ValueError("evaluation.html_title must be a non-empty string")
        return self


class AgenticRagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: PathsConfig
    file_selection: FileSelectionConfig
    io_limits: IOLimitsConfig
    llm: LLMConfig
    embedding: EmbeddingConfig
    fallback: FallbackConfig
    stack_detection: StackDetectionConfig
    parser: ParserConfig
    security_tagging: SecurityTaggingConfig
    logging: LoggingConfig
    evaluation: EvaluationConfig
    chat: ChatConfig

    @model_validator(mode="after")
    def _validate_cross_fields(self):
        if not self.llm.enabled:
            raise ValueError("llm.enabled must be true (LLM is required for this runtime)")
        if not self.llm.base_url or not self.llm.base_url.strip():
            raise ValueError("llm.base_url is required when llm.enabled is true")
        if not self.llm.model or not self.llm.model.strip():
            raise ValueError("llm.model is required when llm.enabled is true")

        return self
