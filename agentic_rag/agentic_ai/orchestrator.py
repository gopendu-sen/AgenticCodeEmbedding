import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from agentic_rag.agentic_ai.stack_detect import StackDetector, llm_stack_fallback
from agentic_rag.agentic_ai.tool_plan_agent import SecurityTagAgent, ToolPlanParseAgent, execute_tool_plan
from agentic_rag.code_parser.node_builder import build_nodes_from_agent, build_security_nodes_from_agent
from agentic_rag.code_parser.service import CodeParserService
from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.llm_client import LLMClient
from agentic_rag.core.repo_tools import RepoTools
from agentic_rag.core.sqlite_store import SQLiteStore
from agentic_rag.core.utils import sha256_text
from agentic_rag.embedding.chroma_store import ChromaStore
from agentic_rag.embedding.client import EmbeddingClient
from agentic_rag.embedding.service import EmbeddingService


logger = logging.getLogger(__name__)


def is_important_file(rel_path: str, text: str, important_dir_hints: List[str], important_keywords: List[str]) -> bool:
    lp = rel_path.lower().replace("\\", "/")
    if any(h.lower() in lp for h in important_dir_hints):
        return True
    t = text.lower()
    return any(k.lower() in t for k in important_keywords)


class AgenticRagOrchestrator:
    def __init__(self, cfg: AgenticRagConfig, repo_name: str):
        self.cfg = cfg
        self.repo_name = repo_name.strip()
        if not self.repo_name:
            raise ValueError("repo_name must be a non-empty string")
        logger.info("Initializing AgenticRagOrchestrator: repo=%s repo_name=%s", cfg.paths.repo_path, self.repo_name)

        os.makedirs(cfg.paths.out_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cfg.paths.sqlite_path), exist_ok=True)
        os.makedirs(cfg.paths.reports_dir, exist_ok=True)
        logger.info(
            "Output directories ensured: out_dir=%s sqlite_dir=%s reports_dir=%s",
            cfg.paths.out_dir,
            os.path.dirname(cfg.paths.sqlite_path),
            cfg.paths.reports_dir,
        )

        self.repo = RepoTools(
            repo_path=cfg.paths.repo_path,
            exclude_dirs=cfg.file_selection.exclude_dirs,
            max_file_size=cfg.file_selection.max_file_size_bytes,
            read_file_max_chars=cfg.io_limits.read_file_max_chars,
            repo_tree_max_files=cfg.io_limits.repo_tree_max_files,
            tool_search_max_hits=cfg.io_limits.tool_search_max_hits,
        )
        self.sqlite = SQLiteStore(cfg.paths.sqlite_path)
        self.chroma = ChromaStore(cfg.paths.chroma_dir)

        self.embedder = EmbeddingClient(
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            timeout_s=cfg.embedding.timeout_s,
        )
        self.embedding = EmbeddingService(
            embedder=self.embedder,
            chroma=self.chroma,
            collections=cfg.embedding.collections,
            repo_name=self.repo_name,
            node_text_max_chars=cfg.io_limits.node_text_max_chars,
            embed_doc_max_chars=cfg.io_limits.embed_doc_max_chars,
            enable_audit_dimensions=cfg.embedding.enable_audit_dimensions,
            verbose_per_node=cfg.logging.embedding_verbose_per_node,
        )
        self.parser = CodeParserService(parser_config=cfg.parser, io_limits=cfg.io_limits)

        self.stack_detector = StackDetector(
            signals=cfg.stack_detection.signals,
            dotnet_project_suffixes=cfg.stack_detection.dotnet_project_suffixes,
            confidence_base=cfg.stack_detection.confidence.base,
            confidence_per_signal=cfg.stack_detection.confidence.per_signal,
            confidence_max=cfg.stack_detection.confidence.max,
        )

        if not cfg.llm.enabled:
            raise ValueError("LLM is required and must be enabled")

        self.llm = LLMClient(
            base_url=(cfg.llm.base_url or "").strip(),
            model=(cfg.llm.model or "").strip(),
            timeout_s=cfg.llm.timeout_s,
        )
        self.agent = ToolPlanParseAgent(self.llm)
        self.security_llm = None

        logger.info(
            "Probing primary LLM endpoint: base_url=%s model=%s timeout_s=%s",
            self.llm.base_url,
            self.llm.model,
            self.llm.timeout_s,
        )
        self.llm.probe()
        logger.info("Primary LLM probe succeeded")

        self.security_agent = None
        if cfg.security_tagging.enabled:
            security_llm = LLMClient(
                base_url=(cfg.llm.base_url or "").strip(),
                model=(cfg.llm.model or "").strip(),
                timeout_s=cfg.security_tagging.llm_timeout_s,
            )
            logger.info(
                "Probing security tagger LLM endpoint: base_url=%s model=%s timeout_s=%s",
                security_llm.base_url,
                security_llm.model,
                security_llm.timeout_s,
            )
            security_llm.probe()
            logger.info("Security tagger LLM probe succeeded")
            self.security_llm = security_llm
            self.security_agent = SecurityTagAgent(security_llm)

    def _write_report(self, report_payload: Dict[str, Any]) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"ingestion_report_{ts}.json"
        report_path = os.path.join(self.cfg.paths.reports_dir, filename)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report_payload, handle, ensure_ascii=False, indent=2)
        return report_path

    def _file_hash(self, text: str) -> str:
        hash_input = f"{self.cfg.parser.index_version}\\n{text}"
        return sha256_text(hash_input)

    def run(self) -> Dict[str, Any]:
        self.embedder.reset_usage()
        self.llm.reset_usage()
        if self.security_llm:
            self.security_llm.reset_usage()

        run_started = datetime.now(timezone.utc)
        logger.info(
            "Ingestion run started: repo=%s repo_name=%s started_at=%s",
            self.cfg.paths.repo_path,
            self.repo_name,
            run_started.isoformat(),
        )
        stacks = self.stack_detector.detect(self.cfg.paths.repo_path)
        stack_conf = self.stack_detector.confidence(stacks)
        frameworks: List[str] = []
        logger.info("Stack detection completed: stacks=%s confidence=%.4f", sorted(stacks.keys()), stack_conf)

        if self.cfg.fallback.stack.enabled and stack_conf < self.cfg.fallback.stack.llm_min_confidence:
            logger.info(
                "Stack fallback LLM triggered: confidence=%.4f threshold=%.4f",
                stack_conf,
                self.cfg.fallback.stack.llm_min_confidence,
            )
            tree = self.repo.repo_tree_snapshot(include_exts=self.cfg.file_selection.include_exts)
            manifest_snips: Dict[str, str] = {}
            for manifest in self.cfg.fallback.stack.manifest_files:
                manifest_path = os.path.join(self.cfg.paths.repo_path, manifest)
                if os.path.exists(manifest_path):
                    manifest_snips[manifest] = self.repo.read_file(
                        manifest,
                        max_chars=self.cfg.io_limits.manifest_snippet_chars,
                    )

            logger.info(
                "Stack fallback context prepared: tree_files=%d manifest_snippets=%d",
                len(tree),
                len(manifest_snips),
            )
            out = llm_stack_fallback(self.llm, tree, manifest_snips)
            frameworks = out.get("frameworks", []) or []
            if not stacks and out.get("stacks"):
                stacks = {s: [] for s in out.get("stacks", [])}
            stack_conf = float(out.get("confidence", stack_conf))
            logger.info(
                "Stack fallback completed: stacks=%s frameworks=%s confidence=%.4f",
                sorted(stacks.keys()),
                frameworks,
                stack_conf,
            )

        files = list(self.repo.iter_files(self.cfg.file_selection.include_exts))
        logger.info("Repository scan listed files: total=%d", len(files))
        changed_files = 0
        total_nodes = 0
        embedded_nodes = 0
        pending_nodes: List[CodeNode] = []
        pending_file_paths: Dict[str, bool] = {}
        fallback_allowed_exts = set(self.cfg.fallback.parse.allowed_exts)
        security_supported_exts = set(self.cfg.security_tagging.supported_exts)
        node_type_counts: Dict[str, int] = {}
        security_tagger_attempts = 0
        security_tagger_failures = 0
        security_tags_generated = 0
        parser_llm_fallback_attempts = 0
        parser_llm_fallback_failures = 0
        parser_llm_fallback_nodes_added = 0
        embedding_flush_failures = 0
        stale_delete_attempts = 0
        stale_delete_failures = 0
        stale_vectors_deleted = 0
        sqlite_stale_nodes_deleted = 0
        file_reports: List[Dict[str, Any]] = []
        file_report_index: Dict[str, Dict[str, Any]] = {}

        def flush_pending(flush_reason: str) -> None:
            nonlocal embedded_nodes, embedding_flush_failures
            if not pending_nodes:
                return

            involved_files = list(pending_file_paths.keys())
            logger.info(
                "Embedding flush triggered: reason=%s pending_nodes=%d files=%d batch_size=%d",
                flush_reason,
                len(pending_nodes),
                len(involved_files),
                self.cfg.embedding.batch_size,
            )
            try:
                upserted = self.embedding.upsert_nodes(
                    pending_nodes,
                    batch_size=self.cfg.embedding.batch_size,
                )
                embedded_nodes += upserted
                for file_path in involved_files:
                    file_report = file_report_index.get(file_path)
                    if not file_report:
                        continue
                    embedding_meta = file_report.get("embedding")
                    if isinstance(embedding_meta, dict):
                        embedding_meta["status"] = "success"
                        embedding_meta["error"] = ""
                logger.info(
                    "Embedding flush completed: reason=%s files=%d nodes_upserted=%d",
                    flush_reason,
                    len(involved_files),
                    upserted,
                )
            except Exception as exc:  # noqa: BLE001
                embedding_flush_failures += 1
                err_text = str(exc)
                logger.exception(
                    "Embedding flush failed: reason=%s files=%d pending_nodes=%d error=%s",
                    flush_reason,
                    len(involved_files),
                    len(pending_nodes),
                    err_text,
                )
                for file_path in involved_files:
                    file_report = file_report_index.get(file_path)
                    if not file_report:
                        continue
                    file_report["partial"] = True
                    errors = file_report.get("errors")
                    if isinstance(errors, list):
                        errors.append(f"embedding_error: {err_text}")
                    file_report["status"] = "failed"
                    embedding_meta = file_report.get("embedding")
                    if isinstance(embedding_meta, dict):
                        embedding_meta["status"] = "failed"
                        embedding_meta["error"] = err_text
            finally:
                pending_nodes.clear()
                pending_file_paths.clear()

        for idx, rf in enumerate(files, start=1):
            file_report: Dict[str, Any] = {
                "file_path": rf.rel_path,
                "ext": rf.ext,
                "size_bytes": rf.size,
                "changed": False,
                "status": "success",
                "partial": False,
                "language": "",
                "confidence": 0.0,
                "deterministic_nodes": 0,
                "total_nodes": 0,
                "llm_fallback": {
                    "attempted": False,
                    "success": False,
                    "nodes_added": 0,
                    "error": "",
                },
                "security_tagging": {
                    "attempted": False,
                    "success": False,
                    "nodes_added": 0,
                    "error": "",
                },
                "embedding": {
                    "status": "pending",
                    "queued_nodes": 0,
                    "error": "",
                },
                "stale_cleanup": {
                    "attempted": False,
                    "vectors_deleted": 0,
                    "sqlite_nodes_deleted": 0,
                    "errors": [],
                },
                "errors": [],
            }
            file_reports.append(file_report)
            file_report_index[rf.rel_path] = file_report

            logger.info("Processing file %d/%d: rel_path=%s ext=%s size=%d", idx, len(files), rf.rel_path, rf.ext, rf.size)
            try:
                text = self.repo.read_file(rf.rel_path)
                file_hash = self._file_hash(text)

                prev_hash = self.sqlite.get_file_hash(rf.rel_path)
                if prev_hash == file_hash:
                    logger.debug("Skipping unchanged file: %s", rf.rel_path)
                    file_report["status"] = "skipped_unchanged"
                    embedding_meta = file_report.get("embedding")
                    if isinstance(embedding_meta, dict):
                        embedding_meta["status"] = "skipped"
                    continue

                file_report["changed"] = True
                changed_files += 1
                logger.info("File changed and selected for indexing: %s", rf.rel_path)

                stale_meta = file_report.get("stale_cleanup")
                if isinstance(stale_meta, dict):
                    stale_meta["attempted"] = True
                stale_delete_attempts += 1

                try:
                    delete_result = self.embedding.delete_vectors_for_file(rf.rel_path)
                    deleted_vectors = int(delete_result.get("deleted_total", 0))
                    stale_vectors_deleted += deleted_vectors
                    if isinstance(stale_meta, dict):
                        stale_meta["vectors_deleted"] = deleted_vectors
                    if delete_result.get("errors"):
                        stale_delete_failures += int(len(delete_result.get("errors", [])))
                        file_report["partial"] = True
                        errors = file_report.get("errors")
                        if isinstance(errors, list):
                            errors.append("stale_vector_delete_partial_failure")
                        stale_errors = stale_meta.get("errors") if isinstance(stale_meta, dict) else None
                        if isinstance(stale_errors, list):
                            for item in delete_result.get("errors", []):
                                if isinstance(item, dict):
                                    stale_errors.append(str(item.get("error", "")))
                                else:
                                    stale_errors.append(str(item))
                except Exception as exc:  # noqa: BLE001
                    stale_delete_failures += 1
                    err_text = str(exc)
                    file_report["partial"] = True
                    errors = file_report.get("errors")
                    if isinstance(errors, list):
                        errors.append(f"stale_vector_delete_failed: {err_text}")
                    if isinstance(stale_meta, dict):
                        stale_meta["errors"] = [err_text]

                try:
                    deleted_sqlite = self.sqlite.delete_nodes_for_file(rf.rel_path)
                    sqlite_stale_nodes_deleted += deleted_sqlite
                    if isinstance(stale_meta, dict):
                        stale_meta["sqlite_nodes_deleted"] = deleted_sqlite
                except Exception as exc:  # noqa: BLE001
                    stale_delete_failures += 1
                    err_text = str(exc)
                    file_report["partial"] = True
                    errors = file_report.get("errors")
                    if isinstance(errors, list):
                        errors.append(f"sqlite_stale_delete_failed: {err_text}")
                    stale_errors = stale_meta.get("errors") if isinstance(stale_meta, dict) else None
                    if isinstance(stale_errors, list):
                        stale_errors.append(err_text)

                self.sqlite.upsert_file(rf.rel_path, file_hash)

                nodes, conf, lang = self.parser.parse_file(rf.rel_path, text, rf.ext)
                seen = {n.node_id for n in nodes}
                parser_error = any((n.metadata or {}).get("parse_error") for n in nodes)
                file_report["language"] = lang
                file_report["confidence"] = float(conf)
                file_report["deterministic_nodes"] = len(nodes)
                logger.info(
                    "Deterministic parse result: file=%s lang=%s nodes=%d confidence=%.4f parser_error=%s",
                    rf.rel_path,
                    lang,
                    len(nodes),
                    conf,
                    parser_error,
                )

                low_confidence_fallback = (
                    self.cfg.fallback.parse.enabled
                    and conf < self.cfg.fallback.parse.confidence_threshold
                    and rf.ext in fallback_allowed_exts
                    and is_important_file(
                        rf.rel_path,
                        text,
                        important_dir_hints=self.cfg.fallback.parse.important_dir_hints,
                        important_keywords=self.cfg.fallback.parse.important_keywords,
                    )
                )
                should_try_llm_fallback = parser_error or low_confidence_fallback

                if should_try_llm_fallback:
                    parser_llm_fallback_attempts += 1
                    llm_meta = file_report.get("llm_fallback")
                    if isinstance(llm_meta, dict):
                        llm_meta["attempted"] = True
                    reason = "parser_error" if parser_error else "low_confidence_important_file"
                    logger.info(
                        "LLM parser fallback triggered: file=%s reason=%s conf=%.4f threshold=%.4f",
                        rf.rel_path,
                        reason,
                        conf,
                        self.cfg.fallback.parse.confidence_threshold,
                    )
                    try:
                        header = "\n".join(text.splitlines()[: self.cfg.io_limits.llm_header_preview_lines])
                        plan = self.agent.propose_plan(rf.rel_path, lang, header)
                        logger.info(
                            "LLM fallback plan generated: file=%s steps=%d",
                            rf.rel_path,
                            len(plan.get("steps", [])) if isinstance(plan, dict) else 0,
                        )
                        tool_out = execute_tool_plan(self.repo, plan)
                        final = self.agent.finalize(rf.rel_path, lang, tool_out)

                        llm_nodes = build_nodes_from_agent(
                            file_path=rf.rel_path,
                            file_text=text,
                            language=final.get("language", lang),
                            agent_json=final,
                        )

                        llm_added = 0
                        for n in llm_nodes:
                            if n.node_id not in seen:
                                nodes.append(n)
                                seen.add(n.node_id)
                                llm_added += 1
                        parser_llm_fallback_nodes_added += llm_added
                        if isinstance(llm_meta, dict):
                            llm_meta["success"] = True
                            llm_meta["nodes_added"] = llm_added
                            llm_meta["error"] = ""
                        logger.info(
                            "LLM parser fallback completed: file=%s llm_nodes_total=%d llm_nodes_added=%d",
                            rf.rel_path,
                            len(llm_nodes),
                            llm_added,
                        )
                    except Exception as exc:  # noqa: BLE001
                        parser_llm_fallback_failures += 1
                        err_text = str(exc)
                        logger.exception("LLM parser fallback failed: file=%s error=%s", rf.rel_path, err_text)
                        logger.warning(
                            "Continuing with deterministic nodes after fallback failure: file=%s nodes=%d",
                            rf.rel_path,
                            len(nodes),
                        )
                        file_report["partial"] = True
                        errors = file_report.get("errors")
                        if isinstance(errors, list):
                            errors.append(f"llm_fallback_error: {err_text}")
                        if isinstance(llm_meta, dict):
                            llm_meta["success"] = False
                            llm_meta["error"] = err_text

                if (
                    self.security_agent
                    and self.cfg.security_tagging.enabled
                    and rf.ext in security_supported_exts
                ):
                    security_tagger_attempts += 1
                    security_meta = file_report.get("security_tagging")
                    if isinstance(security_meta, dict):
                        security_meta["attempted"] = True
                    logger.info("Security tagging triggered: file=%s ext=%s", rf.rel_path, rf.ext)
                    try:
                        security_header = "\n".join(text.splitlines()[: self.cfg.security_tagging.max_header_lines])
                        security_plan = self.security_agent.propose_plan(rf.rel_path, lang, security_header)
                        logger.info(
                            "Security plan generated: file=%s steps=%d",
                            rf.rel_path,
                            len(security_plan.get("steps", [])) if isinstance(security_plan, dict) else 0,
                        )
                        security_tool_out = execute_tool_plan(self.repo, security_plan)
                        security_final = self.security_agent.finalize(rf.rel_path, lang, security_tool_out)
                        security_nodes = build_security_nodes_from_agent(
                            file_path=rf.rel_path,
                            file_text=text,
                            language=security_final.get("language", lang),
                            agent_json=security_final,
                        )
                        security_added = 0
                        for n in security_nodes:
                            if n.node_id in seen:
                                continue
                            nodes.append(n)
                            seen.add(n.node_id)
                            security_added += 1
                            if n.node_type in {"auth_guard", "policy_check", "audit_log", "sensitive_op"}:
                                security_tags_generated += 1
                        if isinstance(security_meta, dict):
                            security_meta["success"] = True
                            security_meta["nodes_added"] = security_added
                            security_meta["error"] = ""
                        logger.info(
                            "Security tagging completed: file=%s security_nodes_total=%d security_nodes_added=%d",
                            rf.rel_path,
                            len(security_nodes),
                            security_added,
                        )
                    except Exception as exc:  # noqa: BLE001
                        security_tagger_failures += 1
                        err_text = str(exc)
                        logger.exception("Security tagging failed: file=%s error=%s", rf.rel_path, err_text)
                        file_report["partial"] = True
                        errors = file_report.get("errors")
                        if isinstance(errors, list):
                            errors.append(f"security_tagger_error: {err_text}")
                        if isinstance(security_meta, dict):
                            security_meta["success"] = False
                            security_meta["error"] = err_text

                file_report["total_nodes"] = len(nodes)
                embedding_meta = file_report.get("embedding")
                if isinstance(embedding_meta, dict):
                    embedding_meta["queued_nodes"] = len(nodes)

                if not nodes:
                    file_report["status"] = "error"
                    file_report["partial"] = True
                    errors = file_report.get("errors")
                    if isinstance(errors, list):
                        errors.append("no_nodes_generated")
                    if isinstance(embedding_meta, dict):
                        embedding_meta["status"] = "skipped"
                    continue

                for n in nodes:
                    total_nodes += 1
                    node_type_counts[n.node_type] = int(node_type_counts.get(n.node_type, 0)) + 1
                    self.sqlite.upsert_node(
                        node_id=n.node_id,
                        file_path=n.file_path,
                        node_type=n.node_type,
                        language=n.language,
                        start_line=n.start_line,
                        end_line=n.end_line,
                        symbol=n.symbol,
                        content_hash=n.content_hash,
                        confidence=float(n.confidence),
                        text=n.text,
                        metadata=n.metadata,
                    )
                    pending_nodes.append(n)
                pending_file_paths[rf.rel_path] = True

                if len(pending_nodes) >= self.cfg.embedding.batch_size:
                    flush_pending("batch_threshold")

            except Exception as exc:  # noqa: BLE001
                err_text = str(exc)
                logger.exception("Failed processing file: file=%s error=%s", rf.rel_path, err_text)
                file_report["status"] = "failed"
                file_report["partial"] = False
                errors = file_report.get("errors")
                if isinstance(errors, list):
                    errors.append(f"file_processing_error: {err_text}")
                embedding_meta = file_report.get("embedding")
                if isinstance(embedding_meta, dict):
                    embedding_meta["status"] = "failed"
                    embedding_meta["error"] = err_text

        if pending_nodes:
            flush_pending("final_flush")

        file_status_counts: Dict[str, int] = {}
        partial_files_count = 0
        for file_report in file_reports:
            status = str(file_report.get("status", "success"))
            if status not in {"failed", "skipped_unchanged"}:
                llm_meta = file_report.get("llm_fallback")
                security_meta = file_report.get("security_tagging")
                embedding_meta = file_report.get("embedding")
                errors = file_report.get("errors")
                has_errors = isinstance(errors, list) and len(errors) > 0
                embedding_status = ""
                if isinstance(embedding_meta, dict):
                    embedding_status = str(embedding_meta.get("status", ""))
                llm_success = isinstance(llm_meta, dict) and bool(llm_meta.get("success"))
                security_success = isinstance(security_meta, dict) and bool(security_meta.get("success"))
                used_llm_success = llm_success or security_success

                if embedding_status == "failed":
                    status = "failed"
                    file_report["partial"] = True
                elif has_errors:
                    status = "error"
                    file_report["partial"] = True
                elif used_llm_success:
                    status = "success_with_llm"
                elif file_report.get("changed"):
                    status = "success"
                else:
                    status = "skipped_unchanged"
                file_report["status"] = status

            if bool(file_report.get("partial")):
                partial_files_count += 1
            file_status_counts[status] = int(file_status_counts.get(status, 0)) + 1

        run_finished = datetime.now(timezone.utc)
        embedding_telemetry = self.embedding.telemetry()
        primary_chat_usage = self.llm.usage()
        security_chat_usage = (
            self.security_llm.usage()
            if self.security_llm
            else {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "missing_usage_responses": 0,
            }
        )
        chat_usage = {
            "requests": int(primary_chat_usage.get("requests", 0)) + int(security_chat_usage.get("requests", 0)),
            "prompt_tokens": int(primary_chat_usage.get("prompt_tokens", 0)) + int(security_chat_usage.get("prompt_tokens", 0)),
            "completion_tokens": int(primary_chat_usage.get("completion_tokens", 0))
            + int(security_chat_usage.get("completion_tokens", 0)),
            "total_tokens": int(primary_chat_usage.get("total_tokens", 0)) + int(security_chat_usage.get("total_tokens", 0)),
            "missing_usage_responses": int(primary_chat_usage.get("missing_usage_responses", 0))
            + int(security_chat_usage.get("missing_usage_responses", 0)),
        }
        embedding_usage = self.embedder.usage()
        chat_token_usage = {
            "requests": int(chat_usage["requests"]),
            "input_tokens": int(chat_usage["prompt_tokens"]),
            "output_tokens": int(chat_usage["completion_tokens"]),
            "total_tokens": int(chat_usage["total_tokens"]),
            "missing_usage_responses": int(chat_usage["missing_usage_responses"]),
        }
        embedding_token_usage = {
            "requests": int(embedding_usage.get("requests", 0)),
            "input_tokens": int(embedding_usage.get("prompt_tokens", 0)),
            "output_tokens": int(embedding_usage.get("completion_tokens", 0)),
            "total_tokens": int(embedding_usage.get("total_tokens", 0)),
            "missing_usage_responses": int(embedding_usage.get("missing_usage_responses", 0)),
            "input_items": int(embedding_usage.get("input_items", 0)),
        }
        failed_files_count = int(file_status_counts.get("failed", 0))
        errored_files_count = int(file_status_counts.get("error", 0))
        run_status = "completed_partial" if (failed_files_count > 0 or errored_files_count > 0) else "completed"

        summary = {
            "repo_path": self.cfg.paths.repo_path,
            "repo_name": self.repo_name,
            "embedding_run_status": run_status,
            "parser_index_version": self.cfg.parser.index_version,
            "base_url": self.cfg.embedding.base_url,
            "stacks": stacks,
            "frameworks": frameworks,
            "stack_confidence": stack_conf,
            "files_scanned": len(files),
            "files_changed_indexed": changed_files,
            "files_partial": partial_files_count,
            "files_failed": failed_files_count,
            "files_error": errored_files_count,
            "file_status_counts": dict(sorted(file_status_counts.items(), key=lambda item: item[0])),
            "nodes_created_or_updated": total_nodes,
            "nodes_embedded_upserted": embedded_nodes,
            "node_type_counts": dict(sorted(node_type_counts.items(), key=lambda item: item[0])),
            "chroma_counts": self.embedding.counts(),
            "embedding_collection_totals_run": embedding_telemetry.get("collection_totals", {}),
            "embedding_store_types": embedding_telemetry.get("store_types", {}),
            "embedding_store_summary": embedding_telemetry.get("store_summary", []),
            "embedding_flush_failures": embedding_flush_failures,
            "stale_delete_attempts": stale_delete_attempts,
            "stale_delete_failures": stale_delete_failures,
            "stale_vectors_deleted": stale_vectors_deleted,
            "sqlite_stale_nodes_deleted": sqlite_stale_nodes_deleted,
            "security_tags_generated": security_tags_generated,
            "security_tagger_attempts": security_tagger_attempts,
            "security_tagger_failures": security_tagger_failures,
            "parser_llm_fallback_attempts": parser_llm_fallback_attempts,
            "parser_llm_fallback_failures": parser_llm_fallback_failures,
            "parser_llm_fallback_nodes_added": parser_llm_fallback_nodes_added,
            "chat_token_usage": chat_token_usage,
            "embedding_token_usage": embedding_token_usage,
            "files": file_reports,
        }
        report_payload = {
            "report_type": "agentic_rag_ingestion",
            "generated_at_utc": run_finished.isoformat(),
            "run_started_at_utc": run_started.isoformat(),
            "run_finished_at_utc": run_finished.isoformat(),
            "summary": summary,
            "files": file_reports,
            "embedding": embedding_telemetry,
            "token_usage": {
                "chat": chat_token_usage,
                "chat_breakdown": {
                    "primary_llm": {
                        "requests": int(primary_chat_usage.get("requests", 0)),
                        "input_tokens": int(primary_chat_usage.get("prompt_tokens", 0)),
                        "output_tokens": int(primary_chat_usage.get("completion_tokens", 0)),
                        "total_tokens": int(primary_chat_usage.get("total_tokens", 0)),
                        "missing_usage_responses": int(primary_chat_usage.get("missing_usage_responses", 0)),
                    },
                    "security_tagger_llm": {
                        "requests": int(security_chat_usage.get("requests", 0)),
                        "input_tokens": int(security_chat_usage.get("prompt_tokens", 0)),
                        "output_tokens": int(security_chat_usage.get("completion_tokens", 0)),
                        "total_tokens": int(security_chat_usage.get("total_tokens", 0)),
                        "missing_usage_responses": int(security_chat_usage.get("missing_usage_responses", 0)),
                    },
                },
                "embedding": embedding_token_usage,
            },
            "security_tagging": {
                "enabled": self.cfg.security_tagging.enabled,
                "supported_exts": self.cfg.security_tagging.supported_exts,
                "llm_timeout_s": self.cfg.security_tagging.llm_timeout_s,
                "max_header_lines": self.cfg.security_tagging.max_header_lines,
                "attempts": security_tagger_attempts,
                "failures": security_tagger_failures,
                "generated": security_tags_generated,
            },
            "parser_llm_fallback": {
                "attempts": parser_llm_fallback_attempts,
                "failures": parser_llm_fallback_failures,
                "nodes_added": parser_llm_fallback_nodes_added,
            },
            "config_snapshot": {
                "repo_path": self.cfg.paths.repo_path,
                "repo_name": self.repo_name,
                "sqlite_path": self.cfg.paths.sqlite_path,
                "chroma_dir": self.cfg.paths.chroma_dir,
                "reports_dir": self.cfg.paths.reports_dir,
                "embedding_model": self.cfg.embedding.model,
                "embedding_base_url": self.cfg.embedding.base_url,
                "embedding_batch_size": self.cfg.embedding.batch_size,
                "embedding_enable_audit_dimensions": self.cfg.embedding.enable_audit_dimensions,
                "embedding_collections": self.cfg.embedding.collections.model_dump(),
                "parser_index_version": self.cfg.parser.index_version,
                "logging_level": self.cfg.logging.level,
                "embedding_verbose_per_node": self.cfg.logging.embedding_verbose_per_node,
            },
        }
        report_path = self._write_report(report_payload)
        summary["report_path"] = report_path
        summary["report_generated_at_utc"] = run_finished.isoformat()
        logger.info(
            (
                "Ingestion run completed: repo_name=%s status=%s changed_files=%d nodes=%d embedded=%d "
                "partial_files=%d failed_files=%d error_files=%d security_tags=%d parser_fallback_failures=%d report=%s"
            ),
            self.repo_name,
            run_status,
            changed_files,
            total_nodes,
            embedded_nodes,
            partial_files_count,
            failed_files_count,
            errored_files_count,
            security_tags_generated,
            parser_llm_fallback_failures,
            report_path,
        )
        return summary
