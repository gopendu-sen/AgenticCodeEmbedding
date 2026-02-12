import logging
import time
from typing import Dict, List, Optional

from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.config import EmbeddingCollectionsConfig
from agentic_rag.embedding.chroma_store import ChromaStore
from agentic_rag.embedding.client import EmbeddingClient


logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        embedder: EmbeddingClient,
        chroma: ChromaStore,
        collections: EmbeddingCollectionsConfig,
        repo_name: str,
        node_text_max_chars: int,
        embed_doc_max_chars: int,
        verbose_per_node: bool = False,
    ):
        self.embedder = embedder
        self.chroma = chroma
        self.collections = collections
        self.repo_name = repo_name.strip()
        if not self.repo_name:
            raise ValueError("repo_name must be a non-empty string")
        self.node_text_max_chars = node_text_max_chars
        self.embed_doc_max_chars = embed_doc_max_chars
        self.verbose_per_node = verbose_per_node
        self._run_stats: Dict[str, object] = {
            "upsert_calls": 0,
            "total_nodes_received": 0,
            "total_nodes_upserted": 0,
            "total_batches": 0,
            "total_embed_seconds": 0.0,
            "total_upsert_seconds": 0.0,
            "total_elapsed_seconds": 0.0,
            "total_embed_input_tokens": 0,
            "total_embed_output_tokens": 0,
            "total_embed_tokens": 0,
            "embed_usage_missing_batches": 0,
            "collection_totals": {},
            "batch_events": [],
        }

    def _choose_collection(self, node: CodeNode) -> Optional[str]:
        if node.node_type in (
            "function",
            "class",
            "component",
            "table",
            "view",
            "query",
            "notebook_code",
        ):
            return self.collections.code_symbols
        if node.node_type in (
            "route",
            "entrypoint",
        ):
            return self.collections.code_routes
        if node.node_type in (
            "call_edge",
            "import_usage",
        ):
            return self.collections.code_usage
        if node.node_type in (
            "doc",
            "comment",
            "comment_line",
            "comment_block",
            "comment_doc",
            "comment_other",
            "doc_code",
        ):
            return self.collections.docs
        if node.node_type in (
            "auth_guard",
            "policy_check",
            "audit_log",
            "sensitive_op",
        ):
            return self.collections.security_tags
        if node.node_type == "flow_chain":
            return self.collections.flows
        if node.node_type == "config":
            return self.collections.configs
        return None

    def _make_embed_text(self, node: CodeNode) -> str:
        meta = node.metadata or {}

        base = [
            f"TYPE: {node.node_type}",
            f"LANG: {node.language}",
            f"FILE: {node.file_path}",
            f"LINES: {node.start_line}-{node.end_line}",
        ]
        if node.symbol:
            base.append(f"SYMBOL: {node.symbol}")

        if node.node_type in ("doc", "doc_code"):
            doc_title = meta.get("doc_title")
            section_path = meta.get("section_path")
            section_title = meta.get("section_title")
            doc_format = meta.get("doc_format")
            code_language = meta.get("code_language")

            if doc_title:
                base.append(f"DOC_TITLE: {doc_title}")
            if section_path:
                base.append(f"SECTION_PATH: {section_path}")
            if section_title:
                base.append(f"SECTION_TITLE: {section_title}")
            if doc_format:
                base.append(f"DOC_FORMAT: {doc_format}")
            if code_language:
                base.append(f"CODE_LANG: {code_language}")

            base.append("DOC_CONTENT:\n" + node.text[:self.node_text_max_chars])
            return "\n".join(base)

        if node.node_type == "call_edge":
            caller = meta.get("caller")
            callee = meta.get("callee")
            relation = meta.get("relation")
            if caller:
                base.append(f"CALLER: {caller}")
            if callee:
                base.append(f"CALLEE: {callee}")
            if relation:
                base.append(f"RELATION: {relation}")
            base.append("CODE_CONTEXT:\n" + node.text[:self.node_text_max_chars])
            return "\n".join(base)

        why = meta.get("why_relevant")
        if why:
            base.append(f"WHY: {why}")
        base.append("CODE:\n" + node.text[:self.node_text_max_chars])
        return "\n".join(base)

    def upsert_nodes(self, nodes: List[CodeNode], batch_size: int) -> int:
        if not nodes:
            logger.info("Embedding upsert skipped: empty node batch")
            return 0

        call_started = time.perf_counter()
        self._run_stats["upsert_calls"] = int(self._run_stats["upsert_calls"]) + 1
        self._run_stats["total_nodes_received"] = int(self._run_stats["total_nodes_received"]) + len(nodes)
        logger.info("Embedding upsert call started: nodes=%d requested_batch_size=%d", len(nodes), batch_size)

        batch_size = max(1, batch_size)
        batch_by_collection: Dict[str, Dict[str, List]] = {}

        for node in nodes:
            collection = self._choose_collection(node)
            if not collection:
                if self.verbose_per_node:
                    logger.info(
                        "Embedding node skipped: id=%s type=%s file=%s",
                        node.node_id,
                        node.node_type,
                        node.file_path,
                    )
                continue
            if collection not in batch_by_collection:
                batch_by_collection[collection] = {"ids": [], "embed_texts": [], "docs": [], "metas": []}

            batch_by_collection[collection]["ids"].append(node.node_id)
            batch_by_collection[collection]["embed_texts"].append(self._make_embed_text(node))
            batch_by_collection[collection]["docs"].append(node.text[:self.embed_doc_max_chars])
            batch_by_collection[collection]["metas"].append({
                "repo_name": self.repo_name,
                "file_path": node.file_path,
                "start_line": node.start_line,
                "end_line": node.end_line,
                "node_type": node.node_type,
                "language": node.language,
                "symbol": node.symbol or "",
                "confidence": float(node.confidence),
            })
            if self.verbose_per_node:
                logger.info(
                    "Embedding node mapped: id=%s type=%s file=%s lines=%d-%d collection=%s",
                    node.node_id,
                    node.node_type,
                    node.file_path,
                    node.start_line,
                    node.end_line,
                    collection,
                )

        upserted = 0
        for collection_name, pack in batch_by_collection.items():
            logger.info(
                "Embedding collection prepared: collection=%s nodes=%d",
                collection_name,
                len(pack["ids"]),
            )

        for collection_name, pack in batch_by_collection.items():
            for start in range(0, len(pack["ids"]), batch_size):
                end = start + batch_size
                ids = pack["ids"][start:end]
                embed_texts = pack["embed_texts"][start:end]
                docs = pack["docs"][start:end]
                metas = pack["metas"][start:end]

                batch_event = {
                    "collection": collection_name,
                    "batch_start_index": start,
                    "batch_end_index": min(end, len(pack["ids"])),
                    "batch_size": len(ids),
                }
                batch_started = time.perf_counter()
                logger.info(
                    "Embedding batch started: collection=%s batch_index=%d batch_size=%d",
                    collection_name,
                    (start // batch_size) + 1,
                    len(ids),
                )
                embed_started = time.perf_counter()
                vectors = self.embedder.embed(embed_texts)
                embed_elapsed = time.perf_counter() - embed_started
                usage = self.embedder.last_call_usage()
                batch_prompt_tokens = int(usage.get("prompt_tokens", 0))
                batch_completion_tokens = int(usage.get("completion_tokens", 0))
                batch_total_tokens = int(usage.get("total_tokens", 0))
                has_usage = bool(usage.get("has_usage", False))
                batch_event["embed_input_tokens"] = batch_prompt_tokens
                batch_event["embed_output_tokens"] = batch_completion_tokens
                batch_event["embed_total_tokens"] = batch_total_tokens
                batch_event["embed_usage_reported"] = has_usage

                upsert_started = time.perf_counter()
                self.chroma.upsert(
                    collection_name=collection_name,
                    ids=ids,
                    embeddings=vectors,
                    documents=docs,
                    metadatas=metas,
                )
                upsert_elapsed = time.perf_counter() - upsert_started
                batch_elapsed = time.perf_counter() - batch_started

                batch_event["embed_seconds"] = round(embed_elapsed, 6)
                batch_event["upsert_seconds"] = round(upsert_elapsed, 6)
                batch_event["elapsed_seconds"] = round(batch_elapsed, 6)
                cast_events = self._run_stats["batch_events"]
                if isinstance(cast_events, list):
                    cast_events.append(batch_event)

                self._run_stats["total_batches"] = int(self._run_stats["total_batches"]) + 1
                self._run_stats["total_embed_seconds"] = float(self._run_stats["total_embed_seconds"]) + embed_elapsed
                self._run_stats["total_upsert_seconds"] = float(self._run_stats["total_upsert_seconds"]) + upsert_elapsed
                self._run_stats["total_embed_input_tokens"] = (
                    int(self._run_stats["total_embed_input_tokens"]) + batch_prompt_tokens
                )
                self._run_stats["total_embed_output_tokens"] = (
                    int(self._run_stats["total_embed_output_tokens"]) + batch_completion_tokens
                )
                self._run_stats["total_embed_tokens"] = (
                    int(self._run_stats["total_embed_tokens"]) + batch_total_tokens
                )
                if not has_usage:
                    self._run_stats["embed_usage_missing_batches"] = (
                        int(self._run_stats["embed_usage_missing_batches"]) + 1
                    )
                collection_totals = self._run_stats["collection_totals"]
                if isinstance(collection_totals, dict):
                    collection_totals[collection_name] = int(collection_totals.get(collection_name, 0)) + len(ids)

                logger.info(
                    (
                        "Embedding batch completed: collection=%s batch_size=%d "
                        "embed_s=%.4f upsert_s=%.4f total_s=%.4f embed_input_tokens=%d embed_output_tokens=%d"
                    ),
                    collection_name,
                    len(ids),
                    embed_elapsed,
                    upsert_elapsed,
                    batch_elapsed,
                    batch_prompt_tokens,
                    batch_completion_tokens,
                )
                upserted += len(ids)

        elapsed = time.perf_counter() - call_started
        self._run_stats["total_nodes_upserted"] = int(self._run_stats["total_nodes_upserted"]) + upserted
        self._run_stats["total_elapsed_seconds"] = float(self._run_stats["total_elapsed_seconds"]) + elapsed
        logger.info(
            "Embedding upsert call completed: nodes_upserted=%d elapsed_s=%.4f",
            upserted,
            elapsed,
        )
        return upserted

    def counts(self) -> Dict[str, int]:
        names = [
            self.collections.code_symbols,
            self.collections.code_routes,
            self.collections.code_usage,
            self.collections.docs,
            self.collections.configs,
            self.collections.security_tags,
            self.collections.flows,
        ]
        return {name: self.chroma.count(name) for name in names}

    def telemetry(self) -> Dict[str, object]:
        events = self._run_stats.get("batch_events")
        safe_events = list(events) if isinstance(events, list) else []
        return {
            "upsert_calls": int(self._run_stats.get("upsert_calls", 0)),
            "total_nodes_received": int(self._run_stats.get("total_nodes_received", 0)),
            "total_nodes_upserted": int(self._run_stats.get("total_nodes_upserted", 0)),
            "total_batches": int(self._run_stats.get("total_batches", 0)),
            "total_embed_seconds": round(float(self._run_stats.get("total_embed_seconds", 0.0)), 6),
            "total_upsert_seconds": round(float(self._run_stats.get("total_upsert_seconds", 0.0)), 6),
            "total_elapsed_seconds": round(float(self._run_stats.get("total_elapsed_seconds", 0.0)), 6),
            "total_embed_input_tokens": int(self._run_stats.get("total_embed_input_tokens", 0)),
            "total_embed_output_tokens": int(self._run_stats.get("total_embed_output_tokens", 0)),
            "total_embed_tokens": int(self._run_stats.get("total_embed_tokens", 0)),
            "embed_usage_missing_batches": int(self._run_stats.get("embed_usage_missing_batches", 0)),
            "collection_totals": dict(self._run_stats.get("collection_totals", {})),
            "batch_events": safe_events,
        }
