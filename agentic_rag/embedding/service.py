import json
import logging
import time
from typing import Any, Dict, List, Optional

from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.config import EmbeddingCollectionsConfig
from agentic_rag.embedding.chroma_store import ChromaStore
from agentic_rag.embedding.client import EmbeddingClient


logger = logging.getLogger(__name__)


class EmbeddingService:
    _BASE_STORE_KEYS = (
        "code_symbols",
        "code_routes",
        "code_usage",
        "docs",
        "configs",
        "security_tags",
        "flows",
    )
    _AUDIT_STORE_KEYS = (
        "audit_identity_profile",
        "audit_auth_controls",
        "audit_money_movement",
        "audit_payee_recipient",
        "audit_docs_disclosures",
        "audit_limits_access",
    )

    _AUDIT_TOKEN_MAP: Dict[str, List[str]] = {
        "audit_identity_profile": [
            "customer",
            "profile",
            "account_details",
            "account detail",
            "pii",
            "personally identifiable",
            "ssn",
            "sin",
            "dob",
            "kyc",
            "mergecustomer",
            "mergecif",
            "dedupecustomer",
            "registerdevice",
            "trusteddevice",
            "recoverusername",
            "forgotusername",
            "addbusinessadmin",
            "grantentitlement",
        ],
        "audit_auth_controls": [
            "resetpassword",
            "changepassword",
            "resetpin",
            "changepin",
            "enrollmfa",
            "disablemfa",
            "unlockaccount",
            "reissuecard",
            "replacecard",
            "otp",
            "trusteddevice",
            "registerdevice",
            "forgotusername",
            "recoverusername",
            "grantentitlement",
            "revokeentitlement",
            "blockaccount",
            "freezeaccount",
            "suspendcard",
            "unblockaccount",
            "unfreezeaccount",
            "restorecard",
            "reactivateaccount",
        ],
        "audit_money_movement": [
            "transfer",
            "internaltransfer",
            "samecustomertransfer",
            "intercustomertransfer",
            "transfertotdcustomer",
            "recipientcustomerid",
            "ach",
            "eft",
            "wire",
            "swift",
            "iban",
            "sepa",
            "routing number",
            "beneficiary",
            "payee",
            "outbound payment",
            "promise to pay",
            "promisetopay",
            "ptp",
            "payment arrangement",
        ],
        "audit_payee_recipient": [
            "payee",
            "beneficiary",
            "recipient",
            "saved payee",
            "payeeid",
            "createpayee",
            "updatepayee",
            "deletepayee",
            "saverecipient",
            "deleterecipient",
            "recipientcustomerid",
            "recipient lookup",
        ],
        "audit_docs_disclosures": [
            "docusign",
            "adobe sign",
            "esign",
            "envelope",
            "signature request",
            "generate pdf",
            "disclosure",
            "consent",
            "statement delivery",
            "doc_format",
            "document template",
        ],
        "audit_limits_access": [
            "blockaccount",
            "freezeaccount",
            "reducelimit",
            "lowerdailylimit",
            "suspendcard",
            "hold funds",
            "lock transaction",
            "decrease transfer limit",
            "unblockaccount",
            "unfreezeaccount",
            "increaselimit",
            "raisedailylimit",
            "restorecard",
            "remove hold",
            "reactivateaccount",
            "setdailytransferlimit",
            "setdailywithdrawlimit",
        ],
    }

    def __init__(
        self,
        embedder: EmbeddingClient,
        chroma: ChromaStore,
        collections: EmbeddingCollectionsConfig,
        repo_name: str,
        node_text_max_chars: int,
        embed_doc_max_chars: int,
        enable_audit_dimensions: bool = True,
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
        self.enable_audit_dimensions = bool(enable_audit_dimensions)
        self.verbose_per_node = verbose_per_node

        self._store_catalog = self._build_store_catalog()
        self._store_key_to_collection: Dict[str, str] = {
            item["store_key"]: item["collection_name"] for item in self._store_catalog
        }

        self._store_stats: Dict[str, Dict[str, Any]] = {
            item["store_key"]: {
                "store_key": item["store_key"],
                "collection_name": item["collection_name"],
                "store_type": item["store_type"],
                "nodes_queued": 0,
                "nodes_upserted": 0,
                "stale_vectors_deleted": 0,
                "final_vector_count": 0,
                "status": "empty",
                "error": "",
            }
            for item in self._store_catalog
        }

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
            "stale_delete_calls": 0,
            "stale_delete_failures": 0,
            "total_stale_vectors_deleted": 0,
            "collection_totals": {},
            "batch_events": [],
            "stale_delete_events": [],
        }

    def _build_store_catalog(self) -> List[Dict[str, str]]:
        catalog: List[Dict[str, str]] = []
        for key in self._BASE_STORE_KEYS:
            catalog.append(
                {
                    "store_key": key,
                    "collection_name": str(getattr(self.collections, key)),
                    "store_type": "base",
                }
            )

        if self.enable_audit_dimensions:
            for key in self._AUDIT_STORE_KEYS:
                catalog.append(
                    {
                        "store_key": key,
                        "collection_name": str(getattr(self.collections, key)),
                        "store_type": "audit_dimension",
                    }
                )

        return catalog

    def collection_descriptors(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self._store_catalog]

    def _mark_store_failed(self, store_key: str, error: str) -> None:
        stats = self._store_stats.get(store_key)
        if not stats:
            return
        stats["status"] = "failed"
        stats["error"] = error.strip()

    def _base_collection_key(self, node: CodeNode) -> Optional[str]:
        if node.node_type in (
            "function",
            "class",
            "component",
            "table",
            "view",
            "query",
            "notebook_code",
        ):
            return "code_symbols"
        if node.node_type in ("route", "entrypoint"):
            return "code_routes"
        if node.node_type in ("call_edge", "import_usage"):
            return "code_usage"
        if node.node_type in (
            "doc",
            "comment",
            "comment_line",
            "comment_block",
            "comment_doc",
            "comment_other",
            "doc_code",
        ):
            return "docs"
        if node.node_type in ("auth_guard", "policy_check", "audit_log", "sensitive_op"):
            return "security_tags"
        if node.node_type == "flow_chain":
            return "flows"
        if node.node_type == "config":
            return "configs"
        return None

    @staticmethod
    def _normalize_text_for_match(text: str) -> str:
        lowered = text.lower()
        return " ".join(lowered.replace("\n", " ").replace("\t", " ").split())

    def _build_node_match_text(self, node: CodeNode) -> str:
        meta = node.metadata or {}
        parts = [
            node.node_type,
            node.language,
            node.file_path,
            node.symbol or "",
            node.text[: self.node_text_max_chars],
        ]
        for key, value in meta.items():
            if isinstance(value, (dict, list, tuple, set)):
                rendered = json.dumps(value, ensure_ascii=False)
            else:
                rendered = str(value)
            parts.append(f"{key}:{rendered}")
        return self._normalize_text_for_match("\n".join(parts))

    def _audit_collection_keys(self, node: CodeNode) -> List[str]:
        if not self.enable_audit_dimensions:
            return []

        out: List[str] = []
        text = self._build_node_match_text(node)

        if node.node_type in {"auth_guard", "policy_check", "sensitive_op"}:
            out.append("audit_auth_controls")
        if node.node_type == "audit_log":
            out.append("audit_auth_controls")
            out.append("audit_identity_profile")

        for store_key, tokens in self._AUDIT_TOKEN_MAP.items():
            if any(token in text for token in tokens):
                out.append(store_key)

        seen = set()
        unique: List[str] = []
        for key in out:
            if key not in self._store_key_to_collection:
                continue
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def _choose_collections(self, node: CodeNode) -> List[str]:
        out: List[str] = []

        base_key = self._base_collection_key(node)
        if base_key and base_key in self._store_key_to_collection:
            out.append(base_key)

        for audit_key in self._audit_collection_keys(node):
            if audit_key not in out:
                out.append(audit_key)

        return out

    @staticmethod
    def _render_meta_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (dict, list, tuple, set)):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        text = str(value).strip()
        return text

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

        metadata_fields = (
            ("FRAMEWORK", "framework"),
            ("ANNOTATIONS", "annotations"),
            ("ATTRIBUTES", "attributes"),
            ("SQL_KIND", "sql_kind"),
            ("SQL_OP", "sql_op"),
            ("IMPORT_KIND", "import_kind"),
            ("ENTRYPOINT_KIND", "entrypoint_kind"),
            ("SECTION_PATH", "section_path"),
            ("SECTION_TITLE", "section_title"),
            ("SECTION_LEVEL", "section_level"),
            ("COMMENT_KIND", "comment_kind"),
        )
        for label, key in metadata_fields:
            rendered = self._render_meta_value(meta.get(key))
            if rendered:
                base.append(f"{label}: {rendered}")

        if node.node_type in ("doc", "doc_code"):
            doc_title = self._render_meta_value(meta.get("doc_title"))
            doc_format = self._render_meta_value(meta.get("doc_format"))
            code_language = self._render_meta_value(meta.get("code_language"))
            if doc_title:
                base.append(f"DOC_TITLE: {doc_title}")
            if doc_format:
                base.append(f"DOC_FORMAT: {doc_format}")
            if code_language:
                base.append(f"CODE_LANG: {code_language}")
            base.append("DOC_CONTENT:\n" + node.text[: self.node_text_max_chars])
            return "\n".join(base)

        if node.node_type == "call_edge":
            caller = self._render_meta_value(meta.get("caller"))
            callee = self._render_meta_value(meta.get("callee"))
            relation = self._render_meta_value(meta.get("relation"))
            if caller:
                base.append(f"CALLER: {caller}")
            if callee:
                base.append(f"CALLEE: {callee}")
            if relation:
                base.append(f"RELATION: {relation}")
            base.append("CODE_CONTEXT:\n" + node.text[: self.node_text_max_chars])
            return "\n".join(base)

        why = self._render_meta_value(meta.get("why_relevant"))
        if why:
            base.append(f"WHY: {why}")
        base.append("CODE:\n" + node.text[: self.node_text_max_chars])
        return "\n".join(base)

    def delete_vectors_for_file(self, file_path: str) -> Dict[str, Any]:
        self._run_stats["stale_delete_calls"] = int(self._run_stats.get("stale_delete_calls", 0)) + 1

        event: Dict[str, Any] = {
            "file_path": file_path,
            "repo_name": self.repo_name,
            "stores": [],
            "deleted_total": 0,
            "errors": [],
        }

        for descriptor in self._store_catalog:
            store_key = descriptor["store_key"]
            collection_name = descriptor["collection_name"]
            store_event = {
                "store_key": store_key,
                "collection_name": collection_name,
                "deleted": 0,
                "error": "",
            }
            try:
                deleted = self.chroma.delete_where(
                    collection_name=collection_name,
                    where={"repo_name": self.repo_name, "file_path": file_path},
                )
                deleted = max(0, int(deleted))
                store_event["deleted"] = deleted
                event["deleted_total"] = int(event["deleted_total"]) + deleted
                self._run_stats["total_stale_vectors_deleted"] = (
                    int(self._run_stats.get("total_stale_vectors_deleted", 0)) + deleted
                )
                stats = self._store_stats.get(store_key)
                if stats:
                    stats["stale_vectors_deleted"] = int(stats.get("stale_vectors_deleted", 0)) + deleted
            except Exception as exc:  # noqa: BLE001
                err_text = str(exc)
                store_event["error"] = err_text
                event_errors = event.get("errors")
                if isinstance(event_errors, list):
                    event_errors.append({"store_key": store_key, "error": err_text})
                self._run_stats["stale_delete_failures"] = int(self._run_stats.get("stale_delete_failures", 0)) + 1
                self._mark_store_failed(store_key, f"stale_delete_failed: {err_text}")
            stores = event.get("stores")
            if isinstance(stores, list):
                stores.append(store_event)

        stale_events = self._run_stats.get("stale_delete_events")
        if isinstance(stale_events, list):
            stale_events.append(event)

        return event

    def upsert_nodes(self, nodes: List[CodeNode], batch_size: int) -> int:
        if not nodes:
            logger.info("Embedding upsert skipped: empty node batch")
            return 0

        call_started = time.perf_counter()
        self._run_stats["upsert_calls"] = int(self._run_stats["upsert_calls"]) + 1
        self._run_stats["total_nodes_received"] = int(self._run_stats["total_nodes_received"]) + len(nodes)
        logger.info("Embedding upsert call started: nodes=%d requested_batch_size=%d", len(nodes), batch_size)

        batch_size = max(1, batch_size)
        batch_by_store: Dict[str, Dict[str, List[Any]]] = {}

        for node in nodes:
            store_keys = self._choose_collections(node)
            if not store_keys:
                if self.verbose_per_node:
                    logger.info(
                        "Embedding node skipped: id=%s type=%s file=%s",
                        node.node_id,
                        node.node_type,
                        node.file_path,
                    )
                continue

            for store_key in store_keys:
                collection_name = self._store_key_to_collection.get(store_key)
                if not collection_name:
                    continue

                if store_key not in batch_by_store:
                    batch_by_store[store_key] = {
                        "ids": [],
                        "embed_texts": [],
                        "docs": [],
                        "metas": [],
                    }

                batch_by_store[store_key]["ids"].append(node.node_id)
                batch_by_store[store_key]["embed_texts"].append(self._make_embed_text(node))
                batch_by_store[store_key]["docs"].append(node.text[: self.embed_doc_max_chars])
                batch_by_store[store_key]["metas"].append(
                    {
                        "repo_name": self.repo_name,
                        "file_path": node.file_path,
                        "start_line": node.start_line,
                        "end_line": node.end_line,
                        "node_type": node.node_type,
                        "language": node.language,
                        "symbol": node.symbol or "",
                        "confidence": float(node.confidence),
                        "store_key": store_key,
                    }
                )

                stats = self._store_stats.get(store_key)
                if stats:
                    stats["nodes_queued"] = int(stats.get("nodes_queued", 0)) + 1

                if self.verbose_per_node:
                    logger.info(
                        "Embedding node mapped: id=%s type=%s file=%s lines=%d-%d store=%s collection=%s",
                        node.node_id,
                        node.node_type,
                        node.file_path,
                        node.start_line,
                        node.end_line,
                        store_key,
                        collection_name,
                    )

        upserted = 0
        for store_key, pack in batch_by_store.items():
            logger.info(
                "Embedding collection prepared: store=%s collection=%s nodes=%d",
                store_key,
                self._store_key_to_collection.get(store_key, ""),
                len(pack["ids"]),
            )

        for store_key, pack in batch_by_store.items():
            collection_name = self._store_key_to_collection.get(store_key)
            if not collection_name:
                continue

            for start in range(0, len(pack["ids"]), batch_size):
                end = start + batch_size
                ids = pack["ids"][start:end]
                embed_texts = pack["embed_texts"][start:end]
                docs = pack["docs"][start:end]
                metas = pack["metas"][start:end]

                batch_event = {
                    "store_key": store_key,
                    "collection": collection_name,
                    "batch_start_index": start,
                    "batch_end_index": min(end, len(pack["ids"])),
                    "batch_size": len(ids),
                }
                batch_started = time.perf_counter()
                logger.info(
                    "Embedding batch started: store=%s collection=%s batch_index=%d batch_size=%d",
                    store_key,
                    collection_name,
                    (start // batch_size) + 1,
                    len(ids),
                )
                try:
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

                    stats = self._store_stats.get(store_key)
                    if stats:
                        stats["nodes_upserted"] = int(stats.get("nodes_upserted", 0)) + len(ids)
                        if stats.get("status") != "failed":
                            stats["status"] = "active"

                    logger.info(
                        (
                            "Embedding batch completed: store=%s collection=%s batch_size=%d "
                            "embed_s=%.4f upsert_s=%.4f total_s=%.4f embed_input_tokens=%d embed_output_tokens=%d"
                        ),
                        store_key,
                        collection_name,
                        len(ids),
                        embed_elapsed,
                        upsert_elapsed,
                        batch_elapsed,
                        batch_prompt_tokens,
                        batch_completion_tokens,
                    )
                    upserted += len(ids)
                except Exception as exc:  # noqa: BLE001
                    err_text = str(exc)
                    self._mark_store_failed(store_key, err_text)
                    logger.exception(
                        "Embedding batch failed: store=%s collection=%s batch_size=%d error=%s",
                        store_key,
                        collection_name,
                        len(ids),
                        err_text,
                    )
                    raise

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
        names: List[str] = []
        seen = set()
        for item in self._store_catalog:
            collection_name = item["collection_name"]
            if collection_name in seen:
                continue
            seen.add(collection_name)
            names.append(collection_name)

        out: Dict[str, int] = {}
        for name in names:
            try:
                out[name] = self.chroma.count(name)
            except Exception:  # noqa: BLE001
                out[name] = 0
        return out

    def _store_types(self) -> Dict[str, List[str]]:
        base_stores: List[str] = []
        audit_stores: List[str] = []
        for descriptor in self._store_catalog:
            if descriptor["store_type"] == "base":
                base_stores.append(descriptor["store_key"])
            elif descriptor["store_type"] == "audit_dimension":
                audit_stores.append(descriptor["store_key"])
        return {
            "base_stores": base_stores,
            "audit_dimension_stores": audit_stores,
        }

    def _store_summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for descriptor in self._store_catalog:
            store_key = descriptor["store_key"]
            collection_name = descriptor["collection_name"]
            stats = self._store_stats.get(store_key, {})

            final_count = int(stats.get("final_vector_count", 0))
            status = str(stats.get("status", "empty") or "empty")
            error = str(stats.get("error", "") or "")

            try:
                final_count = int(self.chroma.count(collection_name))
                if status != "failed":
                    status = "active" if final_count > 0 else "empty"
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                error = str(exc)

            stats["final_vector_count"] = final_count
            stats["status"] = status
            stats["error"] = error

            summary.append(
                {
                    "store_key": store_key,
                    "collection_name": collection_name,
                    "store_type": descriptor["store_type"],
                    "nodes_queued": int(stats.get("nodes_queued", 0)),
                    "nodes_upserted": int(stats.get("nodes_upserted", 0)),
                    "stale_vectors_deleted": int(stats.get("stale_vectors_deleted", 0)),
                    "final_vector_count": final_count,
                    "status": status,
                    "error": error,
                }
            )

        return summary

    def telemetry(self) -> Dict[str, object]:
        events = self._run_stats.get("batch_events")
        safe_events = list(events) if isinstance(events, list) else []
        stale_events = self._run_stats.get("stale_delete_events")
        safe_stale_events = list(stale_events) if isinstance(stale_events, list) else []
        store_summary = self._store_summary()
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
            "stale_delete_calls": int(self._run_stats.get("stale_delete_calls", 0)),
            "stale_delete_failures": int(self._run_stats.get("stale_delete_failures", 0)),
            "total_stale_vectors_deleted": int(self._run_stats.get("total_stale_vectors_deleted", 0)),
            "collection_totals": dict(self._run_stats.get("collection_totals", {})),
            "batch_events": safe_events,
            "stale_delete_events": safe_stale_events,
            "store_types": self._store_types(),
            "store_summary": store_summary,
        }
