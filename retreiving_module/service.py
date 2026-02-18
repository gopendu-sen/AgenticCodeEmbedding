import json
import hashlib
import html
import logging
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agentic_rag.agentic_ai.orchestrator import AgenticRagOrchestrator
from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.llm_client import LLMClient
from agentic_rag.embedding.chroma_store import ChromaStore
from agentic_rag.embedding.client import EmbeddingClient


logger = logging.getLogger(__name__)
_CITATION_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CITATION_RULES = (
    "CITATION RULES:\n"
    "- Use only bracketed numeric citations like [1], [2], [3].\n"
    "- Every non-trivial claim must cite at least one source id from Retrieved context.\n"
    "- Do not cite ids that are not present in Retrieved context.\n"
    "- If context is insufficient, say that clearly and cite the most relevant available source ids."
)


class StoreRetriever:
    def __init__(
        self,
        cfg: AgenticRagConfig,
        config_path: str,
        chroma: ChromaStore,
        embedder: EmbeddingClient,
        llm: LLMClient,
    ):
        self.cfg = cfg
        self.config_path = os.path.abspath(config_path)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.chroma = chroma
        self.embedder = embedder
        self.llm = llm
        self.retrieve_log_path = os.path.join(cfg.paths.reports_dir, "retrieve_log.jsonl")
        self.embedding_jobs_dir = os.path.join(cfg.paths.reports_dir, "embedding_jobs")
        self.evaluation_rules_path = cfg.evaluation.rules_json_path
        self.evaluation_jobs_dir = cfg.evaluation.jobs_dir
        self.evaluation_reports_dir = cfg.evaluation.reports_dir
        self.evaluation_log_path = cfg.evaluation.log_jsonl_path
        self._default_evaluation_rules_path = os.path.join(
            os.path.dirname(__file__),
            "default_evaluation_rules.json",
        )
        os.makedirs(cfg.paths.reports_dir, exist_ok=True)
        os.makedirs(self.embedding_jobs_dir, exist_ok=True)
        os.makedirs(self.evaluation_jobs_dir, exist_ok=True)
        os.makedirs(self.evaluation_reports_dir, exist_ok=True)
        self._last_retrieve_debug: Dict[str, Any] = {}
        self._ensure_evaluation_rules_file()

        cols = cfg.embedding.collections
        self.collection_map: Dict[str, str] = {
            "code_symbols": cols.code_symbols,
            "code_routes": cols.code_routes,
            "code_usage": cols.code_usage,
            "configs": cols.configs,
            "docs": cols.docs,
            "security_tags": cols.security_tags,
            "flows": cols.flows,
            "audit_identity_profile": cols.audit_identity_profile,
            "audit_auth_controls": cols.audit_auth_controls,
            "audit_money_movement": cols.audit_money_movement,
            "audit_payee_recipient": cols.audit_payee_recipient,
            "audit_docs_disclosures": cols.audit_docs_disclosures,
            "audit_limits_access": cols.audit_limits_access,
        }
        self.base_collection_keys: List[str] = [
            "code_symbols",
            "code_routes",
            "code_usage",
            "configs",
            "docs",
            "security_tags",
            "flows",
        ]
        self.audit_collection_keys: List[str] = [
            "audit_identity_profile",
            "audit_auth_controls",
            "audit_money_movement",
            "audit_payee_recipient",
            "audit_docs_disclosures",
            "audit_limits_access",
        ]
        self.rule_dimension_preferences: Dict[str, List[str]] = {
            "A": ["audit_identity_profile"],
            "B": ["audit_identity_profile"],
            "C": ["audit_docs_disclosures"],
            "D": ["audit_auth_controls"],
            "E": ["audit_identity_profile"],
            "F": ["audit_identity_profile"],
            "G": ["audit_identity_profile", "audit_auth_controls"],
            "H": ["audit_identity_profile", "audit_auth_controls"],
            "I1": ["audit_money_movement"],
            "I2": ["audit_money_movement", "audit_payee_recipient"],
            "J": ["audit_money_movement"],
            "K": ["audit_money_movement", "audit_payee_recipient"],
            "L": ["audit_money_movement", "audit_payee_recipient"],
            "M": ["audit_identity_profile", "audit_auth_controls"],
            "N": ["audit_money_movement"],
            "O": ["audit_limits_access", "audit_auth_controls"],
            "P": ["audit_limits_access", "audit_auth_controls"],
        }

    @classmethod
    def from_config(cls, cfg: AgenticRagConfig, config_path: str = "config.yml") -> "StoreRetriever":
        if not cfg.llm.enabled:
            raise ValueError("llm.enabled must be true for retrieval")
        llm = LLMClient(
            base_url=(cfg.llm.base_url or "").strip(),
            model=(cfg.llm.model or "").strip(),
            timeout_s=cfg.llm.timeout_s,
        )
        embedder = EmbeddingClient(
            base_url=cfg.embedding.base_url,
            model=cfg.embedding.model,
            timeout_s=cfg.embedding.timeout_s,
        )
        chroma = ChromaStore(cfg.paths.chroma_dir)
        return cls(
            cfg=cfg,
            config_path=config_path,
            chroma=chroma,
            embedder=embedder,
            llm=llm,
        )

    @staticmethod
    def parse_store_names(raw: str) -> List[str]:
        out: List[str] = []
        seen = set()
        for token in raw.split(","):
            value = token.strip()
            if not value or value in seen:
                continue
            out.append(value)
            seen.add(value)
        return out

    @staticmethod
    def default_repo_name(repo_path: str) -> str:
        normalized = os.path.abspath(repo_path.strip()) if repo_path.strip() else ""
        name = os.path.basename(normalized)
        return name or "repo_store"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _read_json_file(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"JSON payload must be an object: {path}")
        return payload

    @staticmethod
    def _write_json_file(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _ensure_evaluation_rules_file(self) -> None:
        if os.path.exists(self.evaluation_rules_path):
            return
        if not os.path.exists(self._default_evaluation_rules_path):
            raise FileNotFoundError(
                f"Missing default evaluation rules file: {self._default_evaluation_rules_path}"
            )
        default_payload = self._read_json_file(self._default_evaluation_rules_path)
        self._write_json_file(self.evaluation_rules_path, default_payload)
        logger.info(
            "Initialized evaluation rules from default: rules_path=%s default=%s",
            self.evaluation_rules_path,
            self._default_evaluation_rules_path,
        )

    @staticmethod
    def _normalize_token(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    def _detect_query_intent(self, query: str) -> str:
        lowered = query.lower()
        scored: Dict[str, int] = {}

        for intent, keywords in self.cfg.chat.retrieval.intent_keywords.items():
            if intent == "general":
                continue
            score = sum(1 for kw in keywords if kw.lower() in lowered)
            if score > 0:
                scored[intent] = score

        if scored:
            return max(scored, key=lambda key: scored[key])

        if "general" in self.cfg.chat.retrieval.intent_weights:
            return "general"

        for intent in self.cfg.chat.retrieval.intent_weights:
            return intent
        return "general"

    def _weighted_limits(self, intent: str) -> Dict[str, int]:
        weights = self.cfg.chat.retrieval.intent_weights.get(intent)
        if weights is None:
            weights = self.cfg.chat.retrieval.intent_weights.get("general", {})

        base_top_k = max(1, self.cfg.chat.retrieval.base_top_k)
        out: Dict[str, int] = {}
        for collection_key, weight in weights.items():
            n_results = int(round(base_top_k * float(weight)))
            if n_results > 0:
                out[collection_key] = n_results
        return out

    def _normalize_store_names(self, store_names: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in store_names:
            value = raw.strip()
            if not value or value in seen:
                continue
            out.append(value)
            seen.add(value)
        return out

    @staticmethod
    def _split_budget(total: int, buckets: int) -> List[int]:
        if buckets <= 0:
            return []
        total = max(0, total)
        base = total // buckets
        remainder = total % buckets
        return [base + (1 if idx < remainder else 0) for idx in range(buckets)]

    def _evaluation_preferred_limits(self, rule_id: str) -> Dict[str, int]:
        preferred = self.rule_dimension_preferences.get(rule_id, [])
        budget = max(2, self.cfg.chat.retrieval.base_top_k * 2)
        limits: Dict[str, int] = {}
        for key in preferred:
            if key in self.collection_map:
                limits[key] = budget
        return limits

    def _evaluation_base_limits(self) -> Dict[str, int]:
        budget = max(1, self.cfg.chat.retrieval.base_top_k)
        return {key: budget for key in self.base_collection_keys if key in self.collection_map}

    @staticmethod
    def _query_terms(query: str) -> List[str]:
        terms = [token for token in _TOKEN_RE.findall(query.lower()) if len(token) >= 2]
        seen = set()
        ordered: List[str] = []
        for term in terms:
            if term in seen:
                continue
            seen.add(term)
            ordered.append(term)
        return ordered

    @staticmethod
    def _lexical_score(terms: List[str], query_text: str, document: str, metadata: Dict[str, Any]) -> float:
        if not terms:
            return 0.0
        haystack = " ".join(
            [
                document or "",
                str(metadata.get("symbol", "")),
                str(metadata.get("file_path", "")),
                str(metadata.get("node_type", "")),
                str(metadata.get("language", "")),
            ]
        ).lower()
        if not haystack:
            return 0.0

        score = 0.0
        for term in terms:
            if term in haystack:
                score += 1.0
        compact_query = query_text.strip().lower()
        if compact_query and compact_query in haystack:
            score += 2.0
        return score

    def _build_text_fallback_res(
        self,
        *,
        query: str,
        collection_name: str,
        store_name: str,
        n_results: int,
        scan_limit: int,
    ) -> Dict[str, Any]:
        records = self.chroma.get_records(
            collection_name=collection_name,
            limit=max(1, scan_limit),
            where={"repo_name": store_name},
        )
        ids = list(records.get("ids") or [])
        docs = list(records.get("documents") or [])
        metas_raw = list(records.get("metadatas") or [])
        metas: List[Dict[str, Any]] = []
        for item in metas_raw:
            if isinstance(item, dict):
                metas.append(item)
            else:
                metas.append({})

        if not ids:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "fallback_candidates": 0,
            }

        terms = self._query_terms(query)
        scored: List[Tuple[float, int]] = []
        for idx in range(len(ids)):
            metadata = metas[idx] if idx < len(metas) else {}
            document = docs[idx] if idx < len(docs) else ""
            score = self._lexical_score(terms, query, document, metadata)
            scored.append((score, idx))

        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[: max(1, n_results)]

        out_ids: List[Any] = []
        out_docs: List[str] = []
        out_metas: List[Dict[str, Any]] = []
        out_distances: List[float] = []
        for score, idx in top:
            out_ids.append(ids[idx])
            out_docs.append(docs[idx] if idx < len(docs) else "")
            meta = metas[idx] if idx < len(metas) else {}
            if "repo_name" not in meta:
                meta["repo_name"] = store_name
            out_metas.append(meta)
            out_distances.append(1.0 / (1.0 + max(0.0, score)))

        return {
            "ids": [out_ids],
            "documents": [out_docs],
            "metadatas": [out_metas],
            "distances": [out_distances],
            "fallback_candidates": len(ids),
        }

    @staticmethod
    def _annotate_sources_with_citations(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for idx, src in enumerate(sources, start=1):
            copied = dict(src)
            copied["citation_index"] = idx
            out.append(copied)
        return out

    @staticmethod
    def _format_context(chunks: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for chunk in chunks:
            citation_index = chunk.get("citation_index", 0)
            meta = chunk.get("metadata") or {}
            file_path = meta.get("file_path", "unknown")
            start_line = meta.get("start_line", "?")
            end_line = meta.get("end_line", "?")
            node_type = meta.get("node_type", "unknown")
            language = meta.get("language", "unknown")
            symbol = meta.get("symbol", "")
            collection = chunk.get("collection", "unknown")
            repo_name = meta.get("repo_name", "unknown")
            blocks.append(
                f"[{citation_index}] repo={repo_name} file={file_path} lines={start_line}-{end_line} "
                f"type={node_type} lang={language} symbol={symbol} collection={collection}\n"
                f"{chunk.get('document', '').strip()}"
            )
        return "\n\n".join(blocks)

    def _build_messages(
        self,
        history: List[Dict[str, Any]],
        context_text: str,
        intent: str,
        store_names: List[str],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.cfg.chat.system_prompt},
            {"role": "system", "content": _CITATION_RULES},
            {"role": "system", "content": f"Detected retrieval intent: {intent}"},
            {"role": "system", "content": f"Active repo stores: {', '.join(store_names)}"},
        ]

        if context_text.strip():
            messages.append({
                "role": "system",
                "content": "Retrieved context:\n" + context_text[: self.cfg.chat.max_context_chars],
            })

        history_window = history[-max(1, self.cfg.chat.history_messages):]
        for msg in history_window:
            role = msg.get("role", "user")
            if role not in {"user", "assistant", "system"}:
                continue
            messages.append({"role": role, "content": msg.get("content", "")})
        return messages

    @staticmethod
    def _extract_citation_indices(text: str, max_index: int) -> List[int]:
        found = set()
        for raw in _CITATION_RE.findall(text):
            idx = int(raw)
            if 1 <= idx <= max_index:
                found.add(idx)
        return sorted(found)

    def _ensure_citations(self, response: str, source_count: int) -> str:
        if source_count <= 0:
            return response
        cited = self._extract_citation_indices(response, source_count)
        if cited:
            return response
        fallback = [f"[{idx}]" for idx in range(1, min(source_count, 3) + 1)]
        return response.rstrip() + "\n\nCitations: " + " ".join(fallback)

    @staticmethod
    def _no_source_response(intent: str, store_names: List[str]) -> str:
        stores_csv = ", ".join(store_names)
        return (
            "I could not retrieve any indexed code/context for the selected repo store(s): "
            f"{stores_csv}.\n\n"
            "Please run or re-run embedding for that repo tag, wait for job status `completed`, "
            "then refresh `Repo Stores` and ask again. "
            f"Detected intent: {intent}."
        )

    def _write_retrieve_log(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        with open(self.retrieve_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _build_chat_log_payload(
        self,
        *,
        timestamp_utc: str,
        prompt: str,
        intent: str,
        repo_stores: List[str],
        sources: List[Dict[str, Any]],
        cited_indices: List[int],
        model: str,
        elapsed_seconds: float,
        session_id: str,
    ) -> Dict[str, Any]:
        return {
            "timestamp_utc": timestamp_utc,
            "event": "chat_turn",
            "session_id": session_id,
            "query": prompt,
            "intent": intent,
            "repo_stores": repo_stores,
            "source_count": len(sources),
            "no_sources": len(sources) == 0,
            "cited_indices": cited_indices,
            "model": model,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "retrieval_debug": self._last_retrieve_debug,
            "sources": [
                {
                    "citation_index": src.get("citation_index"),
                    "repo_name": (src.get("metadata") or {}).get("repo_name", ""),
                    "file_path": (src.get("metadata") or {}).get("file_path", ""),
                    "start_line": (src.get("metadata") or {}).get("start_line", ""),
                    "end_line": (src.get("metadata") or {}).get("end_line", ""),
                    "node_type": (src.get("metadata") or {}).get("node_type", ""),
                    "collection": src.get("collection", ""),
                    "distance": src.get("distance", None),
                }
                for src in sources
            ],
        }

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _job_status_path(self, job_id: str) -> str:
        return os.path.join(self.embedding_jobs_dir, f"{job_id}.json")

    def _job_log_path(self, job_id: str) -> str:
        return os.path.join(self.embedding_jobs_dir, f"{job_id}.log")

    def _read_job(self, job_id: str) -> Dict[str, Any]:
        path = self._job_status_path(job_id)
        if not os.path.exists(path):
            raise ValueError(f"Embedding job not found: {job_id}")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid embedding job payload for {job_id}")
        return payload

    def _write_job(self, job: Dict[str, Any]) -> None:
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("Embedding job payload missing job_id")
        job = self._normalize_embedding_job(job)
        path = self._job_status_path(job_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _normalize_embedding_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(job)
        now = self._utc_now()
        payload["job_id"] = str(payload.get("job_id", "")).strip()
        payload["status"] = str(payload.get("status", "starting") or "starting")
        payload["stage"] = str(payload.get("stage", "init") or "init")
        payload["started_at_utc"] = str(payload.get("started_at_utc", "") or payload.get("created_at_utc", "") or now)
        payload["finished_at_utc"] = str(payload.get("finished_at_utc", "") or "")
        payload["updated_at_utc"] = str(payload.get("updated_at_utc", "") or now)
        payload["error"] = str(payload.get("error", "") or "")
        payload["partial"] = bool(payload.get("partial", False))
        if "summary" not in payload:
            payload["summary"] = None
        return payload

    def _refresh_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        job = self._normalize_embedding_job(job)
        status = str(job.get("status", "")).strip().lower()
        pid_raw = job.get("pid")
        try:
            pid = int(pid_raw) if pid_raw is not None else 0
        except (TypeError, ValueError):
            pid = 0

        if status in {"completed", "completed_partial", "failed"}:
            return job

        if status == "starting":
            return job

        if status == "running" and pid > 0 and self._is_pid_alive(pid):
            return job

        job["status"] = "failed"
        job["stage"] = "failed"
        if not str(job.get("error", "")).strip():
            job["error"] = "Embedding worker process exited unexpectedly"
        if not str(job.get("finished_at_utc", "")).strip():
            job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        job["partial"] = bool(job.get("partial", False))
        job["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_job(job)
        return job

    def start_embedding_job(self, repo_path: str, repo_name: str) -> Dict[str, Any]:
        resolved_repo_path = os.path.abspath(repo_path.strip())
        clean_repo_name = repo_name.strip()
        if not clean_repo_name:
            raise ValueError("repo_name must be a non-empty string")
        if not os.path.isdir(resolved_repo_path):
            raise ValueError(f"repo_path must exist and be a directory: {resolved_repo_path}")

        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        created_at = datetime.now(timezone.utc).isoformat()
        job = {
            "job_id": job_id,
            "status": "starting",
            "stage": "init",
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "config_path": self.config_path,
            "created_at_utc": created_at,
            "started_at_utc": created_at,
            "finished_at_utc": "",
            "updated_at_utc": created_at,
            "pid": None,
            "summary": None,
            "error": "",
            "partial": False,
            "report_path": "",
            "log_path": self._job_log_path(job_id),
        }
        self._write_job(job)

        cmd = [
            sys.executable,
            "-m",
            "retreiving_module.embedding_worker",
            "--config",
            self.config_path,
            "--repo-path",
            resolved_repo_path,
            "--repo-name",
            clean_repo_name,
            "--job-id",
            job_id,
            "--status-path",
            self._job_status_path(job_id),
        ]
        logger.info(
            "Starting embedding background job: job_id=%s repo_name=%s repo_path=%s",
            job_id,
            clean_repo_name,
            resolved_repo_path,
        )

        try:
            with open(self._job_log_path(job_id), "a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=self.project_root,
                )
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
            logger.exception(
                "Failed to start embedding background job: job_id=%s repo_name=%s error=%s",
                job_id,
                clean_repo_name,
                err_text,
            )
            job["status"] = "failed"
            job["stage"] = "failed"
            job["error"] = f"Failed to spawn embedding worker: {err_text}"
            job["pid"] = None
            job["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            job["partial"] = False
            job["traceback"] = traceback.format_exc()
            job["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._write_job(job)
            return job

        job["status"] = "running"
        job["stage"] = "init"
        job["pid"] = process.pid
        job["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        self._write_job(job)
        return job

    def get_embedding_job(self, job_id: str) -> Dict[str, Any]:
        return self._refresh_job(self._read_job(job_id))

    def list_embedding_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        files = [
            os.path.join(self.embedding_jobs_dir, name)
            for name in os.listdir(self.embedding_jobs_dir)
            if name.endswith(".json")
        ]
        jobs: List[Dict[str, Any]] = []
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    jobs.append(self._refresh_job(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read embedding job status: path=%s error=%s", path, exc)
        jobs.sort(key=lambda item: str(item.get("created_at_utc", "")), reverse=True)
        return jobs[: max(1, limit)]

    def run_embedding(self, repo_path: str, repo_name: str) -> Dict[str, Any]:
        resolved_repo_path = os.path.abspath(repo_path.strip())
        clean_repo_name = repo_name.strip()
        if not clean_repo_name:
            raise ValueError("repo_name must be a non-empty string")
        if not os.path.isdir(resolved_repo_path):
            raise ValueError(f"repo_path must exist and be a directory: {resolved_repo_path}")

        logger.info(
            "Embedding run requested from retriever module: repo_path=%s repo_name=%s",
            resolved_repo_path,
            clean_repo_name,
        )

        ingest_cfg = self.cfg.model_copy(
            update={
                "paths": self.cfg.paths.model_copy(update={"repo_path": resolved_repo_path}),
            }
        )
        orchestrator = AgenticRagOrchestrator(ingest_cfg, repo_name=clean_repo_name)
        summary = orchestrator.run()

        logger.info(
            "Embedding run completed from retriever module: repo_name=%s files_changed=%s nodes_embedded=%s",
            clean_repo_name,
            summary.get("files_changed_indexed"),
            summary.get("nodes_embedded_upserted"),
        )
        return summary

    @staticmethod
    def _validate_rule_item(item: Dict[str, Any], seen_ids: set) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("Each rule must be an object")
        required = ("id", "title", "definition", "strong_signals", "weak_signals", "false_positives")
        for key in required:
            if key not in item:
                raise ValueError(f"Rule missing required field: {key}")
        rule_id = str(item.get("id", "")).strip()
        if not rule_id:
            raise ValueError("Rule id must be non-empty")
        if rule_id in seen_ids:
            raise ValueError(f"Duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)

        title = str(item.get("title", "")).strip()
        definition = str(item.get("definition", "")).strip()
        if not title:
            raise ValueError(f"Rule {rule_id} has empty title")
        if not definition:
            raise ValueError(f"Rule {rule_id} has empty definition")

        def _ensure_str_list(key: str) -> List[str]:
            raw = item.get(key)
            if not isinstance(raw, list) or not raw:
                raise ValueError(f"Rule {rule_id}.{key} must be a non-empty list")
            out: List[str] = []
            for value in raw:
                text = str(value).strip()
                if not text:
                    continue
                out.append(text)
            if not out:
                raise ValueError(f"Rule {rule_id}.{key} must contain non-empty values")
            return out

        return {
            "id": rule_id,
            "title": title,
            "definition": definition,
            "strong_signals": _ensure_str_list("strong_signals"),
            "weak_signals": _ensure_str_list("weak_signals"),
            "false_positives": _ensure_str_list("false_positives"),
        }

    def _validate_rules_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Rules payload must be an object")
        if "version" not in payload:
            raise ValueError("Rules payload missing version")
        if "items" not in payload:
            raise ValueError("Rules payload missing items")

        version = int(payload.get("version"))
        if version < 1:
            raise ValueError("Rules version must be >= 1")

        items_raw = payload.get("items")
        if not isinstance(items_raw, list) or not items_raw:
            raise ValueError("Rules items must be a non-empty list")

        seen_ids: set = set()
        items: List[Dict[str, Any]] = []
        for raw in items_raw:
            items.append(self._validate_rule_item(raw, seen_ids))

        updated_at_utc = str(payload.get("updated_at_utc", "")).strip() or self._utc_now()
        return {
            "version": version,
            "updated_at_utc": updated_at_utc,
            "items": items,
        }

    def get_evaluation_rules(self) -> Dict[str, Any]:
        self._ensure_evaluation_rules_file()
        payload = self._read_json_file(self.evaluation_rules_path)
        normalized = self._validate_rules_payload(payload)
        normalized["path"] = self.evaluation_rules_path
        return normalized

    def save_evaluation_rules(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._validate_rules_payload(payload)
        normalized["updated_at_utc"] = self._utc_now()
        self._write_json_file(self.evaluation_rules_path, normalized)
        logger.info(
            "Evaluation rules saved: rules=%d version=%d path=%s",
            len(normalized.get("items", [])),
            int(normalized.get("version", 0)),
            self.evaluation_rules_path,
        )
        out = dict(normalized)
        out["path"] = self.evaluation_rules_path
        return out

    def _evaluation_job_status_path(self, job_id: str) -> str:
        return os.path.join(self.evaluation_jobs_dir, f"{job_id}.json")

    def _evaluation_job_log_path(self, job_id: str) -> str:
        return os.path.join(self.evaluation_jobs_dir, f"{job_id}.log")

    def _read_evaluation_job(self, job_id: str) -> Dict[str, Any]:
        path = self._evaluation_job_status_path(job_id)
        if not os.path.exists(path):
            raise ValueError(f"Evaluation job not found: {job_id}")
        payload = self._read_json_file(path)
        return payload

    def _write_evaluation_job(self, job: Dict[str, Any]) -> None:
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            raise ValueError("Evaluation job payload missing job_id")
        job["updated_at_utc"] = self._utc_now()
        self._write_json_file(self._evaluation_job_status_path(job_id), job)

    def _refresh_evaluation_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        status = str(job.get("status", "")).strip().lower()
        if status != "running":
            return job
        pid_raw = job.get("pid")
        try:
            pid = int(pid_raw) if pid_raw is not None else 0
        except (TypeError, ValueError):
            pid = 0
        if pid > 0 and self._is_pid_alive(pid):
            return job

        job["status"] = "failed"
        if not str(job.get("error", "")).strip():
            job["error"] = "Evaluation worker process exited unexpectedly"
        if not str(job.get("finished_at_utc", "")).strip():
            job["finished_at_utc"] = self._utc_now()
        job["partial"] = bool(job.get("partial", False))
        self._write_evaluation_job(job)
        return job

    def start_evaluation_job(self, repo_name: str, repo_path: str) -> Dict[str, Any]:
        clean_repo_name = repo_name.strip()
        resolved_repo_path = os.path.abspath(repo_path.strip())
        if not clean_repo_name:
            raise ValueError("repo_name must be a non-empty string")
        if not os.path.isdir(resolved_repo_path):
            raise ValueError(f"repo_path must exist and be a directory: {resolved_repo_path}")

        self._ensure_evaluation_rules_file()
        job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        created_at = self._utc_now()
        job = {
            "job_id": job_id,
            "status": "starting",
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "config_path": self.config_path,
            "created_at_utc": created_at,
            "updated_at_utc": created_at,
            "started_at_utc": "",
            "finished_at_utc": "",
            "status_counts": {"detected": 0, "not_detected": 0, "needs_review": 0},
            "rule_count": 0,
            "report_html_path": "",
            "report_json_path": "",
            "error": "",
            "partial": False,
            "log_path": self._evaluation_job_log_path(job_id),
            "pid": None,
            "summary": None,
        }
        self._write_evaluation_job(job)

        cmd = [
            sys.executable,
            "-m",
            "retreiving_module.evaluation_worker",
            "--config",
            self.config_path,
            "--repo-path",
            resolved_repo_path,
            "--repo-name",
            clean_repo_name,
            "--job-id",
            job_id,
            "--status-path",
            self._evaluation_job_status_path(job_id),
        ]
        logger.info(
            "Starting evaluation background job: job_id=%s repo_name=%s repo_path=%s",
            job_id,
            clean_repo_name,
            resolved_repo_path,
        )

        try:
            with open(self._evaluation_job_log_path(job_id), "a", encoding="utf-8") as log_handle:
                process = subprocess.Popen(  # noqa: S603
                    cmd,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=self.project_root,
                )
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)
            logger.exception(
                "Failed to start evaluation background job: job_id=%s repo_name=%s error=%s",
                job_id,
                clean_repo_name,
                err_text,
            )
            job["status"] = "failed"
            job["error"] = f"Failed to spawn evaluation worker: {err_text}"
            job["pid"] = None
            job["finished_at_utc"] = self._utc_now()
            job["partial"] = False
            job["traceback"] = traceback.format_exc()
            self._write_evaluation_job(job)
            return job

        job["status"] = "running"
        job["pid"] = process.pid
        job["started_at_utc"] = self._utc_now()
        self._write_evaluation_job(job)
        return job

    def get_evaluation_job(self, job_id: str) -> Dict[str, Any]:
        return self._refresh_evaluation_job(self._read_evaluation_job(job_id))

    def list_evaluation_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        files = [
            os.path.join(self.evaluation_jobs_dir, name)
            for name in os.listdir(self.evaluation_jobs_dir)
            if name.endswith(".json")
        ]
        jobs: List[Dict[str, Any]] = []
        for path in files:
            try:
                payload = self._read_json_file(path)
                jobs.append(self._refresh_evaluation_job(payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to read evaluation job status: path=%s error=%s", path, exc)
        jobs.sort(key=lambda item: str(item.get("created_at_utc", "")), reverse=True)
        return jobs[: max(1, limit)]

    @staticmethod
    def _safe_abs_file(repo_path: str, rel_path: str) -> Optional[str]:
        repo_abs = os.path.abspath(repo_path)
        candidate = os.path.abspath(os.path.join(repo_abs, rel_path))
        if candidate == repo_abs:
            return candidate
        if candidate.startswith(repo_abs + os.sep):
            return candidate
        return None

    def _read_evidence_snippet(
        self,
        repo_path: str,
        file_path: str,
        start_line: Any,
        end_line: Any,
        fallback_text: str,
    ) -> Tuple[str, bool]:
        abs_file = self._safe_abs_file(repo_path, file_path)
        if not abs_file or not os.path.exists(abs_file):
            return fallback_text[: self.cfg.evaluation.max_snippet_chars], False

        try:
            with open(abs_file, "r", encoding="utf-8", errors="ignore") as handle:
                lines = handle.read(self.cfg.io_limits.read_file_max_chars).splitlines()
            if not lines:
                return fallback_text[: self.cfg.evaluation.max_snippet_chars], False

            s_line = max(1, int(start_line))
            e_line = max(s_line, int(end_line))
            s_idx = min(len(lines), s_line) - 1
            e_idx = min(len(lines), e_line)
            snippet = "\n".join(lines[s_idx:e_idx]).strip()
            if not snippet:
                snippet = fallback_text
            return snippet[: self.cfg.evaluation.max_snippet_chars], True
        except Exception:  # noqa: BLE001
            return fallback_text[: self.cfg.evaluation.max_snippet_chars], False

    @staticmethod
    def _match_signals(signals: List[str], evidence_text: str) -> List[str]:
        haystack = evidence_text.lower()
        matched: List[str] = []
        for signal in signals:
            token = signal.strip().lower()
            if token and token in haystack:
                matched.append(signal)
        return matched

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _usage_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
        keys = {"requests", "prompt_tokens", "completion_tokens", "total_tokens", "missing_usage_responses"}
        out: Dict[str, int] = {}
        for key in keys:
            out[key] = max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        return out

    def _write_evaluation_log(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        os.makedirs(os.path.dirname(self.evaluation_log_path), exist_ok=True)
        with open(self.evaluation_log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    @staticmethod
    def _normalize_eval_status(raw: Any) -> str:
        value = str(raw or "").strip().lower()
        if value in {"detected", "pass", "present", "found"}:
            return "Detected"
        if value in {"not detected", "not_detected", "fail", "absent", "not found"}:
            return "Not Detected"
        return "Needs Review"

    def _evaluate_rule_with_llm(
        self,
        rule: Dict[str, Any],
        evidences: List[Dict[str, Any]],
        matched_strong: List[str],
        matched_weak: List[str],
        matched_false_pos: List[str],
    ) -> Dict[str, Any]:
        evidence_lines: List[str] = []
        for ev in evidences:
            citation_id = ev.get("citation_id")
            evidence_lines.append(
                (
                    f"[{citation_id}] file={ev.get('file_path')} lines={ev.get('start_line')}-{ev.get('end_line')} "
                    f"type={ev.get('node_type')} collection={ev.get('collection')} distance={ev.get('distance')}\n"
                    f"{ev.get('snippet', '')}"
                )
            )

        system_prompt = (
            "You are an audit evaluator. Return ONLY valid JSON object with keys: "
            "status, reason, confidence, evidence_ids, matched_strong_signals, "
            "matched_weak_signals, false_positive_risks. "
            "Allowed status values: Detected, Not Detected, Needs Review."
        )
        user_prompt = (
            f"Rule ID: {rule.get('id')}\n"
            f"Title: {rule.get('title')}\n"
            f"Definition: {rule.get('definition')}\n"
            f"Strong signals: {rule.get('strong_signals')}\n"
            f"Weak signals: {rule.get('weak_signals')}\n"
            f"False positives: {rule.get('false_positives')}\n"
            f"Deterministic matched strong signals: {matched_strong}\n"
            f"Deterministic matched weak signals: {matched_weak}\n"
            f"Deterministic matched false positives: {matched_false_pos}\n\n"
            "Evidence candidates:\n"
            + ("\n\n".join(evidence_lines) if evidence_lines else "No evidence found.")
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        previous_timeout = self.llm.timeout_s
        self.llm.timeout_s = int(self.cfg.evaluation.llm_timeout_s)
        try:
            llm_json = self.llm.chat_json(messages)
        finally:
            self.llm.timeout_s = previous_timeout
        status = self._normalize_eval_status(llm_json.get("status"))
        reason = str(llm_json.get("reason", "")).strip()
        confidence = max(0.0, min(1.0, self._coerce_float(llm_json.get("confidence"), default=0.0)))
        evidence_ids_raw = llm_json.get("evidence_ids", [])
        evidence_ids: List[int] = []
        if isinstance(evidence_ids_raw, list):
            for value in evidence_ids_raw:
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                evidence_ids.append(idx)
        if not evidence_ids:
            evidence_ids = [int(ev.get("citation_id")) for ev in evidences if isinstance(ev.get("citation_id"), int)]

        def _normalize_str_list(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            out: List[str] = []
            for item in value:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out

        return {
            "status": status,
            "reason": reason,
            "confidence": confidence,
            "evidence_ids": evidence_ids,
            "matched_strong_signals": _normalize_str_list(llm_json.get("matched_strong_signals")) or matched_strong,
            "matched_weak_signals": _normalize_str_list(llm_json.get("matched_weak_signals")) or matched_weak,
            "false_positive_risks": _normalize_str_list(llm_json.get("false_positive_risks")) or matched_false_pos,
        }

    def _render_evaluation_html(self, report_payload: Dict[str, Any]) -> str:
        summary = report_payload.get("summary", {})
        items = report_payload.get("items", [])
        rows: List[str] = []
        for item in items:
            status = str(item.get("status", "Needs Review"))
            status_class = (
                "status-detected" if status == "Detected" else "status-not-detected" if status == "Not Detected" else "status-needs-review"
            )
            evidences = item.get("evidences", [])
            evidence_html = []
            for ev in evidences:
                evidence_html.append(
                    "<li>"
                    f"<div class=\"evidence-head\"><strong>[{ev.get('citation_id')}]</strong> "
                    f"{html.escape(str(ev.get('file_path', '')))}:{html.escape(str(ev.get('start_line', '?')))}-"
                    f"{html.escape(str(ev.get('end_line', '?')))} "
                    f"({html.escape(str(ev.get('collection', '')))}/"
                    f"{html.escape(str(ev.get('node_type', '')))}, "
                    f"distance={html.escape(str(ev.get('distance', '')))}"
                    ")</div>"
                    f"<pre>{html.escape(str(ev.get('snippet', '')))}</pre>"
                    "</li>"
                )

            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('id', '')))}</td>"
                f"<td>{html.escape(str(item.get('title', '')))}</td>"
                f"<td><span class=\"status-pill {status_class}\">{html.escape(status)}</span></td>"
                f"<td>{len(evidences)}</td>"
                f"<td class=\"reason-cell\">{html.escape(str(item.get('reason', '')))}</td>"
                "</tr>"
                "<tr><td colspan=\"5\">"
                "<details><summary>Details</summary>"
                "<div class=\"detail-block\">"
                f"<p><strong>Definition:</strong> {html.escape(str(item.get('definition', '')))}</p>"
                f"<p><strong>Matched Strong Signals:</strong> {html.escape(', '.join(item.get('matched_strong_signals', [])))}</p>"
                f"<p><strong>Matched Weak Signals:</strong> {html.escape(', '.join(item.get('matched_weak_signals', [])))}</p>"
                f"<p><strong>False Positive Risks:</strong> {html.escape(', '.join(item.get('false_positive_risks', [])))}</p>"
                "<p><strong>Evidences:</strong></p>"
                f"<ul class=\"evidence-list\">{''.join(evidence_html)}</ul>"
                "</div>"
                "</details>"
                "</td></tr>"
            )

        return (
            "<!doctype html><html><head><meta charset=\"utf-8\"/>"
            f"<title>{html.escape(self.cfg.evaluation.html_title)}</title>"
            "<style>"
            ":root{--td-green:#00853f;--td-green-dark:#006632;--td-green-soft:#ecf8f0;--td-border:#d4dfd8;--td-bg:#f4f7f5;"
            "--td-card:#ffffff;--td-text:#1f2d23;--td-muted:#5f6f64;--td-warning:#775200;--td-warning-bg:#fff7df;}"
            "*{box-sizing:border-box;}"
            "html,body{margin:0;min-height:100%;}"
            "body{font-family:'Segoe UI','Helvetica Neue',Arial,sans-serif;color:var(--td-text);"
            "background:linear-gradient(180deg,#f7faf8 0%,#f0f5f2 100%);}"
            ".td-root{min-height:100vh;padding:18px;}"
            ".report-shell{max-width:1200px;margin:0 auto;display:flex;flex-direction:column;gap:12px;}"
            ".card{background:var(--td-card);border:1px solid var(--td-border);border-radius:8px;padding:14px;"
            "box-shadow:0 4px 14px rgba(16,43,28,0.05);}"
            "h1{margin:0;font-size:24px;line-height:1.2;}"
            ".subtitle{margin-top:6px;color:var(--td-muted);font-size:13px;overflow-wrap:anywhere;word-break:break-word;}"
            ".meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px;margin-top:12px;"
            "font-size:12px;color:#345642;}"
            ".meta-cell{overflow-wrap:anywhere;word-break:break-word;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;}"
            ".summary-card{border:1px solid var(--td-border);border-radius:8px;padding:10px 12px;background:#fff;}"
            ".summary-card .label{font-size:12px;color:var(--td-muted);text-transform:uppercase;letter-spacing:0.3px;}"
            ".summary-card .value{font-size:22px;font-weight:700;color:#214934;margin-top:3px;}"
            "table{border-collapse:collapse;width:100%;background:#fff;}"
            "th,td{border:1px solid var(--td-border);padding:8px;vertical-align:top;}"
            "th{background:#f3f8f5;text-align:left;font-size:12px;color:#2e503b;letter-spacing:0.2px;}"
            "td{font-size:13px;overflow-wrap:anywhere;word-break:break-word;}"
            ".reason-cell{max-width:500px;}"
            "pre{margin:8px 0 0;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;"
            "background:#fff;border:1px solid #dce7e1;padding:8px;border-radius:6px;font-size:12px;color:#294936;}"
            ".status-pill{padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;}"
            ".status-detected{background:#e7f7ee;color:#136b39;border:1px solid #9ed3b2;}"
            ".status-not-detected{background:#f0f0f0;color:#444;border:1px solid #cfcfcf;}"
            ".status-needs-review{background:var(--td-warning-bg);color:var(--td-warning);border:1px solid #efcc8d;}"
            "details>summary{cursor:pointer;font-weight:700;color:#2f6043;}"
            ".detail-block p{margin:8px 0;font-size:13px;overflow-wrap:anywhere;word-break:break-word;}"
            ".evidence-list{list-style:none;margin:8px 0 0;padding:0;display:flex;flex-direction:column;gap:8px;}"
            ".evidence-list li{border:1px solid #cfddd5;border-radius:8px;padding:8px;background:#f7fbf8;}"
            ".evidence-head{font-size:12px;color:#304f3d;overflow-wrap:anywhere;word-break:break-word;}"
            "@media (max-width: 900px){.td-root{padding:10px;}table{font-size:12px;}}"
            "</style></head><body><div class=\"td-root\"><div class=\"report-shell\">"
            "<section class=\"card\">"
            f"<h1>{html.escape(self.cfg.evaluation.html_title)}</h1>"
            "<div class=\"subtitle\">Audit evaluation report generated by Vyom.</div>"
            "<div class=\"meta-grid\">"
            f"<div class=\"meta-cell\"><strong>Repo Store:</strong> {html.escape(str(report_payload.get('repo_name', '')))}</div>"
            f"<div class=\"meta-cell\"><strong>Repo Path:</strong> {html.escape(str(report_payload.get('repo_path', '')))}</div>"
            f"<div class=\"meta-cell\"><strong>Generated:</strong> {html.escape(str(report_payload.get('generated_at_utc', '')))}</div>"
            f"<div class=\"meta-cell\"><strong>Rules Version:</strong> {html.escape(str(report_payload.get('rules_version', '')))}</div>"
            f"<div class=\"meta-cell\"><strong>Rules Hash:</strong> {html.escape(str(report_payload.get('rules_hash', '')))}</div>"
            "</div></section>"
            "<section class=\"card\"><div class=\"summary-cards\">"
            f"<div class=\"summary-card\"><div class=\"label\">Detected</div><div class=\"value\">{int(summary.get('detected', 0))}</div></div>"
            f"<div class=\"summary-card\"><div class=\"label\">Not Detected</div><div class=\"value\">{int(summary.get('not_detected', 0))}</div></div>"
            f"<div class=\"summary-card\"><div class=\"label\">Needs Review</div><div class=\"value\">{int(summary.get('needs_review', 0))}</div></div>"
            "</div></section>"
            "<section class=\"card\"><table><thead><tr>"
            "<th>ID</th><th>Evaluation Item</th><th>Status</th><th>Evidence Count</th><th>Reason</th>"
            "</tr></thead><tbody>"
            f"{''.join(rows)}"
            "</tbody></table></section></div></div></body></html>"
        )

    def run_evaluation(self, repo_name: str, repo_path: str, job_id: str = "") -> Dict[str, Any]:
        clean_repo_name = repo_name.strip()
        resolved_repo_path = os.path.abspath(repo_path.strip())
        if not clean_repo_name:
            raise ValueError("repo_name must be a non-empty string")
        if not os.path.isdir(resolved_repo_path):
            raise ValueError(f"repo_path must exist and be a directory: {resolved_repo_path}")

        rules_payload = self.get_evaluation_rules()
        rules = list(rules_payload.get("items", []))
        rules_text = json.dumps(rules_payload, ensure_ascii=False, sort_keys=True)
        rules_hash = hashlib.sha256(rules_text.encode("utf-8")).hexdigest()
        started = time.perf_counter()
        llm_usage_before = self.llm.usage()
        status_counts = {"detected": 0, "not_detected": 0, "needs_review": 0}
        items_out: List[Dict[str, Any]] = []
        partial = False
        total_candidates = 0

        self._write_evaluation_log({
            "timestamp_utc": self._utc_now(),
            "event": "evaluation_start",
            "job_id": job_id,
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "rule_count": len(rules),
        })

        for rule in rules:
            rule_id = str(rule.get("id", "")).strip()
            preferred_limits = self._evaluation_preferred_limits(rule_id)
            base_limits = self._evaluation_base_limits()
            queries = [
                f"{rule.get('title', '')}\n{rule.get('definition', '')}".strip(),
                " ".join(rule.get("strong_signals", [])),
                " ".join(rule.get("weak_signals", [])),
            ]

            candidate_map: Dict[str, Dict[str, Any]] = {}
            retrieval_failures: List[str] = []
            for query in queries:
                if not query.strip():
                    continue
                sources: List[Dict[str, Any]] = []
                if preferred_limits:
                    try:
                        preferred_sources, _ = self.retrieve(
                            query,
                            [clean_repo_name],
                            collection_limits=preferred_limits,
                            intent_override=f"evaluation_{rule_id}_preferred",
                        )
                        sources.extend(preferred_sources)
                    except Exception as exc:  # noqa: BLE001
                        retrieval_failures.append(f"preferred:{exc}")

                try:
                    base_sources, _ = self.retrieve(
                        query,
                        [clean_repo_name],
                        collection_limits=base_limits,
                        intent_override=f"evaluation_{rule_id}_base",
                    )
                    sources.extend(base_sources)
                except Exception as exc:  # noqa: BLE001
                    retrieval_failures.append(f"base:{exc}")
                    continue

                for src in sources:
                    meta = src.get("metadata") or {}
                    key = "|".join(
                        [
                            str(meta.get("file_path", "")),
                            str(meta.get("start_line", "")),
                            str(meta.get("end_line", "")),
                            str(meta.get("node_type", "")),
                        ]
                    )
                    if not key.strip("|"):
                        continue
                    existing = candidate_map.get(key)
                    if existing is None or float(src.get("distance", 1e9)) < float(existing.get("distance", 1e9)):
                        candidate_map[key] = src

            scored_candidates: List[Tuple[float, Dict[str, Any]]] = []
            for src in candidate_map.values():
                meta = src.get("metadata") or {}
                base_text = "\n".join(
                    [
                        str(meta.get("file_path", "")),
                        str(meta.get("symbol", "")),
                        str(meta.get("node_type", "")),
                        str(src.get("document", "")),
                    ]
                )
                strong_hits = self._match_signals(rule.get("strong_signals", []), base_text)
                weak_hits = self._match_signals(rule.get("weak_signals", []), base_text)
                overlap_score = (len(strong_hits) * 2.0) + len(weak_hits)
                distance = float(src.get("distance", 1e9))
                score = overlap_score - distance
                scored_candidates.append((score, src))

            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            max_candidates = max(1, self.cfg.evaluation.max_candidates_per_item)
            top_candidates = [src for _, src in scored_candidates[:max_candidates]]
            total_candidates += len(top_candidates)

            evidences: List[Dict[str, Any]] = []
            for idx, src in enumerate(top_candidates[: max(1, self.cfg.evaluation.evidence_per_item)], start=1):
                meta = src.get("metadata") or {}
                snippet, from_file = self._read_evidence_snippet(
                    repo_path=resolved_repo_path,
                    file_path=str(meta.get("file_path", "")),
                    start_line=meta.get("start_line", 1),
                    end_line=meta.get("end_line", meta.get("start_line", 1)),
                    fallback_text=str(src.get("document", "")),
                )
                evidences.append({
                    "citation_id": idx,
                    "collection": str(src.get("collection", "")),
                    "distance": round(self._coerce_float(src.get("distance"), 1e9), 6),
                    "file_path": str(meta.get("file_path", "")),
                    "start_line": meta.get("start_line", ""),
                    "end_line": meta.get("end_line", ""),
                    "snippet": snippet[: self.cfg.evaluation.max_snippet_chars],
                    "node_type": str(meta.get("node_type", "")),
                    "repo_name": str(meta.get("repo_name", clean_repo_name)),
                    "source_mode": "file" if from_file else "vector",
                })

            evidence_text = "\n".join(
                [
                    ev.get("snippet", "")
                    + "\n"
                    + ev.get("file_path", "")
                    + "\n"
                    + ev.get("node_type", "")
                    for ev in evidences
                ]
            )
            matched_strong = self._match_signals(rule.get("strong_signals", []), evidence_text)
            matched_weak = self._match_signals(rule.get("weak_signals", []), evidence_text)
            matched_false_pos = self._match_signals(rule.get("false_positives", []), evidence_text)

            if retrieval_failures:
                partial = True

            try:
                decision = self._evaluate_rule_with_llm(
                    rule=rule,
                    evidences=evidences,
                    matched_strong=matched_strong,
                    matched_weak=matched_weak,
                    matched_false_pos=matched_false_pos,
                )
            except Exception as exc:  # noqa: BLE001
                partial = True
                decision = {
                    "status": "Needs Review",
                    "reason": f"LLM evaluation failed: {exc}",
                    "confidence": 0.0,
                    "evidence_ids": [ev.get("citation_id") for ev in evidences],
                    "matched_strong_signals": matched_strong,
                    "matched_weak_signals": matched_weak,
                    "false_positive_risks": matched_false_pos,
                }

            status = self._normalize_eval_status(decision.get("status"))
            if status == "Detected":
                status_counts["detected"] += 1
            elif status == "Not Detected":
                status_counts["not_detected"] += 1
            else:
                status_counts["needs_review"] += 1

            items_out.append({
                "id": str(rule.get("id", "")),
                "title": str(rule.get("title", "")),
                "definition": str(rule.get("definition", "")),
                "status": status,
                "reason": str(decision.get("reason", "")),
                "confidence": round(self._coerce_float(decision.get("confidence"), 0.0), 6),
                "matched_strong_signals": list(decision.get("matched_strong_signals", [])),
                "matched_weak_signals": list(decision.get("matched_weak_signals", [])),
                "false_positive_risks": list(decision.get("false_positive_risks", [])),
                "evidences": evidences,
                "candidate_count": len(top_candidates),
                "retrieval_failures": retrieval_failures,
            })

        generated_at = self._utc_now()
        elapsed_seconds = round(time.perf_counter() - started, 6)
        llm_usage_delta = self._usage_delta(llm_usage_before, self.llm.usage())
        status = "completed_partial" if partial else "completed"

        summary_payload = {
            "status": status,
            "job_id": job_id,
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "generated_at_utc": generated_at,
            "rule_count": len(rules),
            "status_counts": status_counts,
            "llm_usage": llm_usage_delta,
            "elapsed_seconds": elapsed_seconds,
            "total_candidates": total_candidates,
        }

        report_payload = {
            "report_type": "evaluation_report",
            "job_id": job_id,
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "generated_at_utc": generated_at,
            "rules_version": int(rules_payload.get("version", 1)),
            "rules_hash": rules_hash,
            "summary": status_counts,
            "items": items_out,
        }

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_repo_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", clean_repo_name)
        report_base = f"evaluation_report_{safe_repo_name}_{ts}"
        report_json_path = os.path.join(self.evaluation_reports_dir, report_base + ".json")
        report_html_path = os.path.join(self.evaluation_reports_dir, report_base + ".html")
        self._write_json_file(report_json_path, report_payload)
        html_text = self._render_evaluation_html(report_payload)
        with open(report_html_path, "w", encoding="utf-8") as handle:
            handle.write(html_text)

        summary_payload["report_json_path"] = report_json_path
        summary_payload["report_html_path"] = report_html_path
        summary_payload["log_path"] = self.evaluation_log_path

        self._write_evaluation_log({
            "timestamp_utc": generated_at,
            "event": "evaluation_completed",
            "job_id": job_id,
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "status": status,
            "rule_count": len(rules),
            "status_counts": status_counts,
            "llm_usage": llm_usage_delta,
            "elapsed_seconds": elapsed_seconds,
            "total_candidates": total_candidates,
            "report_json_path": report_json_path,
            "report_html_path": report_html_path,
        })
        logger.info(
            (
                "Evaluation run completed: repo_name=%s status=%s rules=%d detected=%d "
                "not_detected=%d needs_review=%d elapsed_seconds=%.2f report_json=%s report_html=%s"
            ),
            clean_repo_name,
            status,
            len(rules),
            status_counts["detected"],
            status_counts["not_detected"],
            status_counts["needs_review"],
            elapsed_seconds,
            report_json_path,
            report_html_path,
        )
        return summary_payload

    def discover_store_names(self) -> List[str]:
        sample_size = self.cfg.chat.store_discovery.sample_size_per_collection
        max_store_names = self.cfg.chat.store_discovery.max_store_names
        discovered = set()

        collection_names = []
        seen_collections = set()
        for collection_name in self.collection_map.values():
            if collection_name in seen_collections:
                continue
            collection_names.append(collection_name)
            seen_collections.add(collection_name)

        logger.info(
            "Store discovery started: collections=%d sample_size_per_collection=%d max_store_names=%d",
            len(collection_names),
            sample_size,
            max_store_names,
        )

        for collection_name in collection_names:
            if len(discovered) >= max_store_names:
                break
            try:
                collection_count = self.chroma.count(collection_name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Store discovery count failed: collection=%s error=%s", collection_name, exc)
                continue

            if collection_count <= 0:
                continue

            to_scan = min(collection_count, sample_size)
            offset = 0
            while offset < to_scan and len(discovered) < max_store_names:
                page_limit = min(200, to_scan - offset)
                try:
                    metadatas = self.chroma.get_metadatas(
                        collection_name=collection_name,
                        limit=page_limit,
                        offset=offset,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Store discovery metadata read failed: collection=%s offset=%d limit=%d error=%s",
                        collection_name,
                        offset,
                        page_limit,
                        exc,
                    )
                    break

                if not metadatas:
                    break

                for meta in metadatas:
                    repo_name = meta.get("repo_name")
                    if isinstance(repo_name, str) and repo_name.strip():
                        discovered.add(repo_name.strip())
                        if len(discovered) >= max_store_names:
                            break

                offset += len(metadatas)
                if len(metadatas) < page_limit:
                    break

        out = sorted(discovered)[:max_store_names]
        logger.info("Store discovery completed: discovered=%d", len(out))
        return out

    def retrieve(
        self,
        query: str,
        store_names: List[str],
        collection_limits: Optional[Dict[str, int]] = None,
        intent_override: str = "",
    ) -> Tuple[List[Dict[str, Any]], str]:
        normalized_stores = self._normalize_store_names(store_names)
        if not normalized_stores:
            raise ValueError("store_names must contain at least one non-empty repo tag")

        logger.info(
            "Store retrieval started: stores=%d query_chars=%d",
            len(normalized_stores),
            len(query),
        )

        query_vector = self.embedder.embed([query])[0]
        if collection_limits is None:
            intent = intent_override.strip() or self._detect_query_intent(query)
            weighted_limits = self._weighted_limits(intent)
        else:
            intent = intent_override.strip() or "manual"
            weighted_limits = {}
            for key, value in collection_limits.items():
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    weighted_limits[key] = parsed
        ranked: List[Dict[str, Any]] = []
        debug_collections: Dict[str, Any] = {}

        for collection_key, collection_budget in weighted_limits.items():
            collection_name = self.collection_map.get(collection_key)
            if not collection_name:
                continue
            debug_entry: Dict[str, Any] = {
                "collection": collection_name,
                "budget": collection_budget,
                "collection_count": None,
                "store_hits": {},
            }
            try:
                collection_count = self.chroma.count(collection_name)
                debug_entry["collection_count"] = collection_count
                if collection_count <= 0:
                    debug_collections[collection_key] = debug_entry
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.warning("Collection count failed: collection=%s error=%s", collection_name, exc)
                debug_entry["count_error"] = str(exc)
                debug_collections[collection_key] = debug_entry
                continue

            per_store_limits = self._split_budget(collection_budget, len(normalized_stores))
            for store_name, n_results in zip(normalized_stores, per_store_limits):
                if n_results <= 0:
                    debug_entry["store_hits"][store_name] = {"requested": n_results, "returned": 0}
                    continue
                used_manual_filter = False
                used_text_fallback = False
                filtered_by_repo_count = 0
                text_fallback_candidates = 0
                store_query_error = ""
                try:
                    res = self.chroma.query(
                        collection_name=collection_name,
                        query_embeddings=[query_vector],
                        n_results=n_results,
                        where={"repo_name": store_name},
                    )
                except Exception as exc:  # noqa: BLE001
                    store_query_error = str(exc)
                    logger.warning(
                        "Collection query failed: collection=%s store=%s n_results=%d error=%s",
                        collection_name,
                        store_name,
                        n_results,
                        exc,
                    )
                    # Fallback for stores where Chroma metadata filtering can fail
                    # with internal plan/index errors: query without where and filter
                    # repo_name in-memory.
                    try:
                        fallback_n_results = max(n_results * 8, n_results)
                        if isinstance(collection_count, int) and collection_count > 0:
                            fallback_n_results = min(fallback_n_results, collection_count)
                        res = self.chroma.query(
                            collection_name=collection_name,
                            query_embeddings=[query_vector],
                            n_results=fallback_n_results,
                        )
                        used_manual_filter = True
                    except Exception as fallback_exc:  # noqa: BLE001
                        logger.warning(
                            (
                                "Collection query fallback failed: collection=%s store=%s "
                                "n_results=%d fallback_n_results=%d error=%s"
                            ),
                            collection_name,
                            store_name,
                            n_results,
                            fallback_n_results if "fallback_n_results" in locals() else n_results,
                            fallback_exc,
                        )
                        try:
                            fallback_scan_limit = max(n_results * 64, n_results)
                            if isinstance(collection_count, int) and collection_count > 0:
                                fallback_scan_limit = min(fallback_scan_limit, collection_count)
                            res = self._build_text_fallback_res(
                                query=query,
                                collection_name=collection_name,
                                store_name=store_name,
                                n_results=n_results,
                                scan_limit=fallback_scan_limit,
                            )
                            text_fallback_candidates = int(res.get("fallback_candidates", 0))
                            used_text_fallback = True
                            logger.warning(
                                (
                                    "Collection text fallback used: collection=%s store=%s "
                                    "n_results=%d candidates=%d"
                                ),
                                collection_name,
                                store_name,
                                n_results,
                                text_fallback_candidates,
                            )
                        except Exception as text_fallback_exc:  # noqa: BLE001
                            logger.warning(
                                (
                                    "Collection text fallback failed: collection=%s store=%s "
                                    "n_results=%d scan_limit=%d error=%s"
                                ),
                                collection_name,
                                store_name,
                                n_results,
                                fallback_scan_limit if "fallback_scan_limit" in locals() else n_results,
                                text_fallback_exc,
                            )
                            debug_entry["store_hits"][store_name] = {
                                "requested": n_results,
                                "returned": 0,
                                "error": store_query_error,
                                "fallback_error": str(fallback_exc),
                                "text_fallback_error": str(text_fallback_exc),
                            }
                            continue

                ids = res.get("ids", [[]])[0]
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                distances = res.get("distances", [[]])[0]

                if used_manual_filter:
                    filtered_ids: List[Any] = []
                    filtered_docs: List[Any] = []
                    filtered_metas: List[Any] = []
                    filtered_distances: List[Any] = []
                    for idx, node_id in enumerate(ids):
                        metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
                        repo_name = metadata.get("repo_name") if isinstance(metadata, dict) else None
                        if repo_name != store_name:
                            continue
                        filtered_ids.append(node_id)
                        filtered_docs.append(docs[idx] if idx < len(docs) else "")
                        filtered_metas.append(metadata if isinstance(metadata, dict) else {})
                        filtered_distances.append(distances[idx] if idx < len(distances) else 1e9)
                    filtered_by_repo_count = len(filtered_ids)
                    ids = filtered_ids[:n_results]
                    docs = filtered_docs[:n_results]
                    metas = filtered_metas[:n_results]
                    distances = filtered_distances[:n_results]

                debug_entry["store_hits"][store_name] = {
                    "requested": n_results,
                    "returned": len(ids),
                    "used_manual_filter": used_manual_filter,
                    "used_text_fallback": used_text_fallback,
                    "text_fallback_candidates": text_fallback_candidates if used_text_fallback else 0,
                    "filtered_by_repo_count": filtered_by_repo_count if used_manual_filter else len(ids),
                    "error": store_query_error if (used_manual_filter or used_text_fallback) and store_query_error else "",
                }

                for idx, node_id in enumerate(ids):
                    metadata = metas[idx] if idx < len(metas) and metas[idx] else {}
                    if "repo_name" not in metadata:
                        metadata["repo_name"] = store_name
                    ranked.append({
                        "node_id": node_id,
                        "document": docs[idx] if idx < len(docs) else "",
                        "metadata": metadata,
                        "distance": float(distances[idx]) if idx < len(distances) else 1e9,
                        "collection": collection_name,
                        "collection_key": collection_key,
                        "intent": intent,
                    })
            debug_collections[collection_key] = debug_entry

        ranked.sort(key=lambda item: item["distance"])
        cap = max(1, self.cfg.chat.max_context_chunks)
        out = ranked[:cap]
        self._last_retrieve_debug = {
            "query": query,
            "intent": intent,
            "repo_stores": normalized_stores,
            "weighted_limits": weighted_limits,
            "collections": debug_collections,
            "candidates": len(ranked),
            "returned": len(out),
            "max_context_chunks": cap,
        }
        logger.info(
            "Store retrieval completed: intent=%s candidates=%d returned=%d",
            intent,
            len(ranked),
            len(out),
        )
        return out, intent

    def chat_turn(
        self,
        prompt: str,
        history: List[Dict[str, Any]],
        store_names: List[str],
        session_id: str = "",
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        ts = datetime.now(timezone.utc).isoformat()
        normalized_stores = self._normalize_store_names(store_names)
        if not normalized_stores:
            raise ValueError("store_names must contain at least one non-empty repo tag")

        raw_sources, intent = self.retrieve(prompt, normalized_stores)
        sources = self._annotate_sources_with_citations(raw_sources)
        if not sources:
            response = self._no_source_response(intent=intent, store_names=normalized_stores)
            cited_indices: List[int] = []
            model_used = ""
            logger.warning(
                "Chat turn had zero retrieved sources: intent=%s stores=%s",
                intent,
                ",".join(normalized_stores),
            )
        else:
            context_text = self._format_context(sources)
            messages = self._build_messages(
                history=history,
                context_text=context_text,
                intent=intent,
                store_names=normalized_stores,
            )
            response = self.llm.chat(messages, temperature=self.cfg.chat.temperature)
            response = self._ensure_citations(response, len(sources))
            cited_indices = self._extract_citation_indices(response, len(sources))
            model_used = self.llm.model

        elapsed = time.perf_counter() - started
        log_payload = self._build_chat_log_payload(
            timestamp_utc=ts,
            prompt=prompt,
            intent=intent,
            repo_stores=normalized_stores,
            sources=sources,
            cited_indices=cited_indices,
            model=model_used,
            elapsed_seconds=elapsed,
            session_id=session_id,
        )
        self._write_retrieve_log(log_payload)

        return {
            "response": response,
            "sources": sources,
            "intent": intent,
            "cited_indices": cited_indices,
            "retrieve_log_path": self.retrieve_log_path,
        }

    def chat_turn_stream(
        self,
        prompt: str,
        history: List[Dict[str, Any]],
        store_names: List[str],
        session_id: str = "",
    ) -> Iterator[Dict[str, Any]]:
        started = time.perf_counter()
        ts = datetime.now(timezone.utc).isoformat()
        normalized_stores = self._normalize_store_names(store_names)
        if not normalized_stores:
            raise ValueError("store_names must contain at least one non-empty repo tag")

        raw_sources, intent = self.retrieve(prompt, normalized_stores)
        sources = self._annotate_sources_with_citations(raw_sources)

        yield {
            "event": "meta",
            "intent": intent,
            "repo_stores": normalized_stores,
            "source_count": len(sources),
            "no_sources": len(sources) == 0,
            "sources": sources,
        }

        if not sources:
            response = self._no_source_response(intent=intent, store_names=normalized_stores)
            cited_indices: List[int] = []
            yield {"event": "token", "token": response}
            done_payload = {
                "event": "done",
                "response": response,
                "sources": sources,
                "intent": intent,
                "cited_indices": cited_indices,
                "no_sources": True,
                "model": "",
            }
            yield done_payload
            elapsed = time.perf_counter() - started
            self._write_retrieve_log(
                self._build_chat_log_payload(
                    timestamp_utc=ts,
                    prompt=prompt,
                    intent=intent,
                    repo_stores=normalized_stores,
                    sources=sources,
                    cited_indices=cited_indices,
                    model="",
                    elapsed_seconds=elapsed,
                    session_id=session_id,
                )
            )
            return

        context_text = self._format_context(sources)
        messages = self._build_messages(
            history=history,
            context_text=context_text,
            intent=intent,
            store_names=normalized_stores,
        )
        response_parts: List[str] = []
        for token in self.llm.chat_stream(messages, temperature=self.cfg.chat.temperature):
            if not token:
                continue
            response_parts.append(token)
            yield {"event": "token", "token": token}

        raw_response = "".join(response_parts)
        response = self._ensure_citations(raw_response, len(sources))
        if response != raw_response:
            if response.startswith(raw_response):
                extra = response[len(raw_response) :]
            else:
                extra = "\n\n" + response
            if extra:
                yield {"event": "token", "token": extra}

        cited_indices = self._extract_citation_indices(response, len(sources))
        done_payload = {
            "event": "done",
            "response": response,
            "sources": sources,
            "intent": intent,
            "cited_indices": cited_indices,
            "no_sources": False,
            "model": self.llm.model,
        }
        yield done_payload

        elapsed = time.perf_counter() - started
        self._write_retrieve_log(
            self._build_chat_log_payload(
                timestamp_utc=ts,
                prompt=prompt,
                intent=intent,
                repo_stores=normalized_stores,
                sources=sources,
                cited_indices=cited_indices,
                model=self.llm.model,
                elapsed_seconds=elapsed,
                session_id=session_id,
            )
        )
