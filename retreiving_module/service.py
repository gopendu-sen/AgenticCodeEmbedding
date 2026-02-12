import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from agentic_rag.agentic_ai.orchestrator import AgenticRagOrchestrator
from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.llm_client import LLMClient
from agentic_rag.embedding.chroma_store import ChromaStore
from agentic_rag.embedding.client import EmbeddingClient


logger = logging.getLogger(__name__)
_CITATION_RE = re.compile(r"\[(\d+)\]")
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
        os.makedirs(cfg.paths.reports_dir, exist_ok=True)
        os.makedirs(self.embedding_jobs_dir, exist_ok=True)
        self._last_retrieve_debug: Dict[str, Any] = {}

        cols = cfg.embedding.collections
        self.collection_map: Dict[str, str] = {
            "code_symbols": cols.code_symbols,
            "code_routes": cols.code_routes,
            "code_usage": cols.code_usage,
            "configs": cols.configs,
            "docs": cols.docs,
            "security_tags": cols.security_tags,
            "flows": cols.flows,
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
        path = self._job_status_path(job_id)
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(job, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _refresh_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        status = str(job.get("status", "")).strip().lower()
        pid_raw = job.get("pid")
        try:
            pid = int(pid_raw) if pid_raw is not None else 0
        except (TypeError, ValueError):
            pid = 0

        if status != "running":
            return job

        if pid > 0 and self._is_pid_alive(pid):
            return job

        job["status"] = "failed"
        if not str(job.get("error", "")).strip():
            job["error"] = "Embedding worker process exited unexpectedly"
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
            "repo_name": clean_repo_name,
            "repo_path": resolved_repo_path,
            "config_path": self.config_path,
            "created_at_utc": created_at,
            "updated_at_utc": created_at,
            "pid": None,
            "summary": None,
            "error": "",
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

        with open(self._job_log_path(job_id), "a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=self.project_root,
            )

        job["status"] = "running"
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

    def retrieve(self, query: str, store_names: List[str]) -> Tuple[List[Dict[str, Any]], str]:
        normalized_stores = self._normalize_store_names(store_names)
        if not normalized_stores:
            raise ValueError("store_names must contain at least one non-empty repo tag")

        logger.info(
            "Store retrieval started: stores=%d query_chars=%d",
            len(normalized_stores),
            len(query),
        )

        query_vector = self.embedder.embed([query])[0]
        intent = self._detect_query_intent(query)
        weighted_limits = self._weighted_limits(intent)
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
                try:
                    res = self.chroma.query(
                        collection_name=collection_name,
                        query_embeddings=[query_vector],
                        n_results=n_results,
                        where={"repo_name": store_name},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Collection query failed: collection=%s store=%s n_results=%d error=%s",
                        collection_name,
                        store_name,
                        n_results,
                        exc,
                    )
                    debug_entry["store_hits"][store_name] = {
                        "requested": n_results,
                        "returned": 0,
                        "error": str(exc),
                    }
                    continue

                ids = res.get("ids", [[]])[0]
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                distances = res.get("distances", [[]])[0]
                debug_entry["store_hits"][store_name] = {
                    "requested": n_results,
                    "returned": len(ids),
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
        log_payload = {
            "timestamp_utc": ts,
            "event": "chat_turn",
            "query": prompt,
            "intent": intent,
            "repo_stores": normalized_stores,
            "source_count": len(sources),
            "no_sources": len(sources) == 0,
            "cited_indices": cited_indices,
            "model": model_used,
            "elapsed_seconds": round(elapsed, 6),
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
        self._write_retrieve_log(log_payload)

        return {
            "response": response,
            "sources": sources,
            "intent": intent,
            "cited_indices": cited_indices,
            "retrieve_log_path": self.retrieve_log_path,
        }
