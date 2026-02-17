"""FastAPI backend for embedding/evaluation operations + embedding proxy."""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests
import uvicorn

from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.config_loader import load_agentic_rag_config
from retreiving_module import StoreRetriever


logger = logging.getLogger(__name__)


class EmbeddingJobRequest(BaseModel):
    repo_path: str
    repo_name: str

    @field_validator("repo_path", "repo_name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned


class EvaluationJobRequest(BaseModel):
    repo_path: str
    repo_name: str

    @field_validator("repo_path", "repo_name")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned


class EvaluationRuleRequest(BaseModel):
    id: str
    title: str
    definition: str
    strong_signals: List[str]
    weak_signals: List[str]
    false_positives: List[str]

    @field_validator("id", "title", "definition")
    @classmethod
    def _required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned

    @field_validator("strong_signals", "weak_signals", "false_positives")
    @classmethod
    def _required_text_list(cls, values: List[str]) -> List[str]:
        cleaned = [str(item).strip() for item in values if str(item).strip()]
        if not cleaned:
            raise ValueError("list must contain at least one non-empty value")
        return cleaned


class EvaluationRulesRequest(BaseModel):
    version: int = Field(..., ge=1)
    updated_at_utc: Optional[str] = None
    items: List[EvaluationRuleRequest] = Field(..., min_length=1)


class EmbeddingProxyRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    input: Any


def _normalize_embedding_input(raw_input: Any) -> List[str]:
    if isinstance(raw_input, str):
        candidates = [raw_input]
    elif isinstance(raw_input, list):
        candidates = raw_input
    else:
        raise ValueError("input must be a string or a list of strings")

    normalized: List[str] = []
    for item in candidates:
        if not isinstance(item, str):
            raise ValueError("input items must be strings")
        cleaned = item.strip()
        if not cleaned:
            raise ValueError("input items must not be empty")
        normalized.append(cleaned)

    if not normalized:
        raise ValueError("input must contain at least one non-empty string")
    return normalized


def create_app(cfg: AgenticRagConfig, *, config_path: str) -> FastAPI:
    retriever = StoreRetriever.from_config(cfg, config_path=config_path)

    app = FastAPI(title="Agentic RAG Ops API", version="1.0.0")
    app.state.cfg = cfg
    app.state.retriever = retriever

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.chat.api.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_http_requests(request: Request, call_next):
        started = time.perf_counter()
        client_host = request.client.host if request.client else "unknown"
        query = request.url.query
        logger.info(
            "REST request started: method=%s path=%s query=%s client=%s",
            request.method,
            request.url.path,
            query,
            client_host,
        )
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "REST request failed: method=%s path=%s client=%s elapsed_ms=%.2f error=%s",
                request.method,
                request.url.path,
                client_host,
                elapsed_ms,
                exc,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "REST request completed: method=%s path=%s status=%d client=%s elapsed_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            client_host,
            elapsed_ms,
        )
        return response

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok"}

    @app.get("/stores")
    async def stores() -> Dict[str, Any]:
        try:
            discovered = retriever.discover_store_names()
            logger.info("REST /stores completed: discovered=%d", len(discovered))
            return {"stores": discovered}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Store discovery failed")
            raise HTTPException(status_code=500, detail=f"Store discovery failed: {exc}") from exc

    @app.post("/embedding/jobs")
    async def start_embedding_job(request: EmbeddingJobRequest) -> Dict[str, Any]:
        logger.info(
            "REST /embedding/jobs called: repo_name=%s repo_path=%s",
            request.repo_name,
            request.repo_path,
        )
        try:
            job = retriever.start_embedding_job(request.repo_path, request.repo_name)
            logger.info(
                "REST /embedding/jobs started: job_id=%s status=%s repo_name=%s",
                job.get("job_id"),
                job.get("status"),
                job.get("repo_name"),
            )
            return job
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start embedding job")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/embedding/jobs")
    async def list_embedding_jobs(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
        logger.info("REST /embedding/jobs called: limit=%d", limit)
        try:
            jobs = retriever.list_embedding_jobs(limit=limit)
            logger.info("REST /embedding/jobs completed: returned=%d", len(jobs))
            return {"jobs": jobs}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list embedding jobs")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/embedding/jobs/{job_id}")
    async def get_embedding_job(job_id: str) -> Dict[str, Any]:
        logger.info("REST /embedding/jobs/{job_id} called: job_id=%s", job_id)
        try:
            job = retriever.get_embedding_job(job_id)
            logger.info(
                "REST /embedding/jobs/{job_id} completed: job_id=%s status=%s",
                job_id,
                job.get("status"),
            )
            return job
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch embedding job: job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/evaluation/rules")
    async def get_evaluation_rules() -> Dict[str, Any]:
        logger.info("REST /evaluation/rules called")
        try:
            payload = retriever.get_evaluation_rules()
            logger.info(
                "REST /evaluation/rules completed: version=%s items=%d",
                payload.get("version"),
                len(payload.get("items", [])),
            )
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load evaluation rules")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.put("/evaluation/rules")
    async def put_evaluation_rules(payload: EvaluationRulesRequest) -> Dict[str, Any]:
        logger.info("REST /evaluation/rules PUT called")
        try:
            saved = retriever.save_evaluation_rules(payload.model_dump(exclude_none=True))
            logger.info(
                "REST /evaluation/rules PUT completed: version=%s items=%d",
                saved.get("version"),
                len(saved.get("items", [])),
            )
            return saved
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to save evaluation rules")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/evaluation/jobs")
    async def start_evaluation_job(request: EvaluationJobRequest) -> Dict[str, Any]:
        logger.info(
            "REST /evaluation/jobs called: repo_name=%s repo_path=%s",
            request.repo_name,
            request.repo_path,
        )
        try:
            job = retriever.start_evaluation_job(repo_name=request.repo_name, repo_path=request.repo_path)
            logger.info(
                "REST /evaluation/jobs started: job_id=%s status=%s repo_name=%s",
                job.get("job_id"),
                job.get("status"),
                job.get("repo_name"),
            )
            return job
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to start evaluation job")
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/evaluation/jobs")
    async def list_evaluation_jobs(limit: int = Query(default=20, ge=1, le=200)) -> Dict[str, Any]:
        logger.info("REST /evaluation/jobs called: limit=%d", limit)
        try:
            jobs = retriever.list_evaluation_jobs(limit=limit)
            logger.info("REST /evaluation/jobs completed: returned=%d", len(jobs))
            return {"jobs": jobs}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to list evaluation jobs")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/evaluation/jobs/{job_id}")
    async def get_evaluation_job(job_id: str) -> Dict[str, Any]:
        logger.info("REST /evaluation/jobs/{job_id} called: job_id=%s", job_id)
        try:
            job = retriever.get_evaluation_job(job_id)
            logger.info(
                "REST /evaluation/jobs/{job_id} completed: job_id=%s status=%s",
                job_id,
                job.get("status"),
            )
            return job
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch evaluation job: job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/evaluation/jobs/{job_id}/html")
    async def get_evaluation_html(job_id: str) -> FileResponse:
        logger.info("REST /evaluation/jobs/{job_id}/html called: job_id=%s", job_id)
        try:
            job = retriever.get_evaluation_job(job_id)
            html_path = str(job.get("report_html_path", "")).strip()
            if not html_path and isinstance(job.get("summary"), dict):
                html_path = str((job.get("summary") or {}).get("report_html_path", "")).strip()
            if not html_path:
                raise FileNotFoundError(f"HTML report not ready for job {job_id}")
            if not os.path.exists(html_path):
                raise FileNotFoundError(f"HTML report file missing: {html_path}")
            return FileResponse(path=html_path, media_type="text/html", filename=os.path.basename(html_path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch evaluation HTML: job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/evaluation/jobs/{job_id}/json")
    async def get_evaluation_json(job_id: str) -> FileResponse:
        logger.info("REST /evaluation/jobs/{job_id}/json called: job_id=%s", job_id)
        try:
            job = retriever.get_evaluation_job(job_id)
            json_path = str(job.get("report_json_path", "")).strip()
            if not json_path and isinstance(job.get("summary"), dict):
                json_path = str((job.get("summary") or {}).get("report_json_path", "")).strip()
            if not json_path:
                raise FileNotFoundError(f"JSON report not ready for job {job_id}")
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON report file missing: {json_path}")
            return FileResponse(path=json_path, media_type="application/json", filename=os.path.basename(json_path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to fetch evaluation JSON: job_id=%s", job_id)
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/embeddings")
    async def proxy_embeddings(payload: EmbeddingProxyRequest) -> Response:
        try:
            normalized_input = _normalize_embedding_input(payload.input)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        upstream_model = (payload.model or "").strip() or (cfg.embedding.model or "").strip()
        if not upstream_model:
            raise HTTPException(status_code=500, detail="No embedding model configured")

        forward_payload = payload.model_dump(exclude_none=True)
        forward_payload["model"] = upstream_model
        forward_payload["input"] = normalized_input
        upstream_url = f"{cfg.embedding.base_url.rstrip('/')}/embeddings"

        started = time.perf_counter()
        logger.info(
            "REST /v1/embeddings called: items=%d upstream=%s model=%s",
            len(normalized_input),
            upstream_url,
            upstream_model,
        )
        try:
            upstream_response = requests.post(upstream_url, json=forward_payload, timeout=cfg.embedding.timeout_s)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception(
                "Embedding proxy call failed: upstream=%s elapsed_ms=%.2f error=%s",
                upstream_url,
                elapsed_ms,
                exc,
            )
            raise HTTPException(status_code=502, detail=f"Embedding upstream request failed: {exc}") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Embedding proxy call completed: upstream=%s status=%d elapsed_ms=%.2f",
            upstream_url,
            upstream_response.status_code,
            elapsed_ms,
        )

        content_type = upstream_response.headers.get("content-type", "")
        if "application/json" in content_type.lower():
            try:
                body = upstream_response.json()
            except ValueError as exc:
                raise HTTPException(status_code=502, detail=f"Embedding upstream returned invalid JSON: {exc}") from exc
            return JSONResponse(status_code=upstream_response.status_code, content=body)

        return Response(
            status_code=upstream_response.status_code,
            content=upstream_response.text,
            media_type=content_type or "text/plain",
        )

    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agentic RAG ops backend")
    parser.add_argument("--config", default="config.yml", help="Path to Agentic RAG config file")
    parser.add_argument("--host", help="Override host binding (defaults to chat.api.host in config)")
    parser.add_argument("--port", type=int, help="Override port binding (defaults to 8006)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = load_agentic_rag_config(args.config)

    level_name = (cfg.logging.level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info(
        "Loaded ops config: config=%s embedding_upstream=%s embedding_model=%s reports_dir=%s",
        args.config,
        cfg.embedding.base_url,
        cfg.embedding.model,
        cfg.paths.reports_dir,
    )

    app = create_app(cfg, config_path=args.config)
    host = args.host or cfg.chat.api.host
    port = args.port if args.port is not None else 8006
    logger.info("Starting ops API: host=%s port=%d config=%s", host, port, args.config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
