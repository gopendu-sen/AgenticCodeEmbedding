"""FastAPI backend for chat streaming + session management + store discovery."""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn

from agentic_rag.core.config import AgenticRagConfig
from agentic_rag.core.config_loader import load_agentic_rag_config
from agentic_rag.core.llm_client import LLMClient
from retreiving_module import StoreRetriever

from .session_service import SessionService
from .session_store import SessionStore


logger = logging.getLogger(__name__)


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique chat session identifier.")
    message: str = Field(..., description="User message to send to the model.")
    store_names: List[str] = Field(..., description="One or more repo store names.")
    enable_summarisation: Optional[bool] = Field(None, description="Optional override for summary updates.")
    enable_intent_tracking: Optional[bool] = Field(None, description="Optional override for intent tagging.")

    @field_validator("session_id", "message")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must not be empty")
        return cleaned

    @field_validator("store_names")
    @classmethod
    def _valid_store_names(cls, values: List[str]) -> List[str]:
        cleaned = [item.strip() for item in values if item.strip()]
        if not cleaned:
            raise ValueError("store_names must contain at least one non-empty value")
        return cleaned


def _build_session_service(cfg: AgenticRagConfig) -> SessionService:
    memory_llm = LLMClient(
        base_url=(cfg.llm.base_url or "").strip(),
        model=(cfg.llm.model or "").strip(),
        timeout_s=cfg.llm.timeout_s,
    )
    store = SessionStore(cfg.chat.memory.sqlite_path)
    return SessionService(
        store=store,
        llm=memory_llm,
        max_history_messages=cfg.chat.memory.max_history_messages,
        enable_summarisation=cfg.chat.memory.enable_summarisation,
        enable_intent_tracking=cfg.chat.memory.enable_intent_tracking,
        summarise_prompt=cfg.chat.memory.summarise_prompt,
        intent_prompt=cfg.chat.memory.intent_prompt,
    )


def create_app(cfg: AgenticRagConfig, *, config_path: str) -> FastAPI:
    retriever = StoreRetriever.from_config(cfg, config_path=config_path)
    sessions = _build_session_service(cfg)

    app = FastAPI(title="Vyom Chat API", version="1.0.0")
    app.state.cfg = cfg
    app.state.retriever = retriever
    app.state.sessions = sessions

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

    @app.get("/ui-config")
    async def ui_config() -> Dict[str, Any]:
        chat_cfg = cfg.chat
        return {
            "title": chat_cfg.title,
            "subtitle": chat_cfg.subtitle,
            "assistant_greeting": chat_cfg.assistant_greeting,
            "input_placeholder": chat_cfg.input_placeholder,
            "spinner_text": chat_cfg.spinner_text,
            "show_sources": chat_cfg.show_sources,
            "max_context_chunks": chat_cfg.max_context_chunks,
            "history_messages": chat_cfg.memory.max_history_messages,
            "chat_api_host": chat_cfg.api.host,
            "chat_api_port": chat_cfg.api.port,
            "embedding_api_base_url": cfg.embedding.base_url,
            "ui_host": chat_cfg.ui.host,
            "ui_port": chat_cfg.ui.port,
        }

    @app.get("/stores")
    async def stores() -> Dict[str, Any]:
        try:
            discovered = retriever.discover_store_names()
            logger.info("REST /stores completed: discovered=%d", len(discovered))
            return {"stores": discovered}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Store discovery failed")
            raise HTTPException(status_code=500, detail=f"Store discovery failed: {exc}") from exc

    @app.get("/sessions")
    async def list_sessions(limit: int = Query(default=100, ge=1, le=500)) -> Dict[str, Any]:
        logger.info("REST /sessions called: limit=%d", limit)
        payload = sessions.list_sessions(limit=limit)
        logger.info("REST /sessions completed: returned=%d", len(payload))
        return {"sessions": payload}

    @app.get("/history/{session_id}")
    async def history(session_id: str) -> Dict[str, Any]:
        logger.info("REST /history/{session_id} called: session_id=%s", session_id)
        try:
            payload = sessions.get_history(session_id)
            logger.info(
                "REST /history/{session_id} completed: session_id=%s messages=%d",
                session_id,
                len(payload.get("messages", [])),
            )
            return payload
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str) -> Dict[str, Any]:
        logger.info("REST /sessions/{session_id} delete called: session_id=%s", session_id)
        deleted = sessions.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"No chat session found for id '{session_id}'")
        return {"deleted": True, "session_id": session_id}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        session_id = request.session_id
        message = request.message
        store_names = request.store_names
        logger.info(
            "REST /chat called: session_id=%s stores=%d message_chars=%d summarisation=%s intent_tracking=%s",
            session_id,
            len(store_names),
            len(message),
            request.enable_summarisation,
            request.enable_intent_tracking,
        )

        sessions.ensure_session(session_id)
        sessions.append_user_message(session_id, message)
        history_for_prompt = sessions.get_prompt_history(session_id)

        def event_stream() -> Iterator[str]:
            done_payload: Optional[Dict[str, Any]] = None
            assistant_response_parts: List[str] = []
            stream_event_count = 0
            stream_token_chars = 0
            try:
                for event in retriever.chat_turn_stream(
                    prompt=message,
                    history=history_for_prompt,
                    store_names=store_names,
                    session_id=session_id,
                ):
                    event_name = str(event.get("event", "token"))
                    stream_event_count += 1
                    if event_name == "token":
                        token_text = str(event.get("token", ""))
                        stream_token_chars += len(token_text)
                        assistant_response_parts.append(token_text)
                    elif event_name == "done":
                        done_payload = event
                    yield _sse_event(event_name, event)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Chat stream failed: session_id=%s", session_id)
                yield _sse_event("error", {"message": f"Chat stream failed: {exc}"})
                return

            if done_payload is None:
                done_payload = {
                    "event": "done",
                    "response": "".join(assistant_response_parts),
                    "sources": [],
                    "intent": "",
                    "cited_indices": [],
                    "no_sources": True,
                }
                yield _sse_event("done", done_payload)
            logger.info(
                "REST /chat stream completed: session_id=%s events=%d token_chars=%d intent=%s no_sources=%s",
                session_id,
                stream_event_count,
                stream_token_chars,
                str(done_payload.get("intent", "")),
                bool(done_payload.get("no_sources", False)),
            )

            try:
                sessions.append_assistant_message(
                    session_id,
                    response=str(done_payload.get("response", "")),
                    sources=list(done_payload.get("sources", [])),
                    cited_indices=list(done_payload.get("cited_indices", [])),
                    intent=str(done_payload.get("intent", "")),
                    latest_user_message=message,
                    enable_summarisation=request.enable_summarisation,
                    enable_intent_tracking=request.enable_intent_tracking,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Failed to persist assistant turn: session_id=%s", session_id)

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)

    return app


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Vyom chat backend")
    parser.add_argument("--config", default="config.yml", help="Path to Vyom config file")
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
        "Loaded chat config: config=%s chat_api=%s:%d llm_base_url=%s llm_model=%s memory_sqlite=%s",
        args.config,
        cfg.chat.api.host,
        cfg.chat.api.port,
        cfg.llm.base_url,
        cfg.llm.model,
        cfg.chat.memory.sqlite_path,
    )

    app = create_app(cfg, config_path=args.config)
    host = cfg.chat.api.host
    port = cfg.chat.api.port
    logger.info("Starting chat API: host=%s port=%d config=%s", host, port, args.config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
