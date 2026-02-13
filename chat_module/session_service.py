"""Session orchestration on top of SQLite persistence."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from agentic_rag.core.llm_client import LLMClient

from .session_store import SessionStore


logger = logging.getLogger(__name__)


class SessionService:
    def __init__(
        self,
        store: SessionStore,
        llm: LLMClient,
        *,
        max_history_messages: int,
        enable_summarisation: bool,
        enable_intent_tracking: bool,
        summarise_prompt: str,
        intent_prompt: str,
    ):
        self.store = store
        self.llm = llm
        self.max_history_messages = max(1, int(max_history_messages))
        self.enable_summarisation_default = bool(enable_summarisation)
        self.enable_intent_tracking_default = bool(enable_intent_tracking)
        self.summarise_prompt = summarise_prompt
        self.intent_prompt = intent_prompt

    def ensure_session(self, session_id: str) -> None:
        self.store.ensure_session(session_id)

    def get_prompt_history(self, session_id: str) -> List[Dict[str, str]]:
        messages = self.store.get_messages(session_id)
        return [{"role": m["role"], "content": m["content"]} for m in messages[-self.max_history_messages :]]

    def append_user_message(self, session_id: str, message: str) -> None:
        self.store.add_message(session_id, role="user", content=message)
        self.store.prune_messages(session_id, keep_last=self.max_history_messages)

    def append_assistant_message(
        self,
        session_id: str,
        response: str,
        *,
        sources: List[Dict[str, Any]],
        cited_indices: List[int],
        intent: str,
        latest_user_message: str,
        enable_summarisation: Optional[bool],
        enable_intent_tracking: Optional[bool],
    ) -> None:
        self.store.add_message(
            session_id,
            role="assistant",
            content=response,
            sources=sources,
            cited_indices=cited_indices,
            intent=intent,
        )
        self.store.prune_messages(session_id, keep_last=self.max_history_messages)
        self._post_turn_updates(
            session_id=session_id,
            latest_user_message=latest_user_message,
            enable_summarisation=enable_summarisation,
            enable_intent_tracking=enable_intent_tracking,
        )

    def _post_turn_updates(
        self,
        *,
        session_id: str,
        latest_user_message: str,
        enable_summarisation: Optional[bool],
        enable_intent_tracking: Optional[bool],
    ) -> None:
        do_summary = self.enable_summarisation_default if enable_summarisation is None else bool(enable_summarisation)
        do_intent = self.enable_intent_tracking_default if enable_intent_tracking is None else bool(enable_intent_tracking)

        session = self.store.get_session(session_id)
        if session is None:
            return
        summary = str(session.get("summary", "") or "")
        intents = list(session.get("intents", []) or [])

        if do_summary:
            try:
                summary = self._generate_summary(session_id)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to refresh summary: session_id=%s", session_id)

        if do_intent:
            try:
                intent = self._detect_intent(latest_user_message)
                if intent:
                    intents.append(intent)
                    intents = intents[-50:]
            except Exception:  # noqa: BLE001
                logger.exception("Failed to detect intent: session_id=%s", session_id)

        self.store.update_summary_and_intents(session_id, summary=summary, intents=intents)

    def _generate_summary(self, session_id: str) -> str:
        messages = self.store.get_messages(session_id)
        lines = [f"{m['role']}: {m['content']}" for m in messages[-self.max_history_messages :]]
        conversation = "\n".join(lines)
        prompt_messages = [
            {"role": "system", "content": self.summarise_prompt},
            {"role": "user", "content": conversation},
        ]
        return self.llm.chat(prompt_messages, temperature=0).strip()

    def _detect_intent(self, latest_user_message: str) -> str:
        prompt_messages = [
            {"role": "system", "content": self.intent_prompt},
            {"role": "user", "content": latest_user_message},
        ]
        return self.llm.chat(prompt_messages, temperature=0).strip()

    def get_history(self, session_id: str) -> Dict[str, Any]:
        session = self.store.get_session(session_id)
        if session is None:
            raise ValueError(f"No chat session found for id '{session_id}'")
        return {
            "session_id": session["session_id"],
            "summary": session.get("summary", ""),
            "intents": session.get("intents", []),
            "messages": self.store.get_messages(session_id),
            "updated_at": session.get("updated_at"),
        }

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.store.list_sessions(limit=limit)

    def delete_session(self, session_id: str) -> bool:
        return self.store.delete_session(session_id)
