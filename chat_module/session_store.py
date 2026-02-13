"""SQLite persistence for chat sessions and messages."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional


class SessionStore:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = os.path.abspath(sqlite_path)
        os.makedirs(os.path.dirname(self.sqlite_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL DEFAULT '',
                    intents_json TEXT NOT NULL DEFAULT '[]',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    cited_indices_json TEXT NOT NULL DEFAULT '[]',
                    intent TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at)")

    @staticmethod
    def _to_json(value: Any, default: str) -> str:
        try:
            return json.dumps(value if value is not None else json.loads(default), ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return default

    @staticmethod
    def _from_json(raw: Any, default: Any) -> Any:
        if not isinstance(raw, str) or not raw.strip():
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return default

    def ensure_session(self, session_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions(session_id, summary, intents_json, created_at, updated_at)
                VALUES(?, '', '[]', ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (session_id, now, now),
            )

    def touch_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chat_sessions SET updated_at=? WHERE session_id=?", (time.time(), session_id))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, summary, intents_json, created_at, updated_at FROM chat_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row["session_id"]),
            "summary": str(row["summary"] or ""),
            "intents": self._from_json(row["intents_json"], []),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def update_summary_and_intents(self, session_id: str, summary: str, intents: List[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET summary=?, intents_json=?, updated_at=? WHERE session_id=?",
                (summary, self._to_json(intents, "[]"), time.time(), session_id),
            )

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        sources: Optional[List[Dict[str, Any]]] = None,
        cited_indices: Optional[List[int]] = None,
        intent: str = "",
    ) -> int:
        self.ensure_session(session_id)
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_messages(session_id, role, content, sources_json, cited_indices_json, intent, created_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content,
                    self._to_json(sources or [], "[]"),
                    self._to_json(cited_indices or [], "[]"),
                    intent,
                    now,
                ),
            )
            conn.execute("UPDATE chat_sessions SET updated_at=? WHERE session_id=?", (now, session_id))
            return int(cursor.lastrowid)

    def prune_messages(self, session_id: str, keep_last: int) -> None:
        keep_last = max(1, int(keep_last))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(1) AS c FROM chat_messages WHERE session_id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                return
            count = int(row["c"])
            if count <= keep_last:
                return
            to_delete = count - keep_last
            conn.execute(
                """
                DELETE FROM chat_messages
                WHERE id IN (
                    SELECT id
                    FROM chat_messages
                    WHERE session_id=?
                    ORDER BY id ASC
                    LIMIT ?
                )
                """,
                (session_id, to_delete),
            )

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT id, role, content, sources_json, cited_indices_json, intent, created_at
            FROM chat_messages
            WHERE session_id=?
            ORDER BY id ASC
        """
        params: List[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "role": str(row["role"]),
                    "content": str(row["content"]),
                    "sources": self._from_json(row["sources_json"], []),
                    "cited_indices": self._from_json(row["cited_indices_json"], []),
                    "intent": str(row["intent"] or ""),
                    "created_at": float(row["created_at"]),
                }
            )
        return out

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
            cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (session_id,))
            return cursor.rowcount > 0

    def list_sessions(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.summary,
                    s.intents_json,
                    s.updated_at,
                    (SELECT content FROM chat_messages m WHERE m.session_id=s.session_id ORDER BY m.id DESC LIMIT 1) AS last_message,
                    (SELECT COUNT(1) FROM chat_messages m2 WHERE m2.session_id=s.session_id) AS message_count
                FROM chat_sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "session_id": str(row["session_id"]),
                    "summary": str(row["summary"] or ""),
                    "intents": self._from_json(row["intents_json"], []),
                    "updated_at": float(row["updated_at"]),
                    "last_message": str(row["last_message"] or ""),
                    "message_count": int(row["message_count"] or 0),
                }
            )
        return out
