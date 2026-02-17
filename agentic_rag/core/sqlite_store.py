import sqlite3
import json
from typing import Any, Dict, Optional


class SQLiteStore:
    def __init__(self, sqlite_path: str):
        self.sqlite_path = sqlite_path
        self.conn = sqlite3.connect(sqlite_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
          file_path TEXT PRIMARY KEY,
          file_hash TEXT NOT NULL,
          last_indexed_at TEXT DEFAULT (datetime('now'))
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
          node_id TEXT PRIMARY KEY,
          file_path TEXT NOT NULL,
          node_type TEXT NOT NULL,
          language TEXT NOT NULL,
          start_line INTEGER NOT NULL,
          end_line INTEGER NOT NULL,
          symbol TEXT,
          content_hash TEXT NOT NULL,
          confidence REAL NOT NULL,
          text TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          indexed_at TEXT DEFAULT (datetime('now'))
        );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);")
        self.conn.commit()

    def get_file_hash(self, file_path: str) -> Optional[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT file_hash FROM files WHERE file_path = ?", (file_path,))
        row = cur.fetchone()
        return row[0] if row else None

    def upsert_file(self, file_path: str, file_hash: str) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO files(file_path, file_hash) VALUES(?, ?)
        ON CONFLICT(file_path) DO UPDATE SET file_hash=excluded.file_hash, last_indexed_at=datetime('now');
        """, (file_path, file_hash))
        self.conn.commit()

    def upsert_node(
        self,
        node_id: str,
        file_path: str,
        node_type: str,
        language: str,
        start_line: int,
        end_line: int,
        symbol: Optional[str],
        content_hash: str,
        confidence: float,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO nodes(node_id, file_path, node_type, language, start_line, end_line, symbol,
                          content_hash, confidence, text, metadata_json)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
          content_hash=excluded.content_hash,
          confidence=excluded.confidence,
          text=excluded.text,
          metadata_json=excluded.metadata_json,
          indexed_at=datetime('now');
        """, (
            node_id, file_path, node_type, language, start_line, end_line, symbol,
            content_hash, confidence, text, json.dumps(metadata, ensure_ascii=False)
        ))
        self.conn.commit()

    def delete_nodes_for_file(self, file_path: str) -> int:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM nodes WHERE file_path = ?", (file_path,))
        deleted = int(cur.rowcount if cur.rowcount is not None else 0)
        self.conn.commit()
        return max(0, deleted)

    def close(self):
        self.conn.close()
