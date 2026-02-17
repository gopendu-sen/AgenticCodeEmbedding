from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from agentic_rag.agentic_ai.orchestrator import AgenticRagOrchestrator
from agentic_rag.code_parser.service import CodeParserService
from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.config_loader import load_agentic_rag_config
from agentic_rag.core.sqlite_store import SQLiteStore
from agentic_rag.core.utils import sha256_text
from agentic_rag.embedding.service import EmbeddingService


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yml"


class DummyEmbedder:
    def __init__(self):
        self._last_usage = {
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "total_tokens": 5,
            "has_usage": True,
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def last_call_usage(self) -> Dict[str, Any]:
        return dict(self._last_usage)


class DummyChroma:
    def __init__(self):
        self.records: Dict[str, List[Dict[str, Any]]] = {}

    def upsert(
        self,
        collection_name: str,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        rows = self.records.setdefault(collection_name, [])
        by_id = {str(item.get("id")): item for item in rows}
        for node_id, doc, meta in zip(ids, documents, metadatas):
            by_id[str(node_id)] = {"id": str(node_id), "document": doc, "metadata": dict(meta)}
        self.records[collection_name] = list(by_id.values())

    def count(self, collection_name: str) -> int:
        return len(self.records.get(collection_name, []))

    def delete_where(self, collection_name: str, where: Dict[str, Any]) -> int:
        rows = self.records.get(collection_name, [])
        keep: List[Dict[str, Any]] = []
        deleted = 0
        for row in rows:
            meta = row.get("metadata", {})
            if all(meta.get(key) == value for key, value in where.items()):
                deleted += 1
                continue
            keep.append(row)
        self.records[collection_name] = keep
        return deleted


def _make_node(node_id: str, node_type: str, text: str, file_path: str = "src/sample.py") -> CodeNode:
    return CodeNode(
        node_id=node_id,
        node_type=node_type,
        language="python",
        file_path=file_path,
        start_line=1,
        end_line=20,
        symbol="sample_fn",
        text=text,
        metadata={},
        confidence=0.9,
    ).finalize()


@pytest.fixture
def loaded_cfg():
    return load_agentic_rag_config(str(CONFIG_PATH))


def test_parser_routes_docs_and_config_extensions(loaded_cfg):
    parser = CodeParserService(loaded_cfg.parser, loaded_cfg.io_limits)

    html_nodes, _, _ = parser.parse_file("web/index.html", "<html>\n<body>x</body>\n</html>", ".html")
    jsp_nodes, _, _ = parser.parse_file("web/index.jsp", "<%@ page %>\n<div>x</div>", ".jsp")
    xml_nodes, _, _ = parser.parse_file("config/app.xml", "<a>\n<b>1</b>\n</a>", ".xml")
    props_nodes, _, _ = parser.parse_file("config/app.properties", "a=1\nb=2", ".properties")

    assert html_nodes and all(node.node_type == "doc" for node in html_nodes)
    assert jsp_nodes and all(node.node_type == "doc" for node in jsp_nodes)
    assert xml_nodes and all(node.node_type == "config" for node in xml_nodes)
    assert props_nodes and all(node.node_type == "config" for node in props_nodes)


def test_parser_unknown_and_error_fallbacks_are_chunked_docs(monkeypatch: pytest.MonkeyPatch, loaded_cfg):
    parser = CodeParserService(loaded_cfg.parser, loaded_cfg.io_limits)

    long_text = "\n".join(f"line {idx}" for idx in range(1, 400))
    unknown_nodes, _, _ = parser.parse_file("misc/file.unknown", long_text, ".unknown")
    assert len(unknown_nodes) >= 2
    assert all(node.node_type == "doc" for node in unknown_nodes)
    assert all(node.metadata.get("fallback_mode") == "unknown_extension_chunked_docs" for node in unknown_nodes)

    def boom(*_args, **_kwargs):
        raise RuntimeError("parser exploded")

    monkeypatch.setattr(parser.py_parser, "parse", boom)
    error_nodes, _, _ = parser.parse_file("src/broken.py", "def x():\n  pass\n", ".py")
    assert error_nodes
    assert all(node.node_type == "doc" for node in error_nodes)
    assert all(node.metadata.get("parse_error") == "exception" for node in error_nodes)
    assert all(node.metadata.get("fallback_mode") == "parser_error_chunked_docs" for node in error_nodes)


def test_embedding_routes_into_new_audit_dimensions_and_reports_store_summary(loaded_cfg):
    chroma = DummyChroma()
    service = EmbeddingService(
        embedder=DummyEmbedder(),
        chroma=chroma,
        collections=loaded_cfg.embedding.collections,
        repo_name="demo_repo",
        node_text_max_chars=loaded_cfg.io_limits.node_text_max_chars,
        embed_doc_max_chars=loaded_cfg.io_limits.embed_doc_max_chars,
        enable_audit_dimensions=True,
        verbose_per_node=False,
    )

    node_b = _make_node("n-b", "function", "updateCustomer saveProfile change phone email")
    node_p = _make_node("n-p", "function", "unblockAccount increaseLimit raiseDailyLimit")
    node_k = _make_node("n-k", "function", "transfer using payeeId recipient lookup")
    node_l = _make_node("n-l", "function", "createPayee updatePayee beneficiary")

    assert "audit_identity_profile" in service._choose_collections(node_b)
    assert "audit_limits_access" in service._choose_collections(node_p)
    assert "audit_money_movement" in service._choose_collections(node_k)
    assert "audit_payee_recipient" in service._choose_collections(node_k)
    assert "audit_payee_recipient" in service._choose_collections(node_l)

    upserted = service.upsert_nodes([node_b, node_p, node_k, node_l], batch_size=2)
    assert upserted > 0

    deleted = service.delete_vectors_for_file("src/sample.py")
    assert int(deleted.get("deleted_total", 0)) > 0

    telemetry = service.telemetry()
    summary = telemetry.get("store_summary", [])
    assert summary
    assert set(telemetry.get("store_types", {}).keys()) == {"base_stores", "audit_dimension_stores"}

    required_keys = {
        "store_key",
        "collection_name",
        "store_type",
        "nodes_queued",
        "nodes_upserted",
        "stale_vectors_deleted",
        "final_vector_count",
        "status",
        "error",
    }
    for item in summary:
        assert required_keys.issubset(item.keys())
        assert item["status"] in {"active", "empty", "failed"}


def test_parser_index_version_changes_file_hash(loaded_cfg):
    orchestrator_v1 = AgenticRagOrchestrator.__new__(AgenticRagOrchestrator)
    orchestrator_v1.cfg = loaded_cfg

    cfg_v2 = loaded_cfg.model_copy(
        update={
            "parser": loaded_cfg.parser.model_copy(update={"index_version": "v2_rules_recall_2026_02_18"}),
        }
    )
    orchestrator_v2 = AgenticRagOrchestrator.__new__(AgenticRagOrchestrator)
    orchestrator_v2.cfg = cfg_v2

    text = "def hello():\n    return 1\n"
    hash_v1 = orchestrator_v1._file_hash(text)
    hash_v2 = orchestrator_v2._file_hash(text)

    assert hash_v1 == sha256_text(f"{loaded_cfg.parser.index_version}\n{text}")
    assert hash_v2 == sha256_text(f"{cfg_v2.parser.index_version}\n{text}")
    assert hash_v1 != hash_v2


def test_sqlite_delete_nodes_for_file(tmp_path: Path):
    sqlite_path = tmp_path / "metadata.db"
    store = SQLiteStore(str(sqlite_path))
    try:
        store.upsert_file("src/a.py", "hash-a")
        store.upsert_node(
            node_id="node-1",
            file_path="src/a.py",
            node_type="function",
            language="python",
            start_line=1,
            end_line=5,
            symbol="x",
            content_hash="ch1",
            confidence=0.9,
            text="def x(): pass",
            metadata={},
        )
        store.upsert_node(
            node_id="node-2",
            file_path="src/a.py",
            node_type="function",
            language="python",
            start_line=7,
            end_line=9,
            symbol="y",
            content_hash="ch2",
            confidence=0.9,
            text="def y(): pass",
            metadata={},
        )

        deleted = store.delete_nodes_for_file("src/a.py")
        assert deleted == 2

        cur = store.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes WHERE file_path = ?", ("src/a.py",))
        assert int(cur.fetchone()[0]) == 0
    finally:
        store.close()
