from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient
import requests

from agentic_rag.core.config_loader import load_agentic_rag_config
from chat_module.api import create_app as create_chat_app
from ops_module.api import create_app as create_ops_app


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yml"


class DummyChatRetriever:
    def discover_store_names(self) -> List[str]:
        return ["demo_repo"]

    def chat_turn_stream(self, prompt: str, history: List[Dict[str, Any]], store_names: List[str], session_id: str = ""):
        yield {
            "event": "meta",
            "intent": "general",
            "repo_stores": store_names,
            "source_count": 0,
            "no_sources": True,
            "sources": [],
        }
        yield {"event": "token", "token": "No sources"}
        yield {
            "event": "done",
            "response": "No sources",
            "sources": [],
            "intent": "general",
            "cited_indices": [],
            "no_sources": True,
            "model": "",
        }


class DummyOpsRetriever:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.html_path = tmp_path / "report.html"
        self.json_path = tmp_path / "report.json"
        self.html_path.write_text("<html><body>ok</body></html>", encoding="utf-8")
        self.json_path.write_text('{"status":"ok"}', encoding="utf-8")

    def discover_store_names(self) -> List[str]:
        return ["demo_repo"]

    def start_embedding_job(self, repo_path: str, repo_name: str) -> Dict[str, Any]:
        return {
            "job_id": "emb_job_1",
            "status": "running",
            "repo_name": repo_name,
            "repo_path": repo_path,
            "config_path": str(CONFIG_PATH),
            "created_at_utc": "2026-01-01T00:00:00Z",
            "updated_at_utc": "2026-01-01T00:00:00Z",
        }

    def list_embedding_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [self.start_embedding_job("/tmp/repo", "demo_repo")]

    def get_embedding_job(self, job_id: str) -> Dict[str, Any]:
        return self.start_embedding_job("/tmp/repo", "demo_repo")

    def get_evaluation_rules(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "updated_at_utc": "2026-01-01T00:00:00Z",
            "items": [
                {
                    "id": "rule_1",
                    "title": "Rule",
                    "definition": "Def",
                    "strong_signals": ["a"],
                    "weak_signals": ["b"],
                    "false_positives": ["c"],
                }
            ],
        }

    def save_evaluation_rules(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        saved = dict(payload)
        saved["updated_at_utc"] = "2026-01-01T00:00:01Z"
        return saved

    def start_evaluation_job(self, repo_name: str, repo_path: str) -> Dict[str, Any]:
        return {
            "job_id": "eval_job_1",
            "status": "running",
            "repo_name": repo_name,
            "repo_path": repo_path,
            "config_path": str(CONFIG_PATH),
            "created_at_utc": "2026-01-01T00:00:00Z",
            "updated_at_utc": "2026-01-01T00:00:00Z",
            "report_html_path": str(self.html_path),
            "report_json_path": str(self.json_path),
            "summary": {
                "report_html_path": str(self.html_path),
                "report_json_path": str(self.json_path),
            },
        }

    def list_evaluation_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        return [self.start_evaluation_job("demo_repo", "/tmp/repo")]

    def get_evaluation_job(self, job_id: str) -> Dict[str, Any]:
        return self.start_evaluation_job("demo_repo", "/tmp/repo")


@pytest.fixture
def base_cfg(tmp_path: Path):
    cfg = load_agentic_rag_config(str(CONFIG_PATH))
    return cfg.model_copy(
        update={
            "chat": cfg.chat.model_copy(
                update={
                    "memory": cfg.chat.memory.model_copy(
                        update={
                            "sqlite_path": str(tmp_path / "chat_memory_test.db"),
                            "enable_summarisation": False,
                            "enable_intent_tracking": False,
                        }
                    )
                }
            )
        }
    )


def test_chat_service_routes_exclude_embedding_and_evaluation(monkeypatch: pytest.MonkeyPatch, base_cfg):
    monkeypatch.setattr(
        "chat_module.api.StoreRetriever.from_config",
        classmethod(lambda cls, cfg, config_path="config.yml": DummyChatRetriever()),
    )
    app = create_chat_app(base_cfg, config_path=str(CONFIG_PATH))
    client = TestClient(app)

    assert client.post("/embedding/jobs", json={"repo_path": "/tmp", "repo_name": "x"}).status_code == 404
    assert client.get("/embedding/jobs").status_code == 404
    assert client.get("/evaluation/rules").status_code == 404
    assert client.post("/evaluation/jobs", json={"repo_path": "/tmp", "repo_name": "x"}).status_code == 404


def test_ops_service_exposes_embedding_and_evaluation_routes(monkeypatch: pytest.MonkeyPatch, base_cfg, tmp_path: Path):
    dummy = DummyOpsRetriever(tmp_path)
    monkeypatch.setattr(
        "ops_module.api.StoreRetriever.from_config",
        classmethod(lambda cls, cfg, config_path="config.yml": dummy),
    )
    app = create_ops_app(base_cfg, config_path=str(CONFIG_PATH))
    client = TestClient(app)

    assert client.get("/stores").status_code == 200
    assert client.post("/embedding/jobs", json={"repo_path": "/tmp/repo", "repo_name": "demo"}).status_code == 200
    assert client.get("/embedding/jobs").status_code == 200
    assert client.get("/embedding/jobs/emb_job_1").status_code == 200
    assert client.get("/evaluation/rules").status_code == 200
    assert client.put("/evaluation/rules", json=dummy.get_evaluation_rules()).status_code == 200
    assert client.post("/evaluation/jobs", json={"repo_path": "/tmp/repo", "repo_name": "demo"}).status_code == 200
    assert client.get("/evaluation/jobs").status_code == 200
    assert client.get("/evaluation/jobs/eval_job_1").status_code == 200
    assert client.get("/evaluation/jobs/eval_job_1/html").status_code == 200
    assert client.get("/evaluation/jobs/eval_job_1/json").status_code == 200


class _DummyHTTPResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]):
        self.status_code = status_code
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    def json(self) -> Dict[str, Any]:
        return self._payload



def test_ops_embeddings_proxy_passthrough(monkeypatch: pytest.MonkeyPatch, base_cfg, tmp_path: Path):
    dummy = DummyOpsRetriever(tmp_path)
    monkeypatch.setattr(
        "ops_module.api.StoreRetriever.from_config",
        classmethod(lambda cls, cfg, config_path="config.yml": dummy),
    )

    seen: Dict[str, Any] = {}

    def fake_post(url: str, json: Dict[str, Any], timeout: int):
        seen["url"] = url
        seen["payload"] = json
        seen["timeout"] = timeout
        return _DummyHTTPResponse(
            202,
            {
                "object": "list",
                "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    monkeypatch.setattr("ops_module.api.requests.post", fake_post)

    app = create_ops_app(base_cfg, config_path=str(CONFIG_PATH))
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"input": "hello world"})

    assert response.status_code == 202
    assert seen["url"].endswith("/embeddings")
    assert seen["payload"]["input"] == ["hello world"]
    assert seen["payload"]["model"] == base_cfg.embedding.model
    assert "data" in response.json()



def test_chat_retrieval_fails_fast_when_ops_embedding_unavailable(monkeypatch: pytest.MonkeyPatch, base_cfg, tmp_path: Path):
    cfg = base_cfg.model_copy(
        update={
            "embedding": base_cfg.embedding.model_copy(update={"base_url": "http://127.0.0.1:8006/v1"}),
            "chat": base_cfg.chat.model_copy(
                update={
                    "memory": base_cfg.chat.memory.model_copy(
                        update={
                            "sqlite_path": str(tmp_path / "chat_memory_unavailable.db"),
                            "enable_summarisation": False,
                            "enable_intent_tracking": False,
                        }
                    )
                }
            ),
        }
    )

    seen: Dict[str, Any] = {}

    def failing_post(url: str, json: Dict[str, Any], timeout: int):
        seen["url"] = url
        raise requests.exceptions.ConnectionError("ops unavailable")

    monkeypatch.setattr("agentic_rag.embedding.client.requests.post", failing_post)

    app = create_chat_app(cfg, config_path=str(CONFIG_PATH))
    client = TestClient(app)
    response = client.post(
        "/chat",
        json={
            "session_id": "sess_test_unavailable",
            "message": "where is auth middleware",
            "store_names": ["demo_repo"],
        },
    )

    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:8006/v1/embeddings"
    assert "event: error" in response.text
    assert "Chat stream failed" in response.text
