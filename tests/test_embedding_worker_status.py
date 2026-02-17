from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from agentic_rag.core.config_loader import load_agentic_rag_config
from retreiving_module import embedding_worker


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yml"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _argv(config_path: Path, status_path: Path) -> list[str]:
    return [
        "embedding_worker.py",
        "--config",
        str(config_path),
        "--repo-path",
        str(ROOT),
        "--repo-name",
        "demo_repo",
        "--job-id",
        "job_123",
        "--status-path",
        str(status_path),
    ]


def test_embedding_worker_finalizes_failed_status_when_config_load_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    status_path = tmp_path / "job_status.json"

    def fail_load(_path: str):
        raise ValueError("bad config")

    monkeypatch.setattr("retreiving_module.embedding_worker.load_agentic_rag_config", fail_load)
    monkeypatch.setattr(sys, "argv", _argv(CONFIG_PATH, status_path))

    rc = embedding_worker.main()
    payload = _read_json(status_path)

    assert rc == 1
    assert payload["status"] == "failed"
    assert payload["stage"] == "failed"
    assert "bad config" in payload["error"]
    assert payload["started_at_utc"]
    assert payload["finished_at_utc"]
    assert payload["updated_at_utc"]
    assert payload["summary"] is None


def test_embedding_worker_finalizes_failed_status_when_orchestrator_run_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    status_path = tmp_path / "job_status.json"
    cfg = load_agentic_rag_config(str(CONFIG_PATH))

    class FailingOrchestrator:
        def __init__(self, cfg, repo_name: str):
            self.cfg = cfg
            self.repo_name = repo_name

        def run(self):
            raise RuntimeError("run exploded")

    monkeypatch.setattr("retreiving_module.embedding_worker.load_agentic_rag_config", lambda _path: cfg)
    monkeypatch.setattr("retreiving_module.embedding_worker.AgenticRagOrchestrator", FailingOrchestrator)
    monkeypatch.setattr(sys, "argv", _argv(CONFIG_PATH, status_path))

    rc = embedding_worker.main()
    payload = _read_json(status_path)

    assert rc == 1
    assert payload["status"] == "failed"
    assert payload["stage"] == "failed"
    assert "run exploded" in payload["error"]
    assert payload["finished_at_utc"]
    assert payload["updated_at_utc"]


def test_embedding_worker_writes_completed_partial_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    status_path = tmp_path / "job_status.json"
    cfg = load_agentic_rag_config(str(CONFIG_PATH))

    class PartialOrchestrator:
        def __init__(self, cfg, repo_name: str):
            self.cfg = cfg
            self.repo_name = repo_name

        def run(self):
            return {
                "embedding_run_status": "completed_partial",
                "files_changed_indexed": 2,
                "nodes_embedded_upserted": 11,
                "report_path": str(tmp_path / "report.json"),
                "file_status_counts": {"success": 1, "failed": 1},
                "files": [{"file_path": "a.py", "status": "failed", "partial": True}],
            }

    monkeypatch.setattr("retreiving_module.embedding_worker.load_agentic_rag_config", lambda _path: cfg)
    monkeypatch.setattr("retreiving_module.embedding_worker.AgenticRagOrchestrator", PartialOrchestrator)
    monkeypatch.setattr(sys, "argv", _argv(CONFIG_PATH, status_path))

    rc = embedding_worker.main()
    payload = _read_json(status_path)

    assert rc == 0
    assert payload["status"] == "completed_partial"
    assert payload["stage"] == "completed"
    assert payload["partial"] is True
    assert payload["summary"]["embedding_run_status"] == "completed_partial"
    assert payload["finished_at_utc"]
    assert payload["updated_at_utc"]
