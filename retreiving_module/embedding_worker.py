import argparse
import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

from agentic_rag.agentic_ai.orchestrator import AgenticRagOrchestrator
from agentic_rag.core.config_loader import load_agentic_rag_config


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_status(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        return payload
    return {}


def _write_status(path: str, payload: Dict[str, Any]) -> None:
    payload["updated_at_utc"] = _utc_now()
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Background embedding worker")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--repo-path", required=True, help="Repository path to ingest")
    parser.add_argument("--repo-name", required=True, help="Repo tag for metadata")
    parser.add_argument("--job-id", required=True, help="Embedding job id")
    parser.add_argument("--status-path", required=True, help="Path to job status json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info(
        "Embedding worker started: job_id=%s repo_name=%s repo_path=%s config=%s",
        args.job_id,
        args.repo_name,
        args.repo_path,
        args.config,
    )

    status = _read_status(args.status_path)
    status.update({
        "job_id": args.job_id,
        "status": "running",
        "repo_name": args.repo_name,
        "repo_path": os.path.abspath(args.repo_path),
        "config_path": os.path.abspath(args.config),
        "started_at_utc": status.get("started_at_utc") or _utc_now(),
        "error": "",
    })
    _write_status(args.status_path, status)

    try:
        cfg = load_agentic_rag_config(args.config)
        cfg = cfg.model_copy(
            update={
                "paths": cfg.paths.model_copy(update={"repo_path": os.path.abspath(args.repo_path)}),
            }
        )
        orchestrator = AgenticRagOrchestrator(cfg, repo_name=args.repo_name)
        summary = orchestrator.run()
        status.update({
            "status": "completed",
            "finished_at_utc": _utc_now(),
            "summary": summary,
            "report_path": str(summary.get("report_path", "")),
        })
        _write_status(args.status_path, status)
        logger.info(
            "Embedding worker completed: job_id=%s files_changed=%s nodes_embedded=%s",
            args.job_id,
            summary.get("files_changed_indexed"),
            summary.get("nodes_embedded_upserted"),
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        status.update({
            "status": "failed",
            "finished_at_utc": _utc_now(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _write_status(args.status_path, status)
        logger.exception("Embedding worker failed: job_id=%s error=%s", args.job_id, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
