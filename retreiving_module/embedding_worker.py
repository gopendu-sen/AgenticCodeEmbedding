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

_TERMINAL_STATUSES = {"completed", "completed_partial", "failed"}


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


def _base_status(args: argparse.Namespace, current: Dict[str, Any]) -> Dict[str, Any]:
    now = _utc_now()
    status = dict(current)
    status.update(
        {
            "job_id": args.job_id,
            "repo_name": args.repo_name,
            "repo_path": os.path.abspath(args.repo_path),
            "config_path": os.path.abspath(args.config),
        }
    )
    status["status"] = str(status.get("status", "starting") or "starting")
    status["stage"] = str(status.get("stage", "init") or "init")
    status["started_at_utc"] = str(status.get("started_at_utc", "") or now)
    status["finished_at_utc"] = str(status.get("finished_at_utc", "") or "")
    status["error"] = str(status.get("error", "") or "")
    status["partial"] = bool(status.get("partial", False))
    status["summary"] = status.get("summary") if "summary" in status else None
    status["report_path"] = str(status.get("report_path", "") or "")
    return status


def _set_stage(
    args: argparse.Namespace,
    status: Dict[str, Any],
    *,
    stage: str,
    state: str,
    error: str = "",
    partial: bool = False,
    summary: Any = None,
    report_path: str = "",
    file_status_counts: Any = None,
    files: Any = None,
    trace_text: str = "",
    finished: bool = False,
) -> Dict[str, Any]:
    next_status = dict(status)
    next_status["status"] = state
    next_status["stage"] = stage
    next_status["error"] = error
    next_status["partial"] = bool(partial)
    next_status["summary"] = summary
    next_status["report_path"] = report_path
    if file_status_counts is not None:
        next_status["file_status_counts"] = file_status_counts
    if files is not None:
        next_status["files"] = files
    if trace_text:
        next_status["traceback"] = trace_text
    elif "traceback" in next_status and state != "failed":
        next_status.pop("traceback", None)

    if finished:
        next_status["finished_at_utc"] = _utc_now()
    elif state in _TERMINAL_STATUSES and not str(next_status.get("finished_at_utc", "")).strip():
        next_status["finished_at_utc"] = _utc_now()

    _write_status(args.status_path, next_status)
    return next_status


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

    status = _base_status(args, _read_status(args.status_path))
    status = _set_stage(
        args,
        status,
        stage="init",
        state="starting",
        error="",
        partial=False,
        summary=None,
        report_path="",
    )

    return_code = 1
    try:
        cfg = load_agentic_rag_config(args.config)
        status = _set_stage(
            args,
            status,
            stage="config_loaded",
            state="running",
            error="",
            partial=False,
            summary=None,
            report_path="",
        )

        cfg = cfg.model_copy(
            update={
                "paths": cfg.paths.model_copy(update={"repo_path": os.path.abspath(args.repo_path)}),
            }
        )
        orchestrator = AgenticRagOrchestrator(cfg, repo_name=args.repo_name)
        status = _set_stage(
            args,
            status,
            stage="orchestrator_running",
            state="running",
            error="",
            partial=False,
            summary=None,
            report_path="",
        )

        summary = orchestrator.run()
        run_status = str(summary.get("embedding_run_status", "completed")).strip().lower()
        job_status = "completed_partial" if run_status == "completed_partial" else "completed"
        file_status_counts = summary.get("file_status_counts", {})
        files = summary.get("files", [])
        status = _set_stage(
            args,
            status,
            stage="completed",
            state=job_status,
            error="",
            partial=job_status == "completed_partial",
            summary=summary,
            report_path=str(summary.get("report_path", "")),
            file_status_counts=file_status_counts,
            files=files,
            finished=True,
        )
        logger.info(
            "Embedding worker completed: job_id=%s status=%s files_changed=%s nodes_embedded=%s",
            args.job_id,
            job_status,
            summary.get("files_changed_indexed"),
            summary.get("nodes_embedded_upserted"),
        )
        return_code = 0
    except Exception as exc:  # noqa: BLE001
        err_text = str(exc)
        trace_text = traceback.format_exc()
        summary = status.get("summary")
        status = _set_stage(
            args,
            status,
            stage="failed",
            state="failed",
            error=err_text,
            partial=bool(status.get("partial", False) or summary),
            summary=summary,
            report_path=str(status.get("report_path", "")),
            file_status_counts=status.get("file_status_counts"),
            files=status.get("files"),
            trace_text=trace_text,
            finished=True,
        )
        logger.exception("Embedding worker failed: job_id=%s error=%s", args.job_id, exc)
    finally:
        terminalized = status.get("status") in _TERMINAL_STATUSES
        if not terminalized:
            status = _set_stage(
                args,
                status,
                stage="failed",
                state="failed",
                error=str(status.get("error", "") or "Embedding worker exited without terminal status"),
                partial=bool(status.get("partial", False)),
                summary=status.get("summary"),
                report_path=str(status.get("report_path", "")),
                file_status_counts=status.get("file_status_counts"),
                files=status.get("files"),
                trace_text=str(status.get("traceback", "")),
                finished=True,
            )
        elif not str(status.get("finished_at_utc", "")).strip():
            status = _set_stage(
                args,
                status,
                stage=str(status.get("stage", "completed") or "completed"),
                state=str(status.get("status", "failed") or "failed"),
                error=str(status.get("error", "") or ""),
                partial=bool(status.get("partial", False)),
                summary=status.get("summary"),
                report_path=str(status.get("report_path", "")),
                file_status_counts=status.get("file_status_counts"),
                files=status.get("files"),
                trace_text=str(status.get("traceback", "")),
                finished=True,
            )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
