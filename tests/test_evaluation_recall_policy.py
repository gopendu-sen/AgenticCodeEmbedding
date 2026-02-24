from __future__ import annotations

import os
from types import MethodType
from types import SimpleNamespace
from typing import Any, Dict, List

from retreiving_module.service import StoreRetriever


class _DummyEmbedder:
    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.1, 0.2] for _ in texts]


class _DummyChroma:
    def __init__(self, rows: int = 6):
        self._rows: List[Dict[str, Any]] = []
        for idx in range(rows):
            self._rows.append(
                {
                    "id": f"id-{idx}",
                    "document": f"document-{idx}",
                    "metadata": {
                        "repo_name": "demo",
                        "file_path": f"src/file_{idx}.py",
                        "start_line": idx + 1,
                        "end_line": idx + 1,
                        "node_type": "function",
                    },
                    "distance": 0.01 * idx,
                }
            )

    def count(self, collection_name: str) -> int:
        return len(self._rows)

    def query(
        self,
        *,
        collection_name: str,
        query_embeddings: List[List[float]],
        n_results: int,
        where: Dict[str, Any] | None = None,
    ) -> Dict[str, List[List[Any]]]:
        filtered = list(self._rows)
        if where and "repo_name" in where:
            filtered = [row for row in filtered if row["metadata"].get("repo_name") == where["repo_name"]]
        sliced = filtered[: max(1, n_results)]
        return {
            "ids": [[row["id"] for row in sliced]],
            "documents": [[row["document"] for row in sliced]],
            "metadatas": [[dict(row["metadata"]) for row in sliced]],
            "distances": [[row["distance"] for row in sliced]],
        }


def _build_retriever() -> StoreRetriever:
    retriever = StoreRetriever.__new__(StoreRetriever)
    retriever.embedder = _DummyEmbedder()
    retriever.chroma = _DummyChroma(rows=8)
    retriever.collection_map = {"code_symbols": "code_symbols"}
    retriever.base_collection_keys = ["code_symbols"]
    retriever.audit_collection_keys = []
    retriever._last_retrieve_debug = {}
    retriever.cfg = SimpleNamespace(
        chat=SimpleNamespace(
            max_context_chunks=2,
            retrieval=SimpleNamespace(base_top_k=3),
        ),
        evaluation=SimpleNamespace(
            recall_bias_enabled=True,
            force_detect_min_strong_hits=1,
            weak_hits_min_for_review=1,
            ignore_false_positive_for_downgrade=True,
            retrieval_second_pass_enabled=True,
            retrieval_second_pass_min_candidates=8,
            retrieval_second_pass_multiplier=2.0,
            max_candidates_per_item=60,
        ),
    )
    return retriever


def test_match_signals_handles_case_and_separators_with_token_boundaries():
    signals = ["resetPassword", "user_profile", "transfer-between-own-accounts", "pin"]
    evidence = "Reset password flow and USER profile edit with transfer between own accounts."
    matched = StoreRetriever._match_signals(signals, evidence)

    assert "resetPassword" in matched
    assert "user_profile" in matched
    assert "transfer-between-own-accounts" in matched
    assert "pin" not in matched


def test_retrieve_result_cap_override_is_used_for_evaluation_paths():
    retriever = _build_retriever()

    capped, _ = retriever.retrieve(
        "query",
        ["demo"],
        collection_limits={"code_symbols": 8},
        result_cap=4,
    )
    uncapped_by_eval, _ = retriever.retrieve(
        "query",
        ["demo"],
        collection_limits={"code_symbols": 8},
    )

    assert len(capped) == 4
    assert len(uncapped_by_eval) == 2


def test_second_pass_trigger_conditions():
    retriever = _build_retriever()

    assert retriever._should_run_second_pass(candidate_count=3, matched_strong=[], matched_weak=["w"]) is True
    assert retriever._should_run_second_pass(candidate_count=12, matched_strong=[], matched_weak=[]) is True
    assert retriever._should_run_second_pass(candidate_count=12, matched_strong=["s"], matched_weak=[]) is False
    assert retriever._should_run_second_pass(candidate_count=12, matched_strong=[], matched_weak=["w"]) is False


def test_recall_policy_overrides_not_detected_when_strong_hits_exist():
    retriever = _build_retriever()
    decision = {
        "status": "Not Detected",
        "reason": "None",
        "confidence": 0.2,
        "matched_strong_signals": [],
        "matched_weak_signals": [],
        "false_positive_risks": [],
        "decision_policy": "llm",
    }

    out = retriever._apply_recall_bias(
        rule_id="R1",
        decision=decision,
        matched_strong=["resetPassword"],
        matched_weak=[],
        matched_false_pos=[],
    )

    assert out["status"] == "Detected"
    assert out["decision_policy"] == "recall_override_strong"


def test_recall_policy_overrides_weak_only_not_detected_to_needs_review():
    retriever = _build_retriever()
    decision = {
        "status": "Not Detected",
        "reason": "No evidence",
        "confidence": 0.2,
        "matched_strong_signals": [],
        "matched_weak_signals": [],
        "false_positive_risks": [],
        "decision_policy": "llm",
    }

    out = retriever._apply_recall_bias(
        rule_id="R2",
        decision=decision,
        matched_strong=[],
        matched_weak=["account recovery"],
        matched_false_pos=[],
    )

    assert out["status"] == "Needs Review"
    assert out["decision_policy"] == "recall_override_weak"


def test_false_positive_matches_do_not_downgrade_detected_in_recall_mode():
    retriever = _build_retriever()
    decision = {
        "status": "Detected",
        "reason": "Detected by model",
        "confidence": 0.8,
        "matched_strong_signals": ["resetPassword"],
        "matched_weak_signals": [],
        "false_positive_risks": [],
        "decision_policy": "llm",
    }

    out = retriever._apply_recall_bias(
        rule_id="R3",
        decision=decision,
        matched_strong=["resetPassword"],
        matched_weak=[],
        matched_false_pos=["sample data"],
    )

    assert out["status"] == "Detected"
    assert "False-positive signal matches" in out["reason"]


def test_run_evaluation_uses_retrieval_result_cap_override(tmp_path):
    retriever = StoreRetriever.__new__(StoreRetriever)
    reports_dir = tmp_path / "reports"
    os.makedirs(reports_dir, exist_ok=True)

    retriever.cfg = SimpleNamespace(
        chat=SimpleNamespace(
            max_context_chunks=2,
            retrieval=SimpleNamespace(base_top_k=3),
        ),
        evaluation=SimpleNamespace(
            evidence_per_item=2,
            max_candidates_per_item=5,
            max_snippet_chars=400,
            html_title="Eval",
            recall_bias_enabled=True,
            force_detect_min_strong_hits=1,
            weak_hits_min_for_review=1,
            ignore_false_positive_for_downgrade=True,
            retrieval_result_cap_per_query=7,
            retrieval_second_pass_enabled=False,
            retrieval_second_pass_min_candidates=8,
            retrieval_second_pass_multiplier=2.0,
            llm_auto_split_enabled=True,
            llm_soft_input_tokens=120,
            llm_target_input_tokens=80,
            llm_max_completion_tokens=256,
            llm_batch_max_evidences=2,
            llm_timeout_s=120,
            oom_retry_max_attempts=1,
            oom_retry_shrink_ratio=0.6,
        ),
        io_limits=SimpleNamespace(read_file_max_chars=5000),
    )
    retriever.rule_dimension_preferences = {}
    retriever.base_collection_keys = ["code_symbols"]
    retriever.audit_collection_keys = []
    retriever.collection_map = {"code_symbols": "code_symbols"}
    retriever.evaluation_reports_dir = str(reports_dir)
    retriever.evaluation_log_path = str(tmp_path / "evaluation_log.jsonl")
    retriever.llm = SimpleNamespace(
        usage=lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "missing_usage_responses": 0,
        }
    )

    retriever.get_evaluation_rules = MethodType(
        lambda self: {
            "version": 1,
            "updated_at_utc": "2026-01-01T00:00:00Z",
            "items": [
                {
                    "id": "R1",
                    "title": "Rule",
                    "definition": "Definition",
                    "strong_signals": ["alpha"],
                    "weak_signals": ["beta"],
                    "false_positives": ["gamma"],
                }
            ],
        },
        retriever,
    )
    seen_result_caps: List[Any] = []

    def fake_retrieve(self, query, store_names, collection_limits=None, intent_override="", result_cap=None):
        seen_result_caps.append(result_cap)
        return [], "manual"

    retriever.retrieve = MethodType(fake_retrieve, retriever)  # type: ignore[method-assign]
    retriever._evaluate_rule_with_llm = MethodType(
        lambda self, **kwargs: {
            "status": "Not Detected",
            "reason": "None",
            "confidence": 0.0,
            "evidence_ids": [],
            "matched_strong_signals": [],
            "matched_weak_signals": [],
            "false_positive_risks": [],
            "decision_policy": "llm",
        },
        retriever,
    )

    summary = retriever.run_evaluation(repo_name="demo", repo_path=str(tmp_path))

    assert summary["rule_count"] == 1
    assert seen_result_caps
    assert all(cap == 7 for cap in seen_result_caps)
