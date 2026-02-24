from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any, Dict, List

from retreiving_module.service import StoreRetriever


def _build_retriever() -> StoreRetriever:
    retriever = StoreRetriever.__new__(StoreRetriever)
    retriever.llm = SimpleNamespace(timeout_s=60)
    retriever.cfg = SimpleNamespace(
        evaluation=SimpleNamespace(
            recall_bias_enabled=True,
            force_detect_min_strong_hits=1,
            weak_hits_min_for_review=1,
            ignore_false_positive_for_downgrade=True,
            retrieval_result_cap_per_query=20,
            retrieval_second_pass_enabled=True,
            retrieval_second_pass_min_candidates=8,
            retrieval_second_pass_multiplier=2.0,
            llm_auto_split_enabled=True,
            llm_soft_input_tokens=120,
            llm_target_input_tokens=80,
            llm_max_completion_tokens=256,
            llm_batch_max_evidences=2,
            llm_timeout_s=120,
            oom_retry_max_attempts=2,
            oom_retry_shrink_ratio=0.6,
        )
    )
    return retriever


def _rule() -> Dict[str, Any]:
    return {
        "id": "R1",
        "title": "Audit rule",
        "definition": "Check auth and policy controls.",
        "strong_signals": ["auth", "policy"],
        "weak_signals": ["guard"],
        "false_positives": ["comment only"],
    }


def _evidences() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in range(1, 5):
        out.append(
            {
                "citation_id": idx,
                "file_path": f"f{idx}.py",
                "start_line": 1,
                "end_line": 5,
                "node_type": "function",
                "collection": "code_symbols",
                "distance": 0.1 * idx,
                "snippet": ("auth policy guard " * 30) + str(idx),
            }
        )
    return out


def test_evaluation_splits_and_calls_synthesis():
    retriever = _build_retriever()
    calls: List[str] = []

    def fake_call(self, messages, *, phase, split_batch_idx=None, split_batch_total=None):
        calls.append(str(phase))
        return {
            "status": "Detected",
            "reason": "ok",
            "confidence": 0.8,
            "evidence_ids": [1],
            "matched_strong_signals": ["auth"],
            "matched_weak_signals": ["guard"],
            "false_positive_risks": [],
        }

    retriever._call_evaluation_llm_json = MethodType(fake_call, retriever)  # type: ignore[attr-defined]
    decision = retriever._evaluate_rule_with_llm(
        rule=_rule(),
        evidences=_evidences(),
        matched_strong=["auth"],
        matched_weak=["guard"],
        matched_false_pos=[],
    )
    assert any(phase == "evaluation_batch" for phase in calls)
    assert any(phase == "evaluation_synthesis" for phase in calls)
    assert decision["status"] in {"Detected", "Not Detected", "Needs Review"}


def test_evaluation_synthesis_failure_uses_deterministic_fallback():
    retriever = _build_retriever()

    def fake_call(self, messages, *, phase, split_batch_idx=None, split_batch_total=None):
        if phase == "evaluation_synthesis":
            raise RuntimeError("synth failed")
        return {
            "status": "Needs Review",
            "reason": "batch",
            "confidence": 0.4,
            "evidence_ids": [1],
            "matched_strong_signals": ["auth"],
            "matched_weak_signals": ["guard"],
            "false_positive_risks": [],
        }

    retriever._call_evaluation_llm_json = MethodType(fake_call, retriever)  # type: ignore[attr-defined]
    decision = retriever._evaluate_rule_with_llm(
        rule=_rule(),
        evidences=_evidences(),
        matched_strong=["auth"],
        matched_weak=["guard"],
        matched_false_pos=[],
    )
    assert "fallback" in str(decision.get("reason", "")).lower()
    assert decision["status"] in {"Detected", "Not Detected", "Needs Review"}


def test_evaluation_oom_retry_recovers():
    retriever = _build_retriever()
    state = {"count": 0}

    def fake_call(self, messages, *, phase, split_batch_idx=None, split_batch_total=None):
        state["count"] += 1
        if state["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return {
            "status": "Detected",
            "reason": "ok",
            "confidence": 0.8,
            "evidence_ids": [1],
            "matched_strong_signals": ["auth"],
            "matched_weak_signals": ["guard"],
            "false_positive_risks": [],
        }

    retriever._call_evaluation_llm_json = MethodType(fake_call, retriever)  # type: ignore[attr-defined]
    decision = retriever._evaluate_rule_with_llm(
        rule=_rule(),
        evidences=_evidences(),
        matched_strong=["auth"],
        matched_weak=["guard"],
        matched_false_pos=[],
    )
    assert state["count"] > 1
    assert decision["status"] in {"Detected", "Not Detected", "Needs Review"}


def test_split_decision_can_be_upgraded_by_recall_bias():
    retriever = _build_retriever()

    def fake_call(self, messages, *, phase, split_batch_idx=None, split_batch_total=None):
        return {
            "status": "Not Detected",
            "reason": "batch thinks absent",
            "confidence": 0.3,
            "evidence_ids": [1],
            "matched_strong_signals": [],
            "matched_weak_signals": [],
            "false_positive_risks": [],
        }

    retriever._call_evaluation_llm_json = MethodType(fake_call, retriever)  # type: ignore[attr-defined]
    decision = retriever._evaluate_rule_with_llm(
        rule=_rule(),
        evidences=_evidences(),
        matched_strong=["auth"],
        matched_weak=["guard"],
        matched_false_pos=[],
    )
    adjusted = retriever._apply_recall_bias(
        rule_id="R1",
        decision=decision,
        matched_strong=["auth"],
        matched_weak=["guard"],
        matched_false_pos=[],
    )
    assert adjusted["status"] == "Detected"
    assert adjusted["decision_policy"] == "recall_override_strong"
