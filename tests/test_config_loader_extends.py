from __future__ import annotations

from pathlib import Path

import pytest

from agentic_rag.core.config_loader import ConfigLoaderError, load_agentic_rag_config


ROOT = Path(__file__).resolve().parents[1]


def test_split_configs_extend_common_with_expected_overrides():
    chat_cfg = load_agentic_rag_config(str(ROOT / "config.chat.yml"))
    ops_cfg = load_agentic_rag_config(str(ROOT / "config.embedding.yml"))

    assert chat_cfg.chat.api.port == 8025
    assert ops_cfg.chat.api.port == 8026
    assert chat_cfg.embedding.base_url == "http://127.0.0.1:8026/v1"
    assert ops_cfg.embedding.base_url == "http://localhost:11434/v1"
    assert chat_cfg.llm.model == ops_cfg.llm.model


def test_extends_cycle_is_rejected(tmp_path: Path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text('extends: "./second.yml"\n', encoding="utf-8")
    second.write_text('extends: "./first.yml"\n', encoding="utf-8")

    with pytest.raises(ConfigLoaderError, match="cycle"):
        load_agentic_rag_config(str(first))


def test_evaluation_recall_defaults_are_loaded():
    cfg = load_agentic_rag_config(str(ROOT / "config.yml"))
    evaluation = cfg.evaluation

    assert evaluation.recall_bias_enabled is True
    assert evaluation.force_detect_min_strong_hits == 1
    assert evaluation.weak_hits_min_for_review == 1
    assert evaluation.ignore_false_positive_for_downgrade is True
    assert evaluation.retrieval_result_cap_per_query == 20
    assert evaluation.retrieval_second_pass_enabled is True
    assert evaluation.retrieval_second_pass_min_candidates == 8
    assert evaluation.retrieval_second_pass_multiplier == 2.0


def test_invalid_evaluation_recall_multiplier_is_rejected(tmp_path: Path):
    cfg_path = tmp_path / "invalid.yml"
    common_path = ROOT / "config.common.yml"
    cfg_path.write_text(
        f'extends: "{common_path.as_posix()}"\n'
        "evaluation:\n"
        "  retrieval_second_pass_multiplier: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigLoaderError, match="retrieval_second_pass_multiplier"):
        load_agentic_rag_config(str(cfg_path))
