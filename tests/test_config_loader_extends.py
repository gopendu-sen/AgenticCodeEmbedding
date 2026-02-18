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
