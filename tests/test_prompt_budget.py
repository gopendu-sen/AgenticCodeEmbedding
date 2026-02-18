from __future__ import annotations

from agentic_rag.core.prompt_budget import (
    estimate_chars_for_messages,
    estimate_tokens_for_messages,
    estimate_tokens_from_text,
    is_oom_error,
    pack_items_by_token_budget,
)


def test_estimate_tokens_from_text_is_stable():
    assert estimate_tokens_from_text("") == 0
    assert estimate_tokens_from_text("abcd", chars_per_token=4) == 1
    assert estimate_tokens_from_text("a" * 33, chars_per_token=3.2) >= 10


def test_estimate_tokens_for_messages_and_chars():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello world"},
    ]
    assert estimate_chars_for_messages(messages) >= len("systemrulesuserhello world")
    assert estimate_tokens_for_messages(messages) > 0


def test_pack_items_by_token_budget_respects_target():
    items = [10, 20, 30, 40]
    batches = pack_items_by_token_budget(
        items,
        item_token_fn=lambda value: value,
        soft_budget=100,
        target_budget=50,
    )
    assert batches
    for batch in batches:
        assert sum(batch) <= 100
    # Prefer batching near target rather than one giant batch.
    assert len(batches) >= 2


def test_is_oom_error_catches_common_signatures():
    assert is_oom_error("CUDA out of memory")
    assert is_oom_error("kv cache allocation failed")
    assert not is_oom_error("connection reset by peer")
