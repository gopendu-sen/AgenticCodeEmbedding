"""Prompt budgeting helpers for memory-safe LLM calls."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar


T = TypeVar("T")

_OOM_HINTS = (
    "cuda",
    "out of memory",
    "oom",
    "insufficient memory",
    "kv cache",
)


def estimate_tokens_from_text(text: str, *, chars_per_token: float = 3.2) -> int:
    if not text:
        return 0
    ratio = chars_per_token if chars_per_token > 0 else 3.2
    return max(1, int(math.ceil(len(text) / ratio)))


def estimate_chars_for_messages(messages: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        total += len(str(message.get("role", "")))
        total += len(str(message.get("content", "")))
    return total


def estimate_tokens_for_messages(
    messages: Iterable[Dict[str, Any]],
    *,
    chars_per_token: float = 3.2,
    per_message_overhead_tokens: int = 8,
) -> int:
    total_tokens = 0
    overhead = max(0, int(per_message_overhead_tokens))
    for message in messages:
        if not isinstance(message, dict):
            continue
        role_text = str(message.get("role", ""))
        content_text = str(message.get("content", ""))
        total_tokens += estimate_tokens_from_text(role_text, chars_per_token=chars_per_token)
        total_tokens += estimate_tokens_from_text(content_text, chars_per_token=chars_per_token)
        total_tokens += overhead
    return total_tokens


def pack_items_by_token_budget(
    items: Sequence[T],
    item_token_fn: Callable[[T], int],
    *,
    soft_budget: int,
    target_budget: int,
) -> List[List[T]]:
    if not items:
        return []
    soft = max(1, int(soft_budget))
    target = max(1, min(int(target_budget), soft))

    batches: List[List[T]] = []
    current: List[T] = []
    current_tokens = 0

    for item in items:
        item_tokens = max(1, int(item_token_fn(item)))
        if current and current_tokens + item_tokens > target:
            batches.append(current)
            current = []
            current_tokens = 0

        if not current:
            current = [item]
            current_tokens = item_tokens
            continue

        if current_tokens + item_tokens <= soft:
            current.append(item)
            current_tokens += item_tokens
            continue

        batches.append(current)
        current = [item]
        current_tokens = item_tokens

    if current:
        batches.append(current)
    return batches


def is_oom_error(exc_or_text: Any) -> bool:
    text = str(exc_or_text or "").lower()
    if not text:
        return False
    return any(hint in text for hint in _OOM_HINTS)


def build_llm_call_log_fields(
    *,
    phase: str,
    messages: Iterable[Dict[str, Any]],
    split_batch_idx: Optional[int] = None,
    split_batch_total: Optional[int] = None,
) -> Dict[str, Any]:
    message_list = list(messages)
    payload: Dict[str, Any] = {
        "phase": phase,
        "estimated_input_tokens": estimate_tokens_for_messages(message_list),
        "estimated_input_chars": estimate_chars_for_messages(message_list),
    }
    if split_batch_idx is not None:
        payload["split_batch_idx"] = int(split_batch_idx)
    if split_batch_total is not None:
        payload["split_batch_total"] = int(split_batch_total)
    return payload
