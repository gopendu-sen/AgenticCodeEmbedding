from __future__ import annotations

from typing import Any, Dict, List

import pytest
import requests

from agentic_rag.core.llm_client import LLMClient


class _Response:
    def __init__(self, *, status_code: int = 200, body: Dict[str, Any] | None = None, stream_lines: List[str] | None = None):
        self.status_code = status_code
        self._body = body or {}
        self._stream_lines = stream_lines or []

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"status={self.status_code}")
            err.response = self
            raise err

    def json(self) -> Dict[str, Any]:
        return self._body

    def iter_lines(self):
        for line in self._stream_lines:
            yield line.encode("utf-8")

    def close(self) -> None:
        return None


def test_chat_retries_on_connection_error(monkeypatch: pytest.MonkeyPatch):
    client = LLMClient(base_url="http://llm.local/v1", model="m", timeout_s=5, retry_attempts=2, retry_backoff_base_s=0)
    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ConnectionError("temporary")
        return _Response(
            status_code=200,
            body={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr("agentic_rag.core.llm_client.requests.post", fake_post)

    out = client.chat([{"role": "user", "content": "hello"}], temperature=0)
    assert out == "ok"
    assert calls["count"] == 2


def test_probe_retries_on_retryable_http_error(monkeypatch: pytest.MonkeyPatch):
    client = LLMClient(base_url="http://llm.local/v1", model="m", timeout_s=5, retry_attempts=2, retry_backoff_base_s=0)
    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return _Response(status_code=503, body={"error": "down"})
        return _Response(status_code=200, body={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("agentic_rag.core.llm_client.requests.post", fake_post)

    out = client.probe()
    assert isinstance(out, dict)
    assert calls["count"] == 2


def test_chat_stream_retries_before_first_token(monkeypatch: pytest.MonkeyPatch):
    client = LLMClient(base_url="http://llm.local/v1", model="m", timeout_s=5, retry_attempts=2, retry_backoff_base_s=0)
    calls = {"count": 0}

    def fake_post(*_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.Timeout("start timeout")
        return _Response(
            status_code=200,
            stream_lines=[
                'data: {"choices":[{"delta":{"content":"hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                "data: [DONE]",
            ],
        )

    monkeypatch.setattr("agentic_rag.core.llm_client.requests.post", fake_post)

    tokens = list(client.chat_stream([{"role": "user", "content": "hello"}], temperature=0))
    assert "".join(tokens) == "hello"
    assert calls["count"] == 2
