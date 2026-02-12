from typing import Any, Dict, List

import requests


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, timeout_s: int):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._usage: Dict[str, int] = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "missing_usage_responses": 0,
            "input_items": 0,
        }
        self._last_call_usage: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_usage": False,
            "input_items": 0,
        }

    @staticmethod
    def _coerce_int(value: Any) -> int:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _normalize_usage(cls, body: Dict[str, Any]) -> Dict[str, Any]:
        usage_raw = body.get("usage")
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        has_usage = False

        if isinstance(usage_raw, dict):
            prompt_tokens = cls._coerce_int(
                usage_raw.get("prompt_tokens", usage_raw.get("input_tokens", usage_raw.get("prompt_eval_count")))
            )
            completion_tokens = cls._coerce_int(
                usage_raw.get("completion_tokens", usage_raw.get("output_tokens", usage_raw.get("eval_count")))
            )
            total_tokens = cls._coerce_int(usage_raw.get("total_tokens"))
            has_usage = any(
                key in usage_raw for key in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens", "total_tokens")
            )
        else:
            prompt_tokens = cls._coerce_int(body.get("prompt_eval_count", body.get("prompt_tokens")))
            completion_tokens = cls._coerce_int(body.get("eval_count", body.get("completion_tokens")))
            total_tokens = cls._coerce_int(body.get("total_tokens"))
            has_usage = prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0

        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "has_usage": has_usage,
        }

    def _record_usage(self, body: Dict[str, Any], input_items: int) -> None:
        usage = self._normalize_usage(body)
        self._usage["requests"] += 1
        self._usage["prompt_tokens"] += int(usage["prompt_tokens"])
        self._usage["completion_tokens"] += int(usage["completion_tokens"])
        self._usage["total_tokens"] += int(usage["total_tokens"])
        self._usage["input_items"] += max(0, int(input_items))
        if not usage["has_usage"]:
            self._usage["missing_usage_responses"] += 1
        self._last_call_usage = {
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "has_usage": bool(usage["has_usage"]),
            "input_items": max(0, int(input_items)),
        }

    def reset_usage(self) -> None:
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "missing_usage_responses": 0,
            "input_items": 0,
        }
        self._last_call_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_usage": False,
            "input_items": 0,
        }

    def usage(self) -> Dict[str, int]:
        return {
            "requests": int(self._usage.get("requests", 0)),
            "prompt_tokens": int(self._usage.get("prompt_tokens", 0)),
            "completion_tokens": int(self._usage.get("completion_tokens", 0)),
            "total_tokens": int(self._usage.get("total_tokens", 0)),
            "missing_usage_responses": int(self._usage.get("missing_usage_responses", 0)),
            "input_items": int(self._usage.get("input_items", 0)),
        }

    def last_call_usage(self) -> Dict[str, Any]:
        return dict(self._last_call_usage)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """OpenAI-compatible embeddings: expects response.data[].embedding"""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        self._record_usage(body, input_items=len(texts))
        data = body["data"]
        return [item["embedding"] for item in data]
