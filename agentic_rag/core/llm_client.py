import json
import logging
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class LLMClient:
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
        }
        self._last_call_usage: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_usage": False,
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

    def _record_usage(self, body: Dict[str, Any]) -> None:
        usage = self._normalize_usage(body)
        self._usage["requests"] += 1
        self._usage["prompt_tokens"] += int(usage["prompt_tokens"])
        self._usage["completion_tokens"] += int(usage["completion_tokens"])
        self._usage["total_tokens"] += int(usage["total_tokens"])
        if not usage["has_usage"]:
            self._usage["missing_usage_responses"] += 1
        self._last_call_usage = {
            "prompt_tokens": int(usage["prompt_tokens"]),
            "completion_tokens": int(usage["completion_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "has_usage": bool(usage["has_usage"]),
        }

    def reset_usage(self) -> None:
        self._usage = {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "missing_usage_responses": 0,
        }
        self._last_call_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "has_usage": False,
        }

    def usage(self) -> Dict[str, int]:
        return {
            "requests": int(self._usage.get("requests", 0)),
            "prompt_tokens": int(self._usage.get("prompt_tokens", 0)),
            "completion_tokens": int(self._usage.get("completion_tokens", 0)),
            "total_tokens": int(self._usage.get("total_tokens", 0)),
            "missing_usage_responses": int(self._usage.get("missing_usage_responses", 0)),
        }

    def last_call_usage(self) -> Dict[str, Any]:
        return dict(self._last_call_usage)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        self._record_usage(body)
        return body["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _extract_json_segment(text: str) -> Optional[str]:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for idx, ch in enumerate(text[start:], start=start):
            if escaped:
                escaped = False
                continue
            if ch == "\\" and in_string:
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    @classmethod
    def _loads_best_effort(cls, content: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
            raise ValueError("LLM JSON response must be a JSON object")
        except (json.JSONDecodeError, ValueError):
            segment = cls._extract_json_segment(content)
            if not segment:
                raise
            data = json.loads(segment)
            if not isinstance(data, dict):
                raise ValueError("LLM JSON response must be a JSON object")
            return data

    def chat_json(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        content = self.chat(messages, temperature=0)
        try:
            return self._loads_best_effort(content)
        except Exception as first_exc:  # noqa: BLE001
            logger.warning("LLM JSON parse failed on first attempt: model=%s error=%s", self.model, first_exc)
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "You normalize model output into strict JSON. "
                        "Return ONLY a valid JSON object with no markdown."
                    ),
                },
                {"role": "user", "content": content},
            ]
            repaired = self.chat(repair_messages, temperature=0)
            return self._loads_best_effort(repaired)

    def probe(self) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return exactly: ok"},
                {"role": "user", "content": "ping"},
            ],
            "temperature": 0,
            "max_tokens": 1,
        }
        logger.debug("LLM probe request: url=%s model=%s timeout_s=%s", url, self.model, self.timeout_s)
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM probe failed: missing choices in response")
        logger.debug("LLM probe response received: choices=%d", len(choices))
        return body
