import json
import logging
import time
from typing import Any, Dict, Iterator, List, Optional

import requests


logger = logging.getLogger(__name__)


class LLMClient:
    _RETRYABLE_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: int,
        retry_attempts: int = 2,
        retry_backoff_base_s: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_base_s = max(0.0, float(retry_backoff_base_s))
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

    @classmethod
    def _is_retryable_exception(cls, exc: Exception) -> bool:
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            if response is None:
                return False
            return int(response.status_code) in cls._RETRYABLE_HTTP_STATUSES
        return False

    def _retry_sleep_seconds(self, retry_number: int) -> float:
        # Exponential backoff with bounded growth.
        return min(12.0, self.retry_backoff_base_s * (2 ** max(0, retry_number - 1)))

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        started = time.perf_counter()
        logger.info(
            "LLM REST call started: endpoint=%s model=%s stream=false messages=%d timeout_s=%d",
            url,
            self.model,
            len(messages),
            self.timeout_s,
        )
        attempts = self.retry_attempts + 1
        r: Optional[requests.Response] = None
        body: Dict[str, Any] = {}
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                body = r.json()
                break
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_exception(exc)
                has_next = attempt < attempts
                if retryable and has_next:
                    sleep_s = self._retry_sleep_seconds(attempt)
                    logger.warning(
                        "LLM REST call attempt failed; retrying: endpoint=%s model=%s attempt=%d/%d sleep_s=%.2f error=%s",
                        url,
                        self.model,
                        attempt,
                        attempts,
                        sleep_s,
                        exc,
                    )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.exception(
                    "LLM REST call failed: endpoint=%s model=%s attempt=%d/%d elapsed_ms=%.2f error=%s",
                    url,
                    self.model,
                    attempt,
                    attempts,
                    elapsed_ms,
                    exc,
                )
                raise

        self._record_usage(body)
        usage = self.last_call_usage()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            (
                "LLM REST call completed: endpoint=%s model=%s status=%d elapsed_ms=%.2f "
                "input_tokens=%d output_tokens=%d total_tokens=%d usage_reported=%s"
            ),
            url,
            self.model,
            r.status_code if r is not None else -1,
            elapsed_ms,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            int(usage.get("total_tokens", 0)),
            bool(usage.get("has_usage", False)),
        )
        return body["choices"][0]["message"]["content"].strip()

    def chat_stream(self, messages: List[Dict[str, str]], temperature: float = 0) -> Iterator[str]:
        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "stream": True}
        started = time.perf_counter()
        logger.info(
            "LLM REST stream started: endpoint=%s model=%s stream=true messages=%d timeout_s=%d",
            url,
            self.model,
            len(messages),
            self.timeout_s,
        )
        attempts = self.retry_attempts + 1
        for attempt in range(1, attempts + 1):
            response: Optional[requests.Response] = None
            usage_payload: Optional[Dict[str, Any]] = None
            yielded_chunks = 0
            yielded_chars = 0
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_s, stream=True)
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8").strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue

                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        logger.debug("Skipping non-JSON stream line: %s", line)
                        continue

                    if isinstance(chunk.get("usage"), dict):
                        usage_payload = chunk["usage"]  # type: ignore[assignment]

                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") if isinstance(choice, dict) else None
                    if isinstance(delta, dict):
                        token = delta.get("content")
                        if isinstance(token, str) and token:
                            yielded_chunks += 1
                            yielded_chars += len(token)
                            yield token
                            continue
                    message = choice.get("message") if isinstance(choice, dict) else None
                    if isinstance(message, dict):
                        token = message.get("content")
                        if isinstance(token, str) and token:
                            yielded_chunks += 1
                            yielded_chars += len(token)
                            yield token

                if usage_payload is not None:
                    self._record_usage({"usage": usage_payload})
                else:
                    self._usage["requests"] += 1
                    self._usage["missing_usage_responses"] += 1
                    self._last_call_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "has_usage": False,
                    }
                usage = self.last_call_usage()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.info(
                    (
                        "LLM REST stream completed: endpoint=%s model=%s status=%d elapsed_ms=%.2f chunks=%d chars=%d "
                        "input_tokens=%d output_tokens=%d total_tokens=%d usage_reported=%s"
                    ),
                    url,
                    self.model,
                    response.status_code if response is not None else -1,
                    elapsed_ms,
                    yielded_chunks,
                    yielded_chars,
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                    int(usage.get("total_tokens", 0)),
                    bool(usage.get("has_usage", False)),
                )
                return
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_exception(exc)
                has_next = attempt < attempts
                if retryable and has_next and yielded_chunks == 0 and yielded_chars == 0:
                    sleep_s = self._retry_sleep_seconds(attempt)
                    logger.warning(
                        "LLM REST stream attempt failed before tokens; retrying: endpoint=%s model=%s attempt=%d/%d sleep_s=%.2f error=%s",
                        url,
                        self.model,
                        attempt,
                        attempts,
                        sleep_s,
                        exc,
                    )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if yielded_chunks > 0 or yielded_chars > 0:
                    logger.exception(
                        (
                            "LLM REST stream failed after partial output; no retry to avoid duplicate tokens: "
                            "endpoint=%s model=%s attempt=%d/%d elapsed_ms=%.2f chunks=%d chars=%d error=%s"
                        ),
                        url,
                        self.model,
                        attempt,
                        attempts,
                        elapsed_ms,
                        yielded_chunks,
                        yielded_chars,
                        exc,
                    )
                else:
                    logger.exception(
                        "LLM REST stream failed to start: endpoint=%s model=%s attempt=%d/%d elapsed_ms=%.2f error=%s",
                        url,
                        self.model,
                        attempt,
                        attempts,
                        elapsed_ms,
                        exc,
                    )
                raise
            finally:
                if response is not None:
                    response.close()

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
        started = time.perf_counter()
        logger.info("LLM probe request started: endpoint=%s model=%s timeout_s=%s", url, self.model, self.timeout_s)
        attempts = self.retry_attempts + 1
        r: Optional[requests.Response] = None
        body: Dict[str, Any] = {}
        for attempt in range(1, attempts + 1):
            try:
                r = requests.post(url, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                body = r.json()
                break
            except Exception as exc:  # noqa: BLE001
                retryable = self._is_retryable_exception(exc)
                has_next = attempt < attempts
                if retryable and has_next:
                    sleep_s = self._retry_sleep_seconds(attempt)
                    logger.warning(
                        "LLM probe attempt failed; retrying: endpoint=%s model=%s attempt=%d/%d sleep_s=%.2f error=%s",
                        url,
                        self.model,
                        attempt,
                        attempts,
                        sleep_s,
                        exc,
                    )
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.exception(
                    "LLM probe failed: endpoint=%s model=%s attempt=%d/%d elapsed_ms=%.2f error=%s",
                    url,
                    self.model,
                    attempt,
                    attempts,
                    elapsed_ms,
                    exc,
                )
                raise

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("LLM probe failed: missing choices in response")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "LLM probe response received: endpoint=%s model=%s status=%d elapsed_ms=%.2f choices=%d",
            url,
            self.model,
            r.status_code if r is not None else -1,
            elapsed_ms,
            len(choices),
        )
        return body
