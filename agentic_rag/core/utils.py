import hashlib
import json
from typing import Any, Dict


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_json_dumps(d: Dict[str, Any], max_len: int = 4000) -> str:
    s = json.dumps(d, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "..."
