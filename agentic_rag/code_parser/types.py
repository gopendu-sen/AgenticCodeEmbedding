from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agentic_rag.core.utils import sha256_text


@dataclass
class CodeNode:
    node_id: str
    node_type: str
    language: str
    file_path: str
    start_line: int
    end_line: int
    symbol: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    confidence: float = 1.0

    def finalize(self):
        if not self.content_hash:
            self.content_hash = sha256_text(self.text)
        return self
