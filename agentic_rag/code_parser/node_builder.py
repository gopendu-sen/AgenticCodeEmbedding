from typing import Any, Dict, Iterable, List, Optional, Set

from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.utils import sha256_text


_SECURITY_NODE_TYPES = {
    "auth_guard",
    "policy_check",
    "audit_log",
    "sensitive_op",
}


def _build_nodes_from_items(
    file_path: str,
    file_text: str,
    language: str,
    items: Iterable[Dict[str, Any]],
    source: str,
    allowed_node_types: Optional[Set[str]] = None,
    required_metadata_keys: Optional[Set[str]] = None,
) -> List[CodeNode]:
    lines = file_text.splitlines()
    nodes_out: List[CodeNode] = []

    for item in items:
        try:
            s = int(item["start_line"])
            e = int(item["end_line"])
        except Exception:
            continue

        s = max(1, s)
        e = min(len(lines), e)
        if e < s:
            continue

        snippet = "\n".join(lines[s - 1:e])
        node_type = item.get("node_type", "unknown")
        if allowed_node_types is not None and node_type not in allowed_node_types:
            continue
        symbol = item.get("symbol")
        conf = float(item.get("confidence", 0.6))

        node_id = f"{node_type}::{file_path}::{symbol or 'NA'}::{s}-{e}"
        metadata: Dict[str, Any] = {"why_relevant": item.get("why_relevant", ""), "source": source}
        existing_meta = item.get("metadata")
        if isinstance(existing_meta, dict):
            metadata.update(existing_meta)
        for key in ("control_area", "severity", "evidence"):
            raw = item.get(key)
            if raw is not None and str(raw).strip():
                metadata[key] = str(raw).strip()
        if required_metadata_keys:
            for key in required_metadata_keys:
                metadata.setdefault(key, "")

        nodes_out.append(CodeNode(
            node_id=node_id,
            node_type=node_type,
            language=language,
            file_path=file_path,
            start_line=s,
            end_line=e,
            symbol=symbol,
            text=snippet,
            metadata=metadata,
            content_hash=sha256_text(snippet),
            confidence=conf
        ).finalize())

    return nodes_out


def build_nodes_from_agent(file_path: str, file_text: str, language: str, agent_json: Dict[str, Any]) -> List[CodeNode]:
    final = agent_json.get("final", {})
    items = final.get("nodes", [])
    if not isinstance(items, list):
        return []
    normalized = [item for item in items if isinstance(item, dict)]
    return _build_nodes_from_items(
        file_path=file_path,
        file_text=file_text,
        language=language,
        items=normalized,
        source="llm_fallback",
    )


def build_security_nodes_from_agent(
    file_path: str,
    file_text: str,
    language: str,
    agent_json: Dict[str, Any],
) -> List[CodeNode]:
    final = agent_json.get("final", {})
    items = final.get("nodes", [])
    if not isinstance(items, list):
        return []
    normalized = [item for item in items if isinstance(item, dict)]
    return _build_nodes_from_items(
        file_path=file_path,
        file_text=file_text,
        language=language,
        items=normalized,
        source="llm_security_tagger",
        allowed_node_types=_SECURITY_NODE_TYPES,
        required_metadata_keys={"why_relevant", "control_area", "severity", "evidence", "source"},
    )
