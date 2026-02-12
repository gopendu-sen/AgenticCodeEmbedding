from typing import Dict, List, Optional

from agentic_rag.code_parser.types import CodeNode


def _nearest_before(nodes: List[CodeNode], line_no: int, wanted: set[str]) -> Optional[CodeNode]:
    candidates = [n for n in nodes if n.node_type in wanted and n.end_line <= line_no]
    if not candidates:
        return None
    return max(candidates, key=lambda n: n.end_line)


def _nearest_after(nodes: List[CodeNode], line_no: int, wanted: set[str]) -> Optional[CodeNode]:
    candidates = [n for n in nodes if n.node_type in wanted and n.start_line >= line_no]
    if not candidates:
        return None
    return min(candidates, key=lambda n: n.start_line)


def build_flow_chain_nodes(nodes: List[CodeNode]) -> List[CodeNode]:
    by_file: Dict[str, List[CodeNode]] = {}
    for node in nodes:
        by_file.setdefault(node.file_path, []).append(node)

    out: List[CodeNode] = []
    for file_path, file_nodes in by_file.items():
        route_nodes = [n for n in file_nodes if n.node_type in {"route", "entrypoint"}]
        sensitive_nodes = [n for n in file_nodes if n.node_type == "sensitive_op"]
        if not route_nodes or not sensitive_nodes:
            continue

        for route in route_nodes:
            for sensitive in sensitive_nodes:
                if sensitive.start_line < route.start_line:
                    continue
                guard = _nearest_before(file_nodes, sensitive.start_line, {"auth_guard", "policy_check"})
                audit = _nearest_after(file_nodes, sensitive.end_line, {"audit_log"})
                if not guard or not audit:
                    continue

                start_line = min(route.start_line, guard.start_line, sensitive.start_line, audit.start_line)
                end_line = max(route.end_line, guard.end_line, sensitive.end_line, audit.end_line)
                route_symbol = route.symbol or f"route@{route.start_line}"
                sensitive_symbol = sensitive.symbol or f"sensitive@{sensitive.start_line}"
                flow_symbol = f"{route_symbol}->{sensitive_symbol}"
                flow_text = "\n".join(
                    [
                        f"ENTRYPOINT: {route_symbol}",
                        f"SECURITY_CHECK: {guard.symbol or guard.node_type}",
                        f"SENSITIVE_OPERATION: {sensitive_symbol}",
                        f"AUDIT_LOG: {audit.symbol or audit.node_type}",
                    ]
                )
                out.append(
                    CodeNode(
                        node_id=f"flow_chain::{file_path}::{flow_symbol}::{start_line}-{end_line}",
                        node_type="flow_chain",
                        language=route.language,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        symbol=flow_symbol,
                        text=flow_text,
                        metadata={
                            "entrypoint": route_symbol,
                            "security_check": guard.symbol or guard.node_type,
                            "sensitive_op": sensitive_symbol,
                            "audit_log": audit.symbol or audit.node_type,
                            "source": "flow_builder",
                        },
                        confidence=0.82,
                    ).finalize()
                )
    return out
