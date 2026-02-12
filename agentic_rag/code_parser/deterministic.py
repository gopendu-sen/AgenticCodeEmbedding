import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentic_rag.code_parser.types import CodeNode

_MD_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.+?)\s*$")
_MD_FENCE_RE = re.compile(r"^\s*(```|~~~)\s*([A-Za-z0-9_+.\-#]*)")
_CALL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*){0,2})\s*\(")
_CALL_SKIP = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "typeof", "new", "throw",
    "assert", "await", "lambda", "super", "this", "base", "class", "def", "function",
    "defer", "select", "go",
}


def language_hint_from_ext(ext: str) -> str:
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".scala": "scala",
        ".sc": "scala",
        ".cs": "csharp",
        ".cob": "cobol",
        ".cbl": "cobol",
        ".cpy": "cobol",
        ".r": "r",
        ".sql": "sql",
        ".c": "c",
        ".cc": "cpp",
        ".cpp": "cpp",
        ".cxx": "cpp",
        ".h": "cpp",
        ".hh": "cpp",
        ".hpp": "cpp",
        ".hxx": "cpp",
        ".go": "go",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".md": "markdown",
        ".txt": "text",
        ".ipynb": "notebook",
    }.get(ext, "unknown")


def _file_node(
    file_path: str,
    language: str,
    lines: List[str],
    text: str,
    node_text_max_chars: int,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
) -> CodeNode:
    return CodeNode(
        node_id=f"file::{file_path}",
        node_type="file",
        language=language,
        file_path=file_path,
        start_line=1,
        end_line=max(1, len(lines)),
        symbol=None,
        text=text[:node_text_max_chars],
        metadata=metadata or {},
        confidence=confidence,
    ).finalize()


def _find_block_end(lines: List[str], start_line: int, max_span: int = 300) -> int:
    depth = 0
    started = False
    limit = min(len(lines), start_line - 1 + max_span)

    for idx in range(start_line - 1, limit):
        line = lines[idx]
        for ch in line:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}" and started:
                depth -= 1
                if depth <= 0:
                    return idx + 1
        if started and depth <= 0:
            return idx + 1

    if started:
        return limit
    return start_line


def _annotation_block(lines: List[str], line_no: int, window: int = 6, prefix: str = "@") -> Tuple[List[str], int]:
    annotations: List[str] = []
    start_line = line_no

    for idx in range(line_no - 2, max(-1, line_no - window - 2), -1):
        raw = lines[idx].strip()
        if not raw:
            if annotations:
                break
            continue
        if raw.startswith(prefix):
            annotations.append(raw)
            start_line = idx + 1
            continue
        if annotations:
            break

    annotations.reverse()
    return annotations, start_line


def _extract_path(annotations: List[str]) -> str:
    for ann in annotations:
        match = re.search(r"\(\s*(?:value\s*=\s*|path\s*=\s*)?[\"']([^\"']+)[\"']", ann)
        if match:
            return match.group(1)
    return ""


def _add_node(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    node_type: str,
    symbol: Optional[str],
    start_line: int,
    end_line: int,
    lines: List[str],
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 0.9,
) -> bool:
    start_line = max(1, start_line)
    end_line = max(start_line, min(len(lines), end_line))
    node_id = f"{node_type}::{file_path}::{symbol or 'NA'}::{start_line}-{end_line}"

    if node_id in seen:
        return False

    snippet = "\n".join(lines[start_line - 1:end_line])
    nodes.append(CodeNode(
        node_id=node_id,
        node_type=node_type,
        language=language,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        symbol=symbol,
        text=snippet,
        metadata=metadata or {},
        confidence=confidence,
    ).finalize())
    seen.add(node_id)
    return True


def _add_comment_node(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    start_line: int,
    end_line: int,
    lines: List[str],
    kind: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    node_type = {
        "line": "comment_line",
        "line_block": "comment_line",
        "block": "comment_block",
        "docstring": "comment_doc",
    }.get(kind, "comment_other")
    comment_meta = {"comment_kind": kind}
    if metadata:
        comment_meta.update(metadata)
    return _add_node(
        nodes=nodes,
        seen=seen,
        file_path=file_path,
        language=language,
        node_type=node_type,
        symbol=None,
        start_line=start_line,
        end_line=end_line,
        lines=lines,
        metadata=comment_meta,
        confidence=0.78,
    )


def _add_hash_comment_nodes(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    lines: List[str],
) -> None:
    i = 1
    while i <= len(lines):
        stripped = lines[i - 1].strip()
        if not stripped.startswith("#"):
            i += 1
            continue

        start = i
        while i <= len(lines) and lines[i - 1].strip().startswith("#"):
            i += 1
        end = i - 1
        _add_comment_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            lines=lines,
            kind="line_block",
        )


def _add_sql_comment_nodes(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    lines: List[str],
) -> None:
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        stripped = line.strip()

        if stripped.startswith("--"):
            start = i
            while i <= len(lines) and lines[i - 1].strip().startswith("--"):
                i += 1
            end = i - 1
            _add_comment_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language=language,
                start_line=start,
                end_line=end,
                lines=lines,
                kind="line_block",
            )
            continue

        block_start = line.find("/*")
        if block_start == -1:
            i += 1
            continue

        start = i
        block_end = line.find("*/", block_start + 2)
        if block_end != -1:
            end = i
            i += 1
        else:
            i += 1
            end = len(lines)
            while i <= len(lines):
                if "*/" in lines[i - 1]:
                    end = i
                    i += 1
                    break
                i += 1

        _add_comment_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            lines=lines,
            kind="block",
        )


def _add_slash_comment_nodes(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    lines: List[str],
) -> None:
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        stripped = line.strip()

        if stripped.startswith("//"):
            start = i
            while i <= len(lines) and lines[i - 1].strip().startswith("//"):
                i += 1
            end = i - 1
            _add_comment_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language=language,
                start_line=start,
                end_line=end,
                lines=lines,
                kind="line_block",
            )
            continue

        block_start = line.find("/*")
        if block_start == -1:
            i += 1
            continue

        start = i
        block_end = line.find("*/", block_start + 2)
        if block_end != -1:
            end = i
            i += 1
        else:
            i += 1
            end = len(lines)
            while i <= len(lines):
                if "*/" in lines[i - 1]:
                    end = i
                    i += 1
                    break
                i += 1

        _add_comment_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            lines=lines,
            kind="block",
        )


def _add_cobol_comment_nodes(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    lines: List[str],
) -> None:
    i = 1
    while i <= len(lines):
        line = lines[i - 1]
        indicator = line[6] if len(line) > 6 else ""
        stripped = line.strip()
        is_comment = indicator == "*" or stripped.startswith("*>")
        if not is_comment:
            i += 1
            continue

        start = i
        while i <= len(lines):
            cur = lines[i - 1]
            cur_indicator = cur[6] if len(cur) > 6 else ""
            cur_stripped = cur.strip()
            if not (cur_indicator == "*" or cur_stripped.startswith("*>")):
                break
            i += 1
        end = i - 1
        _add_comment_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            lines=lines,
            kind="line_block",
            metadata={"dialect": "cobol"},
        )


def _find_statement_end(lines: List[str], start_line: int, max_span: int = 400) -> int:
    limit = min(len(lines), start_line - 1 + max_span)
    for idx in range(start_line - 1, limit):
        if ";" in lines[idx]:
            return idx + 1
    return limit


def _clean_sql_identifier(raw: str) -> str:
    out = raw.strip().strip(",")
    if out.startswith("[") and out.endswith("]"):
        out = out[1:-1]
    if out.startswith("`") and out.endswith("`"):
        out = out[1:-1]
    if out.startswith('"') and out.endswith('"'):
        out = out[1:-1]
    return out


def _iter_python_docstring_spans(tree: ast.AST) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []

    def _doc_span(node: ast.AST, owner: str) -> None:
        body = getattr(node, "body", None)
        if not body:
            return
        first = body[0]
        if not isinstance(first, ast.Expr):
            return
        value = getattr(first, "value", None)
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return
        start = getattr(first, "lineno", None)
        end = getattr(first, "end_lineno", start)
        if start and end:
            spans.append((start, end, owner))

    if isinstance(tree, ast.Module):
        _doc_span(tree, "module")
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            _doc_span(node, f"class:{node.name}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _doc_span(node, f"function:{node.name}")

    return spans


def _python_call_name(expr: ast.AST) -> Optional[str]:
    if isinstance(expr, ast.Name):
        return expr.id

    if isinstance(expr, ast.Attribute):
        parent = _python_call_name(expr.value)
        if parent:
            return f"{parent}.{expr.attr}"
        return expr.attr

    return None


def _iter_python_call_edges(tree: ast.AST) -> List[Tuple[str, str, int]]:
    edges: List[Tuple[str, str, int]] = []
    class_stack: List[str] = []
    scope_stack: List[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            class_stack.append(node.name)
            self.generic_visit(node)
            class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            prefix = ".".join(class_stack)
            caller = f"{prefix}.{node.name}" if prefix else node.name
            scope_stack.append(caller)
            self.generic_visit(node)
            scope_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            prefix = ".".join(class_stack)
            caller = f"{prefix}.{node.name}" if prefix else node.name
            scope_stack.append(caller)
            self.generic_visit(node)
            scope_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if scope_stack:
                callee = _python_call_name(node.func)
                if callee:
                    edges.append((scope_stack[-1], callee, getattr(node, "lineno", 1)))
            self.generic_visit(node)

    Visitor().visit(tree)
    return edges


def _add_call_edge_node(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    caller: str,
    callee: str,
    line_no: int,
    lines: List[str],
    confidence: float = 0.78,
) -> bool:
    return _add_node(
        nodes=nodes,
        seen=seen,
        file_path=file_path,
        language=language,
        node_type="call_edge",
        symbol=f"{caller}->{callee}",
        start_line=line_no,
        end_line=line_no,
        lines=lines,
        metadata={"relation": "calls", "caller": caller, "callee": callee},
        confidence=confidence,
    )


def _add_import_usage_node(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    line_no: int,
    lines: List[str],
    module_name: str,
    import_kind: str,
    metadata: Optional[Dict[str, Any]] = None,
    confidence: float = 0.78,
) -> bool:
    clean = module_name.strip()
    if not clean:
        return False
    import_meta: Dict[str, Any] = {"import_kind": import_kind}
    if metadata:
        import_meta.update(metadata)
    return _add_node(
        nodes=nodes,
        seen=seen,
        file_path=file_path,
        language=language,
        node_type="import_usage",
        symbol=clean,
        start_line=line_no,
        end_line=line_no,
        lines=lines,
        metadata=import_meta,
        confidence=confidence,
    )


def _add_entrypoint_node(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    line_no: int,
    lines: List[str],
    symbol: str,
    entrypoint_kind: str,
    confidence: float = 0.84,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    entry_meta: Dict[str, Any] = {"entrypoint_kind": entrypoint_kind}
    if metadata:
        entry_meta.update(metadata)
    return _add_node(
        nodes=nodes,
        seen=seen,
        file_path=file_path,
        language=language,
        node_type="entrypoint",
        symbol=symbol,
        start_line=line_no,
        end_line=line_no,
        lines=lines,
        metadata=entry_meta,
        confidence=confidence,
    )


def _add_python_import_usage_nodes(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    lines: List[str],
    tree: ast.AST,
) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name:
                    continue
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    line_no=getattr(node, "lineno", 1),
                    lines=lines,
                    module_name=alias.name,
                    import_kind="import",
                    metadata={"alias": alias.asname or ""},
                    confidence=0.8,
                )
        elif isinstance(node, ast.ImportFrom):
            source = node.module or ""
            prefix = "." * int(getattr(node, "level", 0))
            base = f"{prefix}{source}" if source else prefix
            for alias in node.names:
                name = alias.name or ""
                module_name = f"{base}.{name}" if base and name else (name or base)
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    line_no=getattr(node, "lineno", 1),
                    lines=lines,
                    module_name=module_name,
                    import_kind="from_import",
                    metadata={"alias": alias.asname or ""},
                    confidence=0.8,
                )


def _python_main_guard_lines(tree: ast.AST) -> List[int]:
    out: List[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if not test.comparators:
            continue
        comp = test.comparators[0]
        if not isinstance(comp, ast.Constant) or comp.value != "__main__":
            continue
        out.append(getattr(node, "lineno", 1))
    return out


def _iter_line_call_tokens(line: str) -> List[str]:
    out: List[str] = []
    for match in _CALL_TOKEN_RE.finditer(line):
        token = match.group(1)
        base = token.split(".")[-1].lower()
        if base in _CALL_SKIP:
            continue
        out.append(token)
    return out


def _add_scoped_call_edges(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    language: str,
    lines: List[str],
    scope_nodes: List[CodeNode],
    max_callees_per_scope: int,
) -> None:
    max_callees = max(1, max_callees_per_scope)

    for scope in scope_nodes:
        caller = scope.symbol or f"{scope.node_type}@{scope.start_line}"
        if not caller:
            continue

        first_seen: Dict[str, int] = {}
        scan_start = scope.start_line + 1 if scope.end_line > scope.start_line else scope.start_line
        for line_no in range(scan_start, scope.end_line + 1):
            line = lines[line_no - 1]
            for token in _iter_line_call_tokens(line):
                callee = token.strip()
                if not callee:
                    continue

                caller_leaf = caller.split(".")[-1]
                callee_leaf = callee.split(".")[-1]
                if caller_leaf == callee_leaf:
                    continue

                if callee in first_seen:
                    continue
                first_seen[callee] = line_no

                if len(first_seen) >= max_callees:
                    break
            if len(first_seen) >= max_callees:
                break

        for callee, line_no in first_seen.items():
            _add_call_edge_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language=language,
                caller=caller,
                callee=callee,
                line_no=line_no,
                lines=lines,
                confidence=0.72,
            )


def _chunk_ranges(start_line: int, end_line: int, chunk_lines: int, overlap: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    i = start_line
    while i <= end_line:
        start = i
        end = min(end_line, i + chunk_lines - 1)
        out.append((start, end))
        i = end - overlap + 1
        if i <= start:
            i = end + 1
    return out


def _markdown_headings(lines: List[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    stack: List[Tuple[int, str]] = []

    for line_no, line in enumerate(lines, start=1):
        match = _MD_HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        title = match.group(2).strip().strip("#").strip()
        if not title:
            continue

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        entries.append({
            "line": line_no,
            "level": level,
            "title": title,
            "path": " > ".join(part for _, part in stack),
        })

    return entries


def _markdown_doc_title(file_path: str, headings: List[Dict[str, Any]]) -> str:
    for heading in headings:
        if heading["level"] == 1:
            return str(heading["title"])
    tail = file_path.replace("\\", "/").split("/")[-1]
    return tail.rsplit(".", 1)[0] or tail


def _append_markdown_doc_chunks(
    nodes: List[CodeNode],
    seen: set,
    file_path: str,
    lines: List[str],
    chunk_lines: int,
    overlap: int,
    start_line: int,
    end_line: int,
    doc_title: str,
    section_title: str,
    section_path: str,
    section_level: int,
) -> None:
    if start_line > end_line:
        return
    if not any(lines[idx - 1].strip() for idx in range(start_line, end_line + 1)):
        return

    ranges = _chunk_ranges(start_line, end_line, chunk_lines=chunk_lines, overlap=overlap)
    for start, end in ranges:
        _add_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="markdown",
            node_type="doc",
            symbol=section_title or None,
            start_line=start,
            end_line=end,
            lines=lines,
            metadata={
                "doc_format": "markdown",
                "doc_title": doc_title,
                "section_title": section_title,
                "section_path": section_path,
                "section_level": section_level,
            },
            confidence=0.92 if section_title else 0.88,
        )


def _confidence(symbol_count: int) -> float:
    if symbol_count >= 6:
        return 0.92
    if symbol_count >= 2:
        return 0.82
    if symbol_count == 1:
        return 0.7
    return 0.5


class PythonASTParser:
    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = []

        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            nodes.append(_file_node(
                file_path=file_path,
                language="python",
                lines=lines,
                text=text,
                node_text_max_chars=self.node_text_max_chars,
                metadata={"parse_error": "syntax", "parse_error_detail": str(exc)[:240]},
                confidence=0.2,
            ))
            seen = {nodes[0].node_id}
            _add_hash_comment_nodes(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="python",
                lines=lines,
            )
            return nodes, 0.2

        nodes.append(_file_node(
            file_path=file_path,
            language="python",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        ))
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_hash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="python",
            lines=lines,
        )
        for start, end, owner in _iter_python_docstring_spans(tree):
            _add_comment_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="python",
                start_line=start,
                end_line=end,
                lines=lines,
                kind="docstring",
                metadata={"owner": owner},
            )

        _add_python_import_usage_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            lines=lines,
            tree=tree,
        )

        main_symbol_line: Optional[int] = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(n, "lineno", 1)
                end = getattr(n, "end_lineno", start)
                node_type = "class" if isinstance(n, ast.ClassDef) else "function"
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    node_type=node_type,
                    symbol=n.name,
                    start_line=start,
                    end_line=end,
                    lines=lines,
                    metadata={},
                    confidence=1.0,
                )
                if added:
                    symbol_count += 1
                    if node_type == "function" and n.name == "main":
                        main_symbol_line = start

        if main_symbol_line:
            _add_entrypoint_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="python",
                line_no=main_symbol_line,
                lines=lines,
                symbol="main",
                entrypoint_kind="main_function",
                confidence=0.86,
            )

        for line_no in _python_main_guard_lines(tree):
            _add_entrypoint_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="python",
                line_no=line_no,
                lines=lines,
                symbol="__main__",
                entrypoint_kind="python_main_guard",
                confidence=0.9,
            )

        by_caller: Dict[str, Dict[str, int]] = {}
        for caller, callee, line_no in _iter_python_call_edges(tree):
            if not caller or not callee:
                continue
            caller_edges = by_caller.setdefault(caller, {})
            if callee in caller_edges:
                continue
            if len(caller_edges) >= self.max_callees_per_scope:
                continue
            caller_edges[callee] = line_no

        for caller, callees in by_caller.items():
            for callee, line_no in callees.items():
                _add_call_edge_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    caller=caller,
                    callee=callee,
                    line_no=line_no,
                    lines=lines,
                    confidence=0.84,
                )

        conf = 0.95 if symbol_count > 0 else 0.6
        return nodes, conf


class JsTsParser:
    CLASS_RE = re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+([A-Za-z_]\w*)")
    FUNCTION_RE = re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_]\w*)\s*\(")
    ARROW_RE = re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_]\w*)\s*=>"
    )
    FUNCTION_EXPR_RE = re.compile(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_]\w*)\s*=\s*(?:async\s*)?function\b"
    )

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str, ext: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        language = language_hint_from_ext(ext)

        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language=language,
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            lines=lines,
        )

        is_react = ext in (".jsx", ".tsx") or bool(
            re.search(r"from\s+[\"']react[\"']|require\([\"']react[\"']\)", text)
        )
        is_angular = bool(
            re.search(r"@(?:Component|Injectable|NgModule|Directive|Pipe)\b|from\s+[\"']@angular", text)
        )

        for i, line in enumerate(lines, start=1):
            for module_name in re.findall(r"^\s*import\s+[^;]*?\s+from\s+[\"']([^\"']+)[\"']", line):
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language=language,
                    line_no=i,
                    lines=lines,
                    module_name=module_name,
                    import_kind="import_from",
                    confidence=0.8,
                )
            for module_name in re.findall(r"^\s*import\s+[\"']([^\"']+)[\"']", line):
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language=language,
                    line_no=i,
                    lines=lines,
                    module_name=module_name,
                    import_kind="import_side_effect",
                    confidence=0.8,
                )
            for module_name in re.findall(r"require\s*\(\s*[\"']([^\"']+)[\"']\s*\)", line):
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language=language,
                    line_no=i,
                    lines=lines,
                    module_name=module_name,
                    import_kind="require",
                    confidence=0.78,
                )

        for i, line in enumerate(lines, start=1):
            class_match = self.CLASS_RE.match(line)
            if class_match:
                symbol = class_match.group(1)
                end = _find_block_end(lines, i) if "{" in line else i
                decorators, ann_start = _annotation_block(lines, i, prefix="@")
                metadata: Dict[str, Any] = {}
                if decorators:
                    metadata["decorators"] = decorators
                if is_angular:
                    metadata["framework"] = "angular"
                node_type = "component" if is_react and symbol[:1].isupper() else "class"
                added = _add_node(nodes, seen, file_path, language, node_type, symbol, ann_start, end, lines,
                                  metadata=metadata)
                if added:
                    symbol_count += 1
                continue

            for pattern in (self.FUNCTION_RE, self.ARROW_RE, self.FUNCTION_EXPR_RE):
                fn_match = pattern.match(line)
                if not fn_match:
                    continue

                symbol = fn_match.group(1)
                if symbol in {"if", "for", "while", "switch", "catch"}:
                    continue
                end = _find_block_end(lines, i) if "{" in line else i
                node_type = "component" if is_react and symbol[:1].isupper() else "function"
                metadata = {"framework": "react"} if node_type == "component" else {}
                added = _add_node(nodes, seen, file_path, language, node_type, symbol, i, end, lines,
                                  metadata=metadata)
                if added:
                    symbol_count += 1
                    if symbol.lower() == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language=language,
                            line_no=i,
                            lines=lines,
                            symbol=symbol,
                            entrypoint_kind="main_function",
                            confidence=0.86,
                        )
                break

            route_jsx = re.search(r"<Route[^>]*\bpath=[\"']([^\"']+)[\"']", line)
            route_obj = re.search(r"\bpath\s*:\s*[\"']([^\"']+)[\"']", line)
            route_path = ""
            route_kind = ""
            if route_jsx:
                route_path = route_jsx.group(1)
                route_kind = "react"
            elif route_obj and ("routes" in text.lower() or "RouterModule" in text):
                route_path = route_obj.group(1)
                route_kind = "angular" if is_angular else "router"

            if route_path:
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    language,
                    "route",
                    route_path,
                    i,
                    i,
                    lines,
                    metadata={"framework": route_kind},
                    confidence=0.8,
                )
                if added:
                    symbol_count += 1

            if re.search(r"\b(?:app|server|httpServer)\.listen\s*\(", line):
                _add_entrypoint_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language=language,
                    line_no=i,
                    lines=lines,
                    symbol="listen",
                    entrypoint_kind="server_start",
                    confidence=0.86,
                )
            if re.search(r"\b(?:ReactDOM\.render|createRoot)\s*\(", line):
                _add_entrypoint_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language=language,
                    line_no=i,
                    lines=lines,
                    symbol="ui_bootstrap",
                    entrypoint_kind="ui_bootstrap",
                    confidence=0.82,
                )

        scope_nodes = [n for n in nodes if n.node_type in {"function", "component", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class JavaParser:
    TYPE_RE = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:abstract\s+|final\s+)?(class|interface|enum|record)\s+([A-Za-z_]\w*)"
    )
    METHOD_RE = re.compile(
        r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:<[^>]+>\s*)?[\w\[\]<>?,\s]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{"
    )
    ROUTE_ANN_RE = re.compile(r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\b")

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="java",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="java",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            import_match = re.match(r"^\s*import\s+(?:static\s+)?([A-Za-z0-9_.*]+)\s*;", line)
            if not import_match:
                continue
            _add_import_usage_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="java",
                line_no=i,
                lines=lines,
                module_name=import_match.group(1),
                import_kind="import",
                confidence=0.8,
            )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                _, symbol = type_match.groups()
                anns, ann_start = _annotation_block(lines, i, prefix="@")
                metadata: Dict[str, Any] = {}
                if anns:
                    metadata["annotations"] = anns
                if any("@RestController" in a or "@Controller" in a or "@Service" in a for a in anns):
                    metadata["framework"] = "spring-boot"
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    "java",
                    "class",
                    symbol,
                    ann_start,
                    end,
                    lines,
                    metadata=metadata,
                )
                if added:
                    symbol_count += 1
                continue

            method_match = self.METHOD_RE.match(line)
            if method_match:
                symbol = method_match.group(1)
                if symbol in {"if", "for", "while", "switch", "catch", "return"}:
                    continue

                anns, ann_start = _annotation_block(lines, i, prefix="@")
                end = _find_block_end(lines, i)
                metadata: Dict[str, Any] = {}
                if anns:
                    metadata["annotations"] = anns
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    "java",
                    "function",
                    symbol,
                    ann_start,
                    end,
                    lines,
                    metadata=metadata,
                )
                if added:
                    symbol_count += 1

                if symbol == "main" and " static " in f" {line} ":
                    _add_entrypoint_node(
                        nodes=nodes,
                        seen=seen,
                        file_path=file_path,
                        language="java",
                        line_no=i,
                        lines=lines,
                        symbol="main",
                        entrypoint_kind="jvm_main",
                        confidence=0.88,
                    )

                route_annotations = [a for a in anns if self.ROUTE_ANN_RE.search(a)]
                if route_annotations:
                    path = _extract_path(route_annotations)
                    added = _add_node(
                        nodes,
                        seen,
                        file_path,
                        "java",
                        "route",
                        f"{symbol}:{path or 'unknown'}",
                        ann_start,
                        end,
                        lines,
                        metadata={"framework": "spring-boot", "annotations": route_annotations},
                        confidence=0.88,
                    )
                    if added:
                        symbol_count += 1

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="java",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class KotlinParser:
    TYPE_RE = re.compile(
        r"^\s*(?:(?:public|private|protected|internal|open|abstract|sealed|final|data|enum|annotation)\s+)*"
        r"(class|interface|object|enum\s+class|sealed\s+class|data\s+class)\s+([A-Za-z_]\w*)"
    )
    FUNCTION_RE = re.compile(
        r"^\s*(?:(?:public|private|protected|internal|open|abstract|final|suspend|inline|tailrec|operator|infix|external|override)\s+)*fun\s+([A-Za-z_]\w*)\s*\("
    )
    ROUTE_ANN_RE = re.compile(r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|PatchMapping|RequestMapping)\b")
    KTOR_ROUTE_RE = re.compile(r"\b(?:get|post|put|delete|patch|route)\s*\(\s*[\"']([^\"']+)[\"']")

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="kotlin",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="kotlin",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            import_match = re.match(r"^\s*import\s+([A-Za-z0-9_.*]+)\s*$", line)
            if not import_match:
                continue
            _add_import_usage_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="kotlin",
                line_no=i,
                lines=lines,
                module_name=import_match.group(1),
                import_kind="import",
                confidence=0.8,
            )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                kind, symbol = type_match.groups()
                anns, ann_start = _annotation_block(lines, i, prefix="@")
                metadata: Dict[str, Any] = {"kotlin_kind": kind}
                if anns:
                    metadata["annotations"] = anns
                if any("@RestController" in a or "@Controller" in a or "@Service" in a for a in anns):
                    metadata["framework"] = "spring-boot"
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="kotlin",
                    node_type="class",
                    symbol=symbol,
                    start_line=ann_start,
                    end_line=end,
                    lines=lines,
                    metadata=metadata,
                    confidence=0.9,
                )
                if added:
                    symbol_count += 1
                    if kind.startswith("object") and re.search(r"\:\s*App\b", line):
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="kotlin",
                            line_no=i,
                            lines=lines,
                            symbol=symbol,
                            entrypoint_kind="kotlin_app_object",
                            confidence=0.86,
                        )
                continue

            func_match = self.FUNCTION_RE.match(line)
            if func_match:
                symbol = func_match.group(1)
                anns, ann_start = _annotation_block(lines, i, prefix="@")
                metadata: Dict[str, Any] = {}
                if anns:
                    metadata["annotations"] = anns
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="kotlin",
                    node_type="function",
                    symbol=symbol,
                    start_line=ann_start,
                    end_line=end,
                    lines=lines,
                    metadata=metadata,
                    confidence=0.88,
                )
                if added:
                    symbol_count += 1
                    if symbol == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="kotlin",
                            line_no=i,
                            lines=lines,
                            symbol="main",
                            entrypoint_kind="kotlin_main",
                            confidence=0.88,
                        )

                route_annotations = [a for a in anns if self.ROUTE_ANN_RE.search(a)]
                if route_annotations:
                    path = _extract_path(route_annotations)
                    added = _add_node(
                        nodes=nodes,
                        seen=seen,
                        file_path=file_path,
                        language="kotlin",
                        node_type="route",
                        symbol=f"{symbol}:{path or 'unknown'}",
                        start_line=ann_start,
                        end_line=end,
                        lines=lines,
                        metadata={"framework": "spring-boot", "annotations": route_annotations},
                        confidence=0.88,
                    )
                    if added:
                        symbol_count += 1

            ktor_match = self.KTOR_ROUTE_RE.search(line)
            if ktor_match:
                path = ktor_match.group(1)
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="kotlin",
                    node_type="route",
                    symbol=path,
                    start_line=i,
                    end_line=i,
                    lines=lines,
                    metadata={"framework": "ktor"},
                    confidence=0.8,
                )
                if added:
                    symbol_count += 1

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="kotlin",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class GoParser:
    TYPE_RE = re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(struct|interface)\b")
    FUNCTION_RE = re.compile(r"^\s*func\s*(?:\([^)]+\)\s*)?([A-Za-z_]\w*)\s*\(")
    ROUTE_RE = re.compile(r"\.(GET|POST|PUT|DELETE|PATCH|Handle|HandleFunc)\s*\(\s*\"([^\"]+)\"")

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="go",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="go",
            lines=lines,
        )

        in_import_block = False
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if in_import_block:
                if stripped.startswith(")"):
                    in_import_block = False
                    continue
                module_match = re.match(r"^(?:[A-Za-z_]\w*\s+)?\"([^\"]+)\"", stripped)
                if module_match:
                    _add_import_usage_node(
                        nodes=nodes,
                        seen=seen,
                        file_path=file_path,
                        language="go",
                        line_no=i,
                        lines=lines,
                        module_name=module_match.group(1),
                        import_kind="import_block",
                        confidence=0.8,
                    )
                continue

            if re.match(r"^\s*import\s*\(\s*$", line):
                in_import_block = True
                continue

            module_match = re.match(r"^\s*import\s+(?:[A-Za-z_]\w*\s+)?\"([^\"]+)\"", line)
            if module_match:
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="go",
                    line_no=i,
                    lines=lines,
                    module_name=module_match.group(1),
                    import_kind="import_single",
                    confidence=0.8,
                )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                symbol, kind = type_match.groups()
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="go",
                    node_type="class",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"go_kind": kind},
                    confidence=0.86,
                )
                if added:
                    symbol_count += 1
                continue

            func_match = self.FUNCTION_RE.match(line)
            if func_match:
                symbol = func_match.group(1)
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="go",
                    node_type="function",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={},
                    confidence=0.88,
                )
                if added:
                    symbol_count += 1
                    if symbol == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="go",
                            line_no=i,
                            lines=lines,
                            symbol="main",
                            entrypoint_kind="go_main",
                            confidence=0.9,
                        )

            route_match = self.ROUTE_RE.search(line)
            if route_match:
                method, path = route_match.groups()
                route_symbol = f"{method.upper()}:{path}"
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="go",
                    node_type="route",
                    symbol=route_symbol,
                    start_line=i,
                    end_line=i,
                    lines=lines,
                    metadata={"framework": "go-router"},
                    confidence=0.8,
                )
                if added:
                    symbol_count += 1

            if re.search(r"\b(?:http\.)?ListenAndServe\s*\(", line):
                _add_entrypoint_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="go",
                    line_no=i,
                    lines=lines,
                    symbol="ListenAndServe",
                    entrypoint_kind="server_start",
                    confidence=0.86,
                )

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="go",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class CSharpParser:
    TYPE_RE = re.compile(
        r"^\s*(?:(?:public|private|protected|internal|sealed|abstract|partial|static)\s+)*(class|interface|struct|record)\s+([A-Za-z_]\w*)"
    )
    METHOD_RE = re.compile(
        r"^\s*(?:(?:public|private|protected|internal|static|virtual|override|async|sealed|new|partial)\s+)+[\w<>,\[\]\.\?]+\s+([A-Za-z_]\w*)\s*\([^;]*\)\s*(?:where\s+[^{]+)?\{"
    )
    ROUTE_ATTR_RE = re.compile(r"\[(HttpGet|HttpPost|HttpPut|HttpDelete|HttpPatch|Route)\b")

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="csharp",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="csharp",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            using_match = re.match(r"^\s*using\s+([A-Za-z0-9_.]+)\s*;", line)
            if not using_match:
                continue
            _add_import_usage_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="csharp",
                line_no=i,
                lines=lines,
                module_name=using_match.group(1),
                import_kind="using",
                confidence=0.8,
            )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                _, symbol = type_match.groups()
                attrs, attr_start = _annotation_block(lines, i, prefix="[")
                metadata: Dict[str, Any] = {}
                if attrs:
                    metadata["attributes"] = attrs
                if any("ApiController" in a or "Route" in a for a in attrs):
                    metadata["framework"] = "dotnet"
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    "csharp",
                    "class",
                    symbol,
                    attr_start,
                    end,
                    lines,
                    metadata=metadata,
                )
                if added:
                    symbol_count += 1
                continue

            method_match = self.METHOD_RE.match(line)
            if method_match:
                symbol = method_match.group(1)
                if symbol in {"if", "for", "while", "switch", "catch", "return"}:
                    continue

                attrs, attr_start = _annotation_block(lines, i, prefix="[")
                end = _find_block_end(lines, i)
                metadata: Dict[str, Any] = {}
                if attrs:
                    metadata["attributes"] = attrs
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    "csharp",
                    "function",
                    symbol,
                    attr_start,
                    end,
                    lines,
                    metadata=metadata,
                )
                if added:
                    symbol_count += 1

                if symbol in {"Main", "main"} and " static " in f" {line} ":
                    _add_entrypoint_node(
                        nodes=nodes,
                        seen=seen,
                        file_path=file_path,
                        language="csharp",
                        line_no=i,
                        lines=lines,
                        symbol="Main",
                        entrypoint_kind="dotnet_main",
                        confidence=0.9,
                    )

                route_attrs = [a for a in attrs if self.ROUTE_ATTR_RE.search(a)]
                if route_attrs:
                    path = _extract_path(route_attrs)
                    added = _add_node(
                        nodes,
                        seen,
                        file_path,
                        "csharp",
                        "route",
                        f"{symbol}:{path or 'unknown'}",
                        attr_start,
                        end,
                        lines,
                        metadata={"framework": "dotnet", "attributes": route_attrs},
                        confidence=0.88,
                    )
                    if added:
                        symbol_count += 1

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="csharp",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class CppParser:
    TYPE_RE = re.compile(r"^\s*(?:template\s*<[^>]+>\s*)?(class|struct)\s+([A-Za-z_]\w*)")
    FUNCTION_RE = re.compile(
        r"^\s*(?:template\s*<[^>]+>\s*)?(?:(?:inline|constexpr|static|virtual|friend|extern)\s+)*(?:[\w:\<\>\~\*&]+\s+)+([A-Za-z_~]\w*(?:::[A-Za-z_~]\w*)?)\s*\([^;{}]*\)\s*(?:const)?\s*(?:noexcept)?\s*(?:override)?\s*(?:final)?\s*(\{|;)"
    )

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str, language: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language=language,
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            include_match = re.match(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", line)
            if not include_match:
                continue
            _add_import_usage_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language=language,
                line_no=i,
                lines=lines,
                module_name=include_match.group(1),
                import_kind="include",
                confidence=0.78,
            )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                _, symbol = type_match.groups()
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(nodes, seen, file_path, language, "class", symbol, i, end, lines)
                if added:
                    symbol_count += 1
                continue

            func_match = self.FUNCTION_RE.match(line)
            if func_match:
                symbol = func_match.group(1)
                terminator = func_match.group(2)
                if symbol in {"if", "for", "while", "switch", "catch", "return"}:
                    continue
                end = _find_block_end(lines, i) if terminator == "{" else i
                metadata = {"declaration": terminator == ";"}
                confidence = 0.8 if terminator == "{" else 0.65
                added = _add_node(
                    nodes,
                    seen,
                    file_path,
                    language,
                    "function",
                    symbol,
                    i,
                    end,
                    lines,
                    metadata=metadata,
                    confidence=confidence,
                )
                if added:
                    symbol_count += 1
                    if symbol.split("::")[-1] == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language=language,
                            line_no=i,
                            lines=lines,
                            symbol=symbol,
                            entrypoint_kind="cpp_main",
                            confidence=0.9,
                        )

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=language,
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class ScalaParser:
    TYPE_RE = re.compile(
        r"^\s*(?:sealed\s+|final\s+|abstract\s+|case\s+|private\s+|protected\s+)*"
        r"(class|object|trait)\s+([A-Za-z_]\w*)"
    )
    DEF_RE = re.compile(
        r"^\s*(?:override\s+|private\s+|protected\s+|implicit\s+|final\s+)*def\s+([A-Za-z_]\w*)\b"
    )

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="scala",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_slash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="scala",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            import_match = re.match(r"^\s*import\s+([A-Za-z0-9_.*{} ,]+)\s*$", line)
            if not import_match:
                continue
            _add_import_usage_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="scala",
                line_no=i,
                lines=lines,
                module_name=import_match.group(1).strip(),
                import_kind="import",
                confidence=0.8,
            )

        for i, line in enumerate(lines, start=1):
            type_match = self.TYPE_RE.match(line)
            if type_match:
                kind, symbol = type_match.groups()
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="scala",
                    node_type="class",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"scala_kind": kind},
                    confidence=0.9,
                )
                if added:
                    symbol_count += 1
                    if kind == "object" and "extends App" in line:
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="scala",
                            line_no=i,
                            lines=lines,
                            symbol=symbol,
                            entrypoint_kind="scala_app_object",
                            confidence=0.86,
                        )
                continue

            def_match = self.DEF_RE.match(line)
            if def_match:
                symbol = def_match.group(1)
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="scala",
                    node_type="function",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={},
                    confidence=0.88,
                )
                if added:
                    symbol_count += 1
                    if symbol == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="scala",
                            line_no=i,
                            lines=lines,
                            symbol="main",
                            entrypoint_kind="scala_main",
                            confidence=0.9,
                        )

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="scala",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class RParser:
    FUNCTION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?:<-|=)\s*function\s*\(")
    CLASS_RE = re.compile(r"setClass\(\s*[\"']([A-Za-z_]\w*)[\"']")

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="r",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0

        _add_hash_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="r",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            for module_name in re.findall(r"\b(?:library|require)\s*\(\s*[\"']?([A-Za-z0-9_.]+)[\"']?\s*\)", line):
                _add_import_usage_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="r",
                    line_no=i,
                    lines=lines,
                    module_name=module_name,
                    import_kind="library",
                    confidence=0.78,
                )

        for i, line in enumerate(lines, start=1):
            func_match = self.FUNCTION_RE.match(line)
            if func_match:
                symbol = func_match.group(1)
                end = _find_block_end(lines, i) if "{" in line else i
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="r",
                    node_type="function",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={},
                    confidence=0.88,
                )
                if added:
                    symbol_count += 1
                    if symbol == "main":
                        _add_entrypoint_node(
                            nodes=nodes,
                            seen=seen,
                            file_path=file_path,
                            language="r",
                            line_no=i,
                            lines=lines,
                            symbol="main",
                            entrypoint_kind="r_main_function",
                            confidence=0.82,
                        )
                continue

            class_match = self.CLASS_RE.search(line)
            if class_match:
                symbol = class_match.group(1)
                end = _find_statement_end(lines, i)
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="r",
                    node_type="class",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"r_kind": "s4"},
                    confidence=0.82,
                )
                if added:
                    symbol_count += 1

        scope_nodes = [n for n in nodes if n.node_type in {"function", "class"} and n.symbol]
        _add_scoped_call_edges(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="r",
            lines=lines,
            scope_nodes=scope_nodes,
            max_callees_per_scope=self.max_callees_per_scope,
        )

        return nodes, _confidence(symbol_count)


class CobolParser:
    PROGRAM_ID_RE = re.compile(r"^\s*PROGRAM-ID\.\s*([A-Za-z0-9_-]+)\.?", flags=re.IGNORECASE)
    SECTION_RE = re.compile(r"^\s*([A-Za-z0-9-]+)\s+SECTION\.\s*$", flags=re.IGNORECASE)
    PARAGRAPH_RE = re.compile(r"^\s*([A-Za-z0-9-]+)\.\s*$")
    CALL_RE = re.compile(r"\bCALL\s+['\"]?([A-Za-z0-9_-]+)", flags=re.IGNORECASE)
    RESERVED_LABELS = {
        "identification",
        "environment",
        "data",
        "procedure",
        "working-storage",
        "linkage",
        "file",
        "configuration",
        "input-output",
    }

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="cobol",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0
        active_scope = "program"

        _add_cobol_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="cobol",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            prog_match = self.PROGRAM_ID_RE.match(line)
            if prog_match:
                symbol = prog_match.group(1)
                active_scope = symbol
                end = min(len(lines), i + 1)
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="cobol",
                    node_type="class",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"cobol_kind": "program"},
                    confidence=0.84,
                )
                if added:
                    symbol_count += 1
                continue

            section_match = self.SECTION_RE.match(line)
            if section_match:
                symbol = section_match.group(1)
                active_scope = symbol
                end = min(len(lines), i + 1)
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="cobol",
                    node_type="function",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"cobol_kind": "section"},
                    confidence=0.78,
                )
                if added:
                    symbol_count += 1
                continue

            paragraph_match = self.PARAGRAPH_RE.match(line)
            if paragraph_match:
                symbol = paragraph_match.group(1)
                if symbol.lower() not in self.RESERVED_LABELS:
                    active_scope = symbol
                    end = min(len(lines), i + 1)
                    added = _add_node(
                        nodes=nodes,
                        seen=seen,
                        file_path=file_path,
                        language="cobol",
                        node_type="function",
                        symbol=symbol,
                        start_line=i,
                        end_line=end,
                        lines=lines,
                        metadata={"cobol_kind": "paragraph"},
                        confidence=0.72,
                    )
                    if added:
                        symbol_count += 1

            for callee in self.CALL_RE.findall(line):
                _add_call_edge_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="cobol",
                    caller=active_scope,
                    callee=callee,
                    line_no=i,
                    lines=lines,
                    confidence=0.76,
                )

        return nodes, _confidence(symbol_count)


class SqlParser:
    CREATE_TABLE_RE = re.compile(
        r"^\s*create\s+(?:or\s+replace\s+)?table\s+([A-Za-z0-9_.$\"`\[\]]+)",
        flags=re.IGNORECASE,
    )
    CREATE_VIEW_RE = re.compile(
        r"^\s*create\s+(?:or\s+replace\s+)?view\s+([A-Za-z0-9_.$\"`\[\]]+)",
        flags=re.IGNORECASE,
    )
    CREATE_PROC_RE = re.compile(
        r"^\s*create\s+(?:or\s+replace\s+)?(?:procedure|proc|function)\s+([A-Za-z0-9_.$\"`\[\]]+)",
        flags=re.IGNORECASE,
    )
    QUERY_RE = re.compile(
        r"^\s*(select\b|with\b|insert\s+into\b|update\b|delete\s+from\b|merge\s+into\b)",
        flags=re.IGNORECASE,
    )
    TABLE_REF_RE = re.compile(
        r"\b(?:from|join|into|update|merge\s+into)\s+([A-Za-z0-9_.$\"`\[\]]+)",
        flags=re.IGNORECASE,
    )
    PROC_CALL_RE = re.compile(
        r"\b(?:call|exec(?:ute)?)\s+([A-Za-z0-9_.$\"`\[\]]+)",
        flags=re.IGNORECASE,
    )

    def __init__(self, node_text_max_chars: int, max_callees_per_scope: int):
        self.node_text_max_chars = node_text_max_chars
        self.max_callees_per_scope = max(1, max_callees_per_scope)

    def parse(self, file_path: str, text: str) -> Tuple[List[CodeNode], float]:
        lines = text.splitlines()
        nodes: List[CodeNode] = [_file_node(
            file_path=file_path,
            language="sql",
            lines=lines,
            text=text,
            node_text_max_chars=self.node_text_max_chars,
        )]
        seen = {nodes[0].node_id}
        symbol_count = 0
        scope_nodes: List[CodeNode] = []

        _add_sql_comment_nodes(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="sql",
            lines=lines,
        )

        for i, line in enumerate(lines, start=1):
            table_match = self.CREATE_TABLE_RE.match(line)
            if table_match:
                symbol = _clean_sql_identifier(table_match.group(1))
                end = _find_statement_end(lines, i)
                if _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    node_type="table",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"sql_kind": "create_table"},
                    confidence=0.9,
                ):
                    symbol_count += 1
                continue

            view_match = self.CREATE_VIEW_RE.match(line)
            if view_match:
                symbol = _clean_sql_identifier(view_match.group(1))
                end = _find_statement_end(lines, i)
                if _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    node_type="view",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"sql_kind": "create_view"},
                    confidence=0.9,
                ):
                    symbol_count += 1
                continue

            proc_match = self.CREATE_PROC_RE.match(line)
            if proc_match:
                symbol = _clean_sql_identifier(proc_match.group(1))
                end = _find_statement_end(lines, i)
                if _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    node_type="function",
                    symbol=symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"sql_kind": "routine"},
                    confidence=0.9,
                ):
                    symbol_count += 1
                continue

            query_match = self.QUERY_RE.match(line)
            if query_match:
                op = query_match.group(1).split()[0].upper()
                end = _find_statement_end(lines, i)
                query_symbol = f"{op}@{i}"
                if _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    node_type="query",
                    symbol=query_symbol,
                    start_line=i,
                    end_line=end,
                    lines=lines,
                    metadata={"sql_op": op},
                    confidence=0.82,
                ):
                    symbol_count += 1

        for node in nodes:
            if node.node_type == "function" and node.symbol:
                scope_nodes.append(node)

        for i, line in enumerate(lines, start=1):
            owner = None
            for scope in scope_nodes:
                if scope.start_line <= i <= scope.end_line:
                    owner = scope.symbol
                    break
            caller = owner or f"query@{i}"

            for ref in self.TABLE_REF_RE.findall(line):
                callee = _clean_sql_identifier(ref)
                if not callee:
                    continue
                _add_call_edge_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    caller=caller,
                    callee=callee,
                    line_no=i,
                    lines=lines,
                    confidence=0.74,
                )

            for ref in self.PROC_CALL_RE.findall(line):
                callee = _clean_sql_identifier(ref)
                if not callee:
                    continue
                _add_call_edge_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="sql",
                    caller=caller,
                    callee=callee,
                    line_no=i,
                    lines=lines,
                    confidence=0.76,
                )

        return nodes, _confidence(symbol_count)


def _notebook_source_text(source: Any) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source or "")


def _notebook_cell_language(cell: Dict[str, Any], notebook_language: str) -> str:
    meta = cell.get("metadata")
    if isinstance(meta, dict):
        lang = meta.get("language")
        if isinstance(lang, str) and lang.strip():
            return lang.strip().lower()
    return notebook_language


def parse_notebook(
    file_path: str,
    text: str,
    node_text_max_chars: int,
    max_callees_per_scope: int,
    doc_chunk_lines: int,
    doc_overlap: int,
) -> Tuple[List[CodeNode], float]:
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        lines = text.splitlines()
        node = _file_node(
            file_path=file_path,
            language="notebook",
            lines=lines,
            text=text,
            node_text_max_chars=node_text_max_chars,
            metadata={"parse_error": "invalid_ipynb", "parse_error_detail": str(exc)[:240]},
            confidence=0.2,
        )
        return [node], 0.2

    if not isinstance(payload, dict):
        lines = text.splitlines()
        node = _file_node(
            file_path=file_path,
            language="notebook",
            lines=lines,
            text=text,
            node_text_max_chars=node_text_max_chars,
            metadata={"parse_error": "invalid_ipynb_root"},
            confidence=0.2,
        )
        return [node], 0.2

    nb_meta = payload.get("metadata")
    notebook_language = "python"
    if isinstance(nb_meta, dict):
        lang_info = nb_meta.get("language_info")
        if isinstance(lang_info, dict):
            lang_name = lang_info.get("name")
            if isinstance(lang_name, str) and lang_name.strip():
                notebook_language = lang_name.strip().lower()

    raw_cells = payload.get("cells")
    cells = raw_cells if isinstance(raw_cells, list) else []
    flat_lines: List[str] = []
    cell_infos: List[Dict[str, Any]] = []

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue

        cell_type = str(cell.get("cell_type", "unknown")).strip().lower()
        source_text = _notebook_source_text(cell.get("source", ""))
        cell_lines = source_text.splitlines()
        if not cell_lines:
            cell_lines = [""]

        start_line = len(flat_lines) + 1
        flat_lines.extend(cell_lines)
        end_line = len(flat_lines)

        cell_infos.append({
            "cell_index": idx,
            "cell_type": cell_type,
            "language": _notebook_cell_language(cell, notebook_language),
            "start_line": start_line,
            "end_line": end_line,
            "lines": cell_lines,
            "execution_count": cell.get("execution_count"),
        })
        flat_lines.append("")

    if flat_lines and flat_lines[-1] == "":
        flat_lines.pop()
    if not flat_lines:
        flat_lines = [""]

    file_text = "\n".join(flat_lines)
    nodes: List[CodeNode] = [_file_node(
        file_path=file_path,
        language="notebook",
        lines=flat_lines,
        text=file_text,
        node_text_max_chars=node_text_max_chars,
        metadata={"notebook_cells": len(cell_infos), "notebook_language": notebook_language},
        confidence=0.92 if cell_infos else 0.7,
    )]
    seen = {nodes[0].node_id}
    symbol_count = 0

    for info in cell_infos:
        start = int(info["start_line"])
        end = int(info["end_line"])
        cell_index = int(info["cell_index"])
        cell_type = str(info["cell_type"])
        cell_language = str(info["language"])
        cell_lines = info["lines"]
        execution_count = info.get("execution_count")

        if cell_type == "markdown":
            ranges = _chunk_ranges(start, end, chunk_lines=doc_chunk_lines, overlap=doc_overlap)
            for chunk_start, chunk_end in ranges:
                _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="markdown",
                    node_type="doc",
                    symbol=f"cell_{cell_index}",
                    start_line=chunk_start,
                    end_line=chunk_end,
                    lines=flat_lines,
                    metadata={
                        "doc_format": "notebook_markdown",
                        "cell_index": cell_index,
                    },
                    confidence=0.9,
                )
            continue

        if cell_type != "code":
            _add_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language="text",
                node_type="doc",
                symbol=f"cell_{cell_index}",
                start_line=start,
                end_line=end,
                lines=flat_lines,
                metadata={"doc_format": f"notebook_{cell_type}", "cell_index": cell_index},
                confidence=0.75,
            )
            continue

        parse_error_detail = None
        py_tree: Optional[ast.AST] = None
        cell_text = "\n".join(cell_lines)
        if cell_language == "python":
            try:
                py_tree = ast.parse(cell_text)
            except SyntaxError as exc:
                parse_error_detail = str(exc)[:240]

        code_meta: Dict[str, Any] = {
            "cell_type": "code",
            "cell_index": cell_index,
            "execution_count": execution_count,
            "code_language": cell_language,
        }
        if parse_error_detail:
            code_meta["parse_error"] = "syntax"
            code_meta["parse_error_detail"] = parse_error_detail

        _add_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language=cell_language,
            node_type="notebook_code",
            symbol=f"cell_{cell_index}",
            start_line=start,
            end_line=end,
            lines=flat_lines,
            metadata=code_meta,
            confidence=0.78 if parse_error_detail else 0.88,
        )

        local = 0
        while local < len(cell_lines):
            if not cell_lines[local].strip().startswith("#"):
                local += 1
                continue
            block_start = local
            while local < len(cell_lines) and cell_lines[local].strip().startswith("#"):
                local += 1
            _add_comment_node(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                language=cell_language,
                start_line=start + block_start,
                end_line=start + local - 1,
                lines=flat_lines,
                kind="line_block",
                metadata={"cell_index": cell_index},
            )

        if cell_language != "python" or py_tree is None:
            continue

        for node in ast.walk(py_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                local_start = getattr(node, "lineno", 1)
                local_end = getattr(node, "end_lineno", local_start)
                node_type = "class" if isinstance(node, ast.ClassDef) else "function"
                added = _add_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    node_type=node_type,
                    symbol=node.name,
                    start_line=start + local_start - 1,
                    end_line=start + local_end - 1,
                    lines=flat_lines,
                    metadata={"cell_index": cell_index, "source": "ipynb"},
                    confidence=0.95,
                )
                if added:
                    symbol_count += 1

        by_caller: Dict[str, Dict[str, int]] = {}
        for caller, callee, local_line in _iter_python_call_edges(py_tree):
            if not caller or not callee:
                continue
            caller_edges = by_caller.setdefault(caller, {})
            if callee in caller_edges:
                continue
            if len(caller_edges) >= max(1, max_callees_per_scope):
                continue
            caller_edges[callee] = local_line

        for caller, callees in by_caller.items():
            for callee, local_line in callees.items():
                _add_call_edge_node(
                    nodes=nodes,
                    seen=seen,
                    file_path=file_path,
                    language="python",
                    caller=caller,
                    callee=callee,
                    line_no=start + local_line - 1,
                    lines=flat_lines,
                    confidence=0.8,
                )

    conf = 0.92 if symbol_count > 0 else (0.86 if cell_infos else 0.6)
    return nodes, conf


def parse_markdown(file_path: str, text: str, chunk_lines: int, overlap: int) -> Tuple[List[CodeNode], float]:
    lines = text.splitlines()
    if not lines:
        return [], 0.0

    nodes: List[CodeNode] = []
    seen: set = set()
    headings = _markdown_headings(lines)
    doc_title = _markdown_doc_title(file_path, headings)

    if headings and headings[0]["line"] > 1:
        _append_markdown_doc_chunks(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            lines=lines,
            chunk_lines=chunk_lines,
            overlap=overlap,
            start_line=1,
            end_line=headings[0]["line"] - 1,
            doc_title=doc_title,
            section_title="Preamble",
            section_path="Preamble",
            section_level=0,
        )

    if headings:
        for idx, heading in enumerate(headings):
            start = heading["line"]
            end = headings[idx + 1]["line"] - 1 if idx + 1 < len(headings) else len(lines)
            _append_markdown_doc_chunks(
                nodes=nodes,
                seen=seen,
                file_path=file_path,
                lines=lines,
                chunk_lines=chunk_lines,
                overlap=overlap,
                start_line=start,
                end_line=end,
                doc_title=doc_title,
                section_title=str(heading["title"]),
                section_path=str(heading["path"]),
                section_level=int(heading["level"]),
            )
    else:
        _append_markdown_doc_chunks(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            lines=lines,
            chunk_lines=chunk_lines,
            overlap=overlap,
            start_line=1,
            end_line=len(lines),
            doc_title=doc_title,
            section_title="Document",
            section_path="Document",
            section_level=0,
        )

    current_heading_idx = -1
    fence_open: Optional[Dict[str, Any]] = None

    for line_no, line in enumerate(lines, start=1):
        while current_heading_idx + 1 < len(headings) and headings[current_heading_idx + 1]["line"] <= line_no:
            current_heading_idx += 1
        current_heading = headings[current_heading_idx] if current_heading_idx >= 0 else None

        match = _MD_FENCE_RE.match(line)
        if not match:
            continue

        fence_marker = match.group(1)
        lang_hint = (match.group(2) or "").strip().lower() or "text"

        if fence_open is None:
            fence_open = {
                "start_line": line_no,
                "marker": fence_marker,
                "lang": lang_hint,
                "heading": current_heading,
            }
            continue

        if fence_open["marker"] != fence_marker:
            continue

        heading = fence_open.get("heading")
        section_title = str(heading["title"]) if heading else "Preamble"
        section_path = str(heading["path"]) if heading else "Preamble"
        section_level = int(heading["level"]) if heading else 0

        _add_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="markdown",
            node_type="doc_code",
            symbol=fence_open["lang"],
            start_line=int(fence_open["start_line"]),
            end_line=line_no,
            lines=lines,
            metadata={
                "doc_format": "markdown",
                "doc_title": doc_title,
                "section_title": section_title,
                "section_path": section_path,
                "section_level": section_level,
                "code_language": fence_open["lang"],
            },
            confidence=0.9,
        )
        fence_open = None

    if fence_open is not None:
        heading = fence_open.get("heading")
        section_title = str(heading["title"]) if heading else "Preamble"
        section_path = str(heading["path"]) if heading else "Preamble"
        section_level = int(heading["level"]) if heading else 0
        _add_node(
            nodes=nodes,
            seen=seen,
            file_path=file_path,
            language="markdown",
            node_type="doc_code",
            symbol=fence_open["lang"],
            start_line=int(fence_open["start_line"]),
            end_line=len(lines),
            lines=lines,
            metadata={
                "doc_format": "markdown",
                "doc_title": doc_title,
                "section_title": section_title,
                "section_path": section_path,
                "section_level": section_level,
                "code_language": fence_open["lang"],
                "fence_closed": False,
            },
            confidence=0.86,
        )

    conf = 0.93 if headings else 0.88
    return nodes, conf


def parse_docs(file_path: str, text: str, chunk_lines: int, overlap: int) -> Tuple[List[CodeNode], float]:
    lines = text.splitlines()
    if not lines:
        return [], 0.0
    nodes: List[CodeNode] = []
    i = 1
    while i <= len(lines):
        start = i
        end = min(len(lines), i + chunk_lines - 1)
        snippet = "\n".join(lines[start - 1:end])
        nodes.append(CodeNode(
            node_id=f"doc::{file_path}::{start}-{end}",
            node_type="doc",
            language="text",
            file_path=file_path,
            start_line=start,
            end_line=end,
            symbol=None,
            text=snippet,
            metadata={},
            confidence=0.9,
        ).finalize())
        i = end - overlap + 1
        if i <= start:
            i = end + 1
    return nodes, 0.9


def parse_config(file_path: str, text: str, chunk_lines: int, overlap: int) -> Tuple[List[CodeNode], float]:
    lines = text.splitlines()
    if not lines:
        return [], 0.0
    nodes: List[CodeNode] = []
    i = 1
    while i <= len(lines):
        start = i
        end = min(len(lines), i + chunk_lines - 1)
        snippet = "\n".join(lines[start - 1:end])
        nodes.append(CodeNode(
            node_id=f"config::{file_path}::{start}-{end}",
            node_type="config",
            language="config",
            file_path=file_path,
            start_line=start,
            end_line=end,
            symbol=None,
            text=snippet,
            metadata={},
            confidence=0.85,
        ).finalize())
        i = end - overlap + 1
        if i <= start:
            i = end + 1
    return nodes, 0.85
