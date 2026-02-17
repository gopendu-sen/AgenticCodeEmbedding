from typing import Dict, List, Tuple

from agentic_rag.code_parser.deterministic import (
    CobolParser,
    CSharpParser,
    CppParser,
    GoParser,
    JavaParser,
    JsTsParser,
    KotlinParser,
    PythonASTParser,
    RParser,
    ScalaParser,
    SqlParser,
    language_hint_from_ext,
    parse_config,
    parse_docs,
    parse_markdown,
    parse_notebook,
)
from agentic_rag.code_parser.types import CodeNode
from agentic_rag.core.config import IOLimitsConfig, ParserConfig


class CodeParserService:
    def __init__(self, parser_config: ParserConfig, io_limits: IOLimitsConfig):
        self.parser_config = parser_config
        self.node_text_max_chars = io_limits.node_text_max_chars
        self.max_callees_per_scope = parser_config.call_graph.max_callees_per_scope

        self.py_parser = PythonASTParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.js_ts_parser = JsTsParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.java_parser = JavaParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.kotlin_parser = KotlinParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.scala_parser = ScalaParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.csharp_parser = CSharpParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.cpp_parser = CppParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.r_parser = RParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.go_parser = GoParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.cobol_parser = CobolParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )
        self.sql_parser = SqlParser(
            node_text_max_chars=self.node_text_max_chars,
            max_callees_per_scope=self.max_callees_per_scope,
        )

    def _chunked_doc_fallback(
        self,
        rel_path: str,
        text: str,
        lang: str,
        metadata: Dict[str, str],
        confidence: float,
    ) -> Tuple[List[CodeNode], float, str]:
        nodes, _ = parse_docs(
            rel_path,
            text,
            chunk_lines=self.parser_config.docs.chunk_lines,
            overlap=self.parser_config.docs.overlap_lines,
        )
        if not nodes:
            lines = text.splitlines()
            node = CodeNode(
                node_id=f"doc::{rel_path}::1-{max(1, len(lines))}",
                node_type="doc",
                language=lang if lang != "unknown" else "text",
                file_path=rel_path,
                start_line=1,
                end_line=max(1, len(lines)),
                symbol=None,
                text=text[: self.node_text_max_chars],
                metadata=dict(metadata),
                confidence=confidence,
            ).finalize()
            return [node], confidence, lang

        for node in nodes:
            merged_meta = dict(node.metadata or {})
            merged_meta.update(metadata)
            node.metadata = merged_meta
            if lang != "unknown":
                node.language = lang
            node.confidence = confidence

        return nodes, confidence, lang

    def _parser_error_fallback(self, rel_path: str, text: str, lang: str, parser_name: str, error: Exception) -> Tuple[
        List[CodeNode], float, str]:
        fallback_confidence = min(self.parser_config.generic.fallback_confidence, 0.1)
        return self._chunked_doc_fallback(
            rel_path=rel_path,
            text=text,
            lang=lang,
            metadata={
                "parse_error": "exception",
                "parser": parser_name,
                "parse_error_detail": str(error)[:240],
                "fallback_mode": "parser_error_chunked_docs",
            },
            confidence=fallback_confidence,
        )

    def parse_file(self, rel_path: str, text: str, ext: str) -> Tuple[List[CodeNode], float, str]:
        lang = language_hint_from_ext(ext)

        try:
            if lang == "python":
                nodes, conf = self.py_parser.parse(rel_path, text)
                return nodes, conf, lang

            if ext in (".js", ".jsx", ".ts", ".tsx"):
                nodes, conf = self.js_ts_parser.parse(rel_path, text, ext)
                return nodes, conf, lang

            if lang == "java":
                nodes, conf = self.java_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "kotlin":
                nodes, conf = self.kotlin_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "scala":
                nodes, conf = self.scala_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "csharp":
                nodes, conf = self.csharp_parser.parse(rel_path, text)
                return nodes, conf, lang

            if ext in (".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"):
                nodes, conf = self.cpp_parser.parse(rel_path, text, language=lang)
                return nodes, conf, lang

            if lang == "r":
                nodes, conf = self.r_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "go":
                nodes, conf = self.go_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "cobol":
                nodes, conf = self.cobol_parser.parse(rel_path, text)
                return nodes, conf, lang

            if lang == "sql":
                nodes, conf = self.sql_parser.parse(rel_path, text)
                return nodes, conf, lang

            if ext == ".md":
                nodes, conf = parse_markdown(
                    rel_path,
                    text,
                    chunk_lines=self.parser_config.markdown.chunk_lines,
                    overlap=self.parser_config.markdown.overlap_lines,
                )
                return nodes, conf, lang

            if ext in (".txt", ".html", ".htm", ".xhtml", ".jsp", ".jspx", ".cshtml"):
                nodes, conf = parse_docs(
                    rel_path,
                    text,
                    chunk_lines=self.parser_config.docs.chunk_lines,
                    overlap=self.parser_config.docs.overlap_lines,
                )
                return nodes, conf, lang

            if ext == ".ipynb":
                nodes, conf = parse_notebook(
                    file_path=rel_path,
                    text=text,
                    node_text_max_chars=self.node_text_max_chars,
                    max_callees_per_scope=self.max_callees_per_scope,
                    doc_chunk_lines=self.parser_config.docs.chunk_lines,
                    doc_overlap=self.parser_config.docs.overlap_lines,
                )
                return nodes, conf, lang

            if ext in (".yml", ".yaml", ".json", ".toml", ".ini", ".xml", ".properties"):
                nodes, conf = parse_config(
                    rel_path,
                    text,
                    chunk_lines=self.parser_config.config.chunk_lines,
                    overlap=self.parser_config.config.overlap_lines,
                )
                return nodes, conf, lang

            fallback_confidence = self.parser_config.generic.fallback_confidence
            return self._chunked_doc_fallback(
                rel_path=rel_path,
                text=text,
                lang=lang,
                metadata={
                    "fallback_mode": "unknown_extension_chunked_docs",
                    "note": "no deterministic parser",
                },
                confidence=fallback_confidence,
            )
        except Exception as exc:  # noqa: BLE001
            parser_name = f"{lang or 'unknown'}:{ext or 'unknown'}"
            return self._parser_error_fallback(
                rel_path=rel_path,
                text=text,
                lang=lang,
                parser_name=parser_name,
                error=exc,
            )
