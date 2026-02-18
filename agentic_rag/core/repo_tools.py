import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class RepoFile:
    rel_path: str
    ext: str
    size: int


class RepoTools:
    def __init__(
        self,
        repo_path: str,
        exclude_dirs: List[str],
        max_file_size: int,
        read_file_max_chars: int,
        repo_tree_max_files: int,
        tool_search_max_hits: int,
    ):
        self.repo_path = os.path.abspath(repo_path)
        self.exclude_dirs = set(exclude_dirs)
        self.max_file_size = max_file_size
        self.read_file_max_chars = read_file_max_chars
        self.repo_tree_max_files = repo_tree_max_files
        self.tool_search_max_hits = tool_search_max_hits

    def _abs(self, rel_path: str) -> str:
        return os.path.join(self.repo_path, rel_path)

    def iter_files(self, include_exts: List[str]) -> Iterator[RepoFile]:
        for root, dirs, files in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for fn in files:
                abs_path = os.path.join(root, fn)
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    continue
                if size > self.max_file_size:
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext not in include_exts and fn != ".env":
                    continue
                rel = os.path.relpath(abs_path, self.repo_path)
                yield RepoFile(rel_path=rel, ext=ext, size=size)

    def read_file(self, rel_path: str, max_chars: Optional[int] = None) -> str:
        limit = self.read_file_max_chars if max_chars is None else max_chars
        with open(self._abs(rel_path), "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)

    def read_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        max_line_span: Optional[int] = None,
    ) -> Dict[str, Any]:
        text = self.read_file(file_path)
        lines = text.splitlines()
        s = max(1, start_line)
        e = min(len(lines), end_line)
        requested_end = e
        if max_line_span is not None and int(max_line_span) > 0:
            span = max(1, int(max_line_span))
            if e - s + 1 > span:
                e = min(len(lines), s + span - 1)
        snippet = "\n".join(lines[s - 1:e])
        return {
            "file_path": file_path,
            "start_line": s,
            "end_line": e,
            "text": snippet,
            "requested_end_line": requested_end,
            "line_span_capped": bool(e < requested_end),
        }

    def search_in_file(self, file_path: str, pattern: str, max_hits: Optional[int] = None) -> Dict[str, Any]:
        rx = re.compile(pattern)
        text = self.read_file(file_path)
        hits = []
        hit_limit = self.tool_search_max_hits if max_hits is None else max_hits
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append({"line": i, "text": line.strip()[:300]})
                if len(hits) >= hit_limit:
                    break
        return {"file_path": file_path, "pattern": pattern, "hits": hits}

    def repo_tree_snapshot(self, include_exts: List[str], max_files: Optional[int] = None) -> List[str]:
        out = []
        file_limit = self.repo_tree_max_files if max_files is None else max_files
        for rf in self.iter_files(include_exts=include_exts):
            out.append(rf.rel_path)
            if len(out) >= file_limit:
                break
        return out
