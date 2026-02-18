from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from agentic_rag.agentic_ai.tool_plan_agent import ToolPlanParseAgent, execute_tool_plan
from agentic_rag.core.repo_tools import RepoTools


class _FakeLLM:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat_json(self, messages: List[Dict[str, str]], max_tokens: int | None = None) -> Dict[str, Any]:
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        idx = len(self.calls)
        return {
            "file_path": "demo.py",
            "language": "python",
            "intent": "extract",
            "final": {
                "nodes": [
                    {
                        "node_type": "function",
                        "start_line": idx,
                        "end_line": idx,
                        "symbol": f"fn_{idx}",
                    }
                ],
                "need_more": [],
            },
        }


def test_finalize_splits_large_tool_outputs():
    llm = _FakeLLM()
    agent = ToolPlanParseAgent(
        llm,  # type: ignore[arg-type]
        auto_split_enabled=True,
        soft_input_tokens=120,
        target_input_tokens=80,
        max_completion_tokens=64,
        oom_retry_max_attempts=1,
        oom_retry_shrink_ratio=0.6,
    )
    tool_outputs = {
        f"step_{idx}": {"text": "x" * 240, "idx": idx}
        for idx in range(1, 6)
    }
    out = agent.finalize("demo.py", "python", tool_outputs)
    assert len(llm.calls) > 1
    assert out.get("final", {}).get("nodes")


def test_execute_tool_plan_clamps_read_lines_and_text(tmp_path: Path):
    target = tmp_path / "demo.py"
    target.write_text("\n".join(f"line {i}" for i in range(1, 201)), encoding="utf-8")
    repo = RepoTools(
        repo_path=str(tmp_path),
        exclude_dirs=[],
        max_file_size=1_000_000,
        read_file_max_chars=200_000,
        repo_tree_max_files=100,
        tool_search_max_hits=100,
    )
    plan = {
        "steps": [
            {
                "id": "r1",
                "tool": "read_lines",
                "args": {"file_path": "demo.py", "start_line": 1, "end_line": 180},
            }
        ]
    }
    out = execute_tool_plan(
        repo,
        plan,
        max_read_line_span=10,
        max_output_text_chars=30,
    )
    payload = out["r1"]
    assert int(payload["end_line"]) <= 10
    assert len(str(payload.get("text", ""))) <= 30
