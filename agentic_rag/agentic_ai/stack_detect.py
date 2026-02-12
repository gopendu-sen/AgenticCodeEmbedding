import os
from typing import Dict, List

from agentic_rag.core.llm_client import LLMClient


class StackDetector:
    def __init__(
        self,
        signals: Dict[str, List[str]],
        dotnet_project_suffixes: List[str],
        confidence_base: float,
        confidence_per_signal: float,
        confidence_max: float,
    ):
        self.signals = signals
        self.dotnet_project_suffixes = dotnet_project_suffixes
        self.confidence_base = confidence_base
        self.confidence_per_signal = confidence_per_signal
        self.confidence_max = confidence_max

    def detect(self, repo_path: str) -> Dict[str, List[str]]:
        root_files = set(os.listdir(repo_path))
        found: Dict[str, List[str]] = {}

        for stack, sigs in self.signals.items():
            evidence = [signal for signal in sigs if signal in root_files]

            if stack == "dotnet":
                for suffix in self.dotnet_project_suffixes:
                    evidence.extend([f for f in root_files if f.endswith(suffix)])

            if evidence:
                found[stack] = evidence

        return found

    def confidence(self, stacks: Dict[str, List[str]]) -> float:
        total = sum(len(v) for v in stacks.values())
        if total == 0:
            return 0.0
        return min(self.confidence_max, self.confidence_base + self.confidence_per_signal * total)


def llm_stack_fallback(llm: LLMClient, repo_tree: List[str], manifest_snips: Dict[str, str]) -> Dict[str, object]:
    schema = {
        "type": "object",
        "properties": {
            "stacks": {"type": "array", "items": {"type": "string"}},
            "frameworks": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["file_path", "reason"]
            }},
            "confidence": {"type": "number"}
        },
        "required": ["stacks", "frameworks", "evidence", "confidence"]
    }

    messages = [
        {"role": "system", "content": "Return ONLY valid JSON. No prose."},
        {"role": "user", "content":
            "Classify repository tech stacks/frameworks from tree and manifests.\n"
            f"Schema:\n{schema}\n\n"
            f"Repo tree sample:\n{repo_tree[:300]}\n\n"
            f"Manifest snippets:\n{manifest_snips}\n"
        }
    ]
    return llm.chat_json(messages)
