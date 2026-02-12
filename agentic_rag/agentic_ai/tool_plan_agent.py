import logging
from typing import Dict, Any

from agentic_rag.core.llm_client import LLMClient
from agentic_rag.core.repo_tools import RepoTools


logger = logging.getLogger(__name__)


TOOL_PLAN_RULES = """
You output a JSON TOOL PLAN for extracting nodes from a single repository file.
Rules:
- Output ONLY valid JSON. No prose.
- Tools allowed: read_lines, search_in_file.
- Keep steps minimal. Prefer read_lines around likely important sections.
- FINAL nodes MUST contain start_line and end_line that exist in the file.
- If uncertain, request more context in final.need_more.
"""

SECURITY_TAG_RULES = """
You output a JSON TOOL PLAN for extracting SECURITY TAG nodes from a single file.
Rules:
- Output ONLY valid JSON. No prose.
- Tools allowed: read_lines, search_in_file.
- Keep steps minimal and targeted to auth/policy/audit/sensitive operation logic.
- FINAL nodes node_type MUST be one of: auth_guard, policy_check, audit_log, sensitive_op.
- Every FINAL node MUST include: node_type, start_line, end_line.
- `symbol` is optional.
- Every FINAL node MUST include: why_relevant, control_area, severity, evidence.
- Set metadata source to llm_security_tagger.
- If uncertain, return fewer nodes and include final.need_more.
"""


class ToolPlanParseAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose_plan(self, file_path: str, language_hint: str, header_excerpt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": TOOL_PLAN_RULES},
            {"role": "user", "content":
                f"File: {file_path}\nLanguage hint: {language_hint}\n\n"
                f"Header excerpt (may be incomplete):\n{header_excerpt}\n\n"
                "Task: Create a tool plan to extract nodes such as functions/classes/routes/handlers/security primitives.\n"
                "Return JSON with keys: file_path, language, intent, steps[], final{nodes[], need_more[]}."
            }
        ]
        return self.llm.chat_json(messages)

    def finalize(self, file_path: str, language_hint: str, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": TOOL_PLAN_RULES},
            {"role": "user", "content":
                f"Produce FINAL nodes for {file_path}.\n"
                f"Language hint: {language_hint}\n\n"
                f"Tool outputs JSON:\n{tool_outputs}\n\n"
                "Return ONLY JSON with keys: file_path, language, intent, steps(optional), final{nodes[], need_more[]}."
            }
        ]
        return self.llm.chat_json(messages)


class SecurityTagAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose_plan(self, file_path: str, language_hint: str, header_excerpt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SECURITY_TAG_RULES},
            {"role": "user", "content":
                f"File: {file_path}\nLanguage hint: {language_hint}\n\n"
                f"Header excerpt (may be incomplete):\n{header_excerpt}\n\n"
                "Task: Create a tool plan to extract only security nodes: auth_guard, policy_check, audit_log, sensitive_op.\n"
                "Return JSON with keys: file_path, language, intent, steps[], final{nodes[], need_more[]}."
            },
        ]
        return self.llm.chat_json(messages)

    def finalize(self, file_path: str, language_hint: str, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": SECURITY_TAG_RULES},
            {"role": "user", "content":
                f"Produce FINAL security nodes for {file_path}.\n"
                f"Language hint: {language_hint}\n\n"
                f"Tool outputs JSON:\n{tool_outputs}\n\n"
                "Return ONLY JSON with keys: file_path, language, intent, steps(optional), final{nodes[], need_more[]}."
            },
        ]
        return self.llm.chat_json(messages)


def execute_tool_plan(repo: RepoTools, plan: Dict[str, Any]) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    for idx, step in enumerate(plan.get("steps", []), start=1):
        tool = step.get("tool")
        args = step.get("args", {})
        step_id = step.get("id") or f"{tool}_{idx}"
        if not isinstance(args, dict):
            outputs[step_id] = {"error": "invalid_args_type", "args": args}
            logger.warning("Tool plan step has invalid args type: step_id=%s tool=%s", step_id, tool)
            continue

        if tool == "read_lines":
            file_path = args.get("file_path")
            start_line = args.get("start_line")
            end_line = args.get("end_line")
            if not isinstance(file_path, str) or not isinstance(start_line, int) or not isinstance(end_line, int):
                outputs[step_id] = {
                    "error": "missing_or_invalid_args",
                    "required": ["file_path:str", "start_line:int", "end_line:int"],
                    "received": args,
                }
                logger.warning("Tool plan read_lines args invalid: step_id=%s args=%s", step_id, args)
                continue
            try:
                outputs[step_id] = repo.read_lines(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                )
            except Exception as exc:  # noqa: BLE001
                outputs[step_id] = {"error": f"read_lines_failed: {exc}", "received": args}
                logger.warning("Tool plan read_lines failed: step_id=%s error=%s", step_id, exc)
        elif tool == "search_in_file":
            file_path = args.get("file_path")
            pattern = args.get("pattern")
            max_hits = args.get("max_hits")
            if not isinstance(file_path, str) or not isinstance(pattern, str):
                outputs[step_id] = {
                    "error": "missing_or_invalid_args",
                    "required": ["file_path:str", "pattern:str", "max_hits?:int"],
                    "received": args,
                }
                logger.warning("Tool plan search_in_file args invalid: step_id=%s args=%s", step_id, args)
                continue
            kwargs = {"file_path": file_path, "pattern": pattern}
            if isinstance(max_hits, int):
                kwargs["max_hits"] = max_hits
            try:
                outputs[step_id] = repo.search_in_file(**kwargs)
            except Exception as exc:  # noqa: BLE001
                outputs[step_id] = {"error": f"search_in_file_failed: {exc}", "received": args}
                logger.warning("Tool plan search_in_file failed: step_id=%s error=%s", step_id, exc)
        else:
            outputs[step_id] = {"error": f"unknown_tool: {tool}", "step": step}
            logger.warning("Tool plan unknown tool: step_id=%s tool=%s", step_id, tool)
    return outputs
