import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from agentic_rag.core.llm_client import LLMClient
from agentic_rag.core.prompt_budget import (
    build_llm_call_log_fields,
    estimate_tokens_for_messages,
    estimate_tokens_from_text,
    is_oom_error,
    pack_items_by_token_budget,
)
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


def _normalize_need_more(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _dedupe_nodes(nodes: Iterable[Dict[str, Any]], default_file_path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in nodes:
        if not isinstance(raw, dict):
            continue
        node_type = str(raw.get("node_type", "")).strip()
        file_path = str(raw.get("file_path", "")).strip() or default_file_path
        try:
            start_line = int(raw.get("start_line"))
            end_line = int(raw.get("end_line"))
        except Exception:  # noqa: BLE001
            continue
        symbol = str(raw.get("symbol", "")).strip()
        key = (node_type, start_line, end_line, symbol, file_path)
        if key in seen:
            continue
        seen.add(key)
        copied = dict(raw)
        copied["file_path"] = file_path
        copied["start_line"] = start_line
        copied["end_line"] = end_line
        out.append(copied)
    return out


def _truncate_tool_output(value: Any, max_chars: int) -> Any:
    if max_chars < 1:
        return value
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        return [_truncate_tool_output(item, max_chars) for item in value]
    if isinstance(value, dict):
        return {key: _truncate_tool_output(item, max_chars) for key, item in value.items()}
    return value


def _tool_output_items(tool_outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for step_id in sorted(tool_outputs.keys()):
        items.append({"step_id": step_id, "output": tool_outputs.get(step_id)})
    return items


def _merge_partial_payloads(
    payloads: List[Dict[str, Any]],
    *,
    file_path: str,
    language_hint: str,
) -> Dict[str, Any]:
    all_nodes: List[Dict[str, Any]] = []
    all_need_more: List[str] = []
    intent = ""
    steps: List[Any] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        if not intent:
            intent = str(payload.get("intent", "")).strip()
        payload_steps = payload.get("steps")
        if isinstance(payload_steps, list):
            steps.extend(payload_steps)
        final = payload.get("final")
        if isinstance(final, dict):
            nodes = final.get("nodes")
            if isinstance(nodes, list):
                for node in nodes:
                    if isinstance(node, dict):
                        all_nodes.append(node)
            all_need_more.extend(_normalize_need_more(final.get("need_more")))

    merged_nodes = _dedupe_nodes(all_nodes, file_path)
    merged_need_more = _normalize_need_more(all_need_more)
    out: Dict[str, Any] = {
        "file_path": file_path,
        "language": language_hint,
        "intent": intent,
        "final": {
            "nodes": merged_nodes,
            "need_more": merged_need_more,
        },
    }
    if steps:
        out["steps"] = steps
    return out


class _BudgetedFinalizeAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        rules: str,
        propose_task: str,
        finalize_task: str,
        auto_split_enabled: bool,
        soft_input_tokens: int,
        target_input_tokens: int,
        max_completion_tokens: int,
        oom_retry_max_attempts: int,
        oom_retry_shrink_ratio: float,
        tool_output_text_max_chars: int,
        agent_label: str,
    ):
        self.llm = llm
        self.rules = rules
        self.propose_task = propose_task
        self.finalize_task = finalize_task
        self.auto_split_enabled = bool(auto_split_enabled)
        self.soft_input_tokens = max(1, int(soft_input_tokens))
        self.target_input_tokens = max(1, min(int(target_input_tokens), self.soft_input_tokens))
        self.max_completion_tokens = max(1, int(max_completion_tokens))
        self.oom_retry_max_attempts = max(0, int(oom_retry_max_attempts))
        self.oom_retry_shrink_ratio = float(oom_retry_shrink_ratio)
        self.tool_output_text_max_chars = max(1, int(tool_output_text_max_chars))
        self.agent_label = agent_label

    def _build_finalize_messages(
        self,
        file_path: str,
        language_hint: str,
        tool_outputs: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.rules},
            {
                "role": "user",
                "content": (
                    f"{self.finalize_task}\n"
                    f"File: {file_path}\n"
                    f"Language hint: {language_hint}\n\n"
                    f"Tool outputs JSON:\n{json.dumps(tool_outputs, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    "Return ONLY JSON with keys: file_path, language, intent, steps(optional), final{nodes[], need_more[]}."
                ),
            },
        ]

    def _build_synthesis_messages(
        self,
        file_path: str,
        language_hint: str,
        partials: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        compact_partials: List[Dict[str, Any]] = []
        for idx, partial in enumerate(partials, start=1):
            final = partial.get("final") if isinstance(partial, dict) else {}
            if not isinstance(final, dict):
                final = {}
            nodes = final.get("nodes")
            if not isinstance(nodes, list):
                nodes = []
            compact_partials.append(
                {
                    "batch": idx,
                    "intent": str(partial.get("intent", "")) if isinstance(partial, dict) else "",
                    "nodes": [node for node in nodes if isinstance(node, dict)],
                    "need_more": _normalize_need_more(final.get("need_more")),
                }
            )

        return [
            {"role": "system", "content": self.rules},
            {
                "role": "user",
                "content": (
                    f"Synthesize FINAL nodes from batched partial outputs.\n"
                    f"File: {file_path}\n"
                    f"Language hint: {language_hint}\n\n"
                    "Merge and deduplicate nodes by node_type/start_line/end_line/symbol/file_path.\n"
                    "Keep only valid, file-grounded nodes and carry forward unresolved needs in final.need_more.\n\n"
                    f"Partial outputs JSON:\n{json.dumps(compact_partials, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    "Return ONLY JSON with keys: file_path, language, intent, final{nodes[], need_more[]}."
                ),
            },
        ]

    def _call_json(
        self,
        *,
        messages: List[Dict[str, str]],
        phase: str,
        split_batch_idx: Optional[int] = None,
        split_batch_total: Optional[int] = None,
    ) -> Dict[str, Any]:
        fields = build_llm_call_log_fields(
            phase=phase,
            messages=messages,
            split_batch_idx=split_batch_idx,
            split_batch_total=split_batch_total,
        )
        logger.info(
            "%s LLM call: phase=%s estimated_input_tokens=%d estimated_input_chars=%d split_batch_idx=%s split_batch_total=%s",
            self.agent_label,
            fields["phase"],
            int(fields["estimated_input_tokens"]),
            int(fields["estimated_input_chars"]),
            fields.get("split_batch_idx"),
            fields.get("split_batch_total"),
        )
        return self.llm.chat_json(messages, max_tokens=self.max_completion_tokens)

    def _finalize_with_budget(
        self,
        file_path: str,
        language_hint: str,
        tool_outputs: Dict[str, Any],
        *,
        soft_budget: int,
        target_budget: int,
    ) -> Dict[str, Any]:
        single_messages = self._build_finalize_messages(file_path, language_hint, tool_outputs)
        single_estimate = estimate_tokens_for_messages(single_messages)
        if not self.auto_split_enabled or single_estimate <= soft_budget:
            return self._call_json(messages=single_messages, phase=f"{self.agent_label}_finalize_single")

        items = _tool_output_items(tool_outputs)
        item_token_fn = lambda item: estimate_tokens_from_text(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        batches = pack_items_by_token_budget(
            items,
            item_token_fn,
            soft_budget=soft_budget,
            target_budget=target_budget,
        )
        if len(batches) <= 1:
            return self._call_json(messages=single_messages, phase=f"{self.agent_label}_finalize_single")

        logger.info(
            "%s finalize split: batches=%d soft_budget=%d target_budget=%d",
            self.agent_label,
            len(batches),
            soft_budget,
            target_budget,
        )
        partials: List[Dict[str, Any]] = []
        for batch_idx, batch in enumerate(batches, start=1):
            batch_outputs = {item["step_id"]: item["output"] for item in batch}
            batch_messages = self._build_finalize_messages(file_path, language_hint, batch_outputs)
            partial = self._call_json(
                messages=batch_messages,
                phase=f"{self.agent_label}_finalize_batch",
                split_batch_idx=batch_idx,
                split_batch_total=len(batches),
            )
            partials.append(partial)

        merged = _merge_partial_payloads(partials, file_path=file_path, language_hint=language_hint)
        if len(partials) <= 1:
            return merged

        synth_messages = self._build_synthesis_messages(file_path, language_hint, partials)
        try:
            synthesized = self._call_json(messages=synth_messages, phase=f"{self.agent_label}_finalize_synth")
            combined = _merge_partial_payloads([merged, synthesized], file_path=file_path, language_hint=language_hint)
            return combined
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s synth failed; using merged partials: error=%s", self.agent_label, exc)
            return merged

    def propose_plan(self, file_path: str, language_hint: str, header_excerpt: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self.rules},
            {
                "role": "user",
                "content": (
                    f"File: {file_path}\nLanguage hint: {language_hint}\n\n"
                    f"Header excerpt (may be incomplete):\n{header_excerpt}\n\n"
                    f"{self.propose_task}\n"
                    "Return JSON with keys: file_path, language, intent, steps[], final{nodes[], need_more[]}."
                ),
            },
        ]
        return self._call_json(messages=messages, phase=f"{self.agent_label}_propose")

    def finalize(self, file_path: str, language_hint: str, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tool_outputs, dict):
            tool_outputs = {"raw_output": tool_outputs}

        effective_outputs = _truncate_tool_output(dict(tool_outputs), self.tool_output_text_max_chars)
        base_soft = self.soft_input_tokens
        base_target = self.target_input_tokens
        for attempt in range(0, self.oom_retry_max_attempts + 1):
            shrink = self.oom_retry_shrink_ratio ** attempt
            soft_budget = max(1, int(round(base_soft * shrink)))
            target_budget = max(1, min(int(round(base_target * shrink)), soft_budget))
            try:
                return self._finalize_with_budget(
                    file_path,
                    language_hint,
                    effective_outputs,
                    soft_budget=soft_budget,
                    target_budget=target_budget,
                )
            except Exception as exc:  # noqa: BLE001
                if is_oom_error(exc) and attempt < self.oom_retry_max_attempts:
                    logger.warning(
                        "%s finalize OOM retry: attempt=%d/%d next_soft_budget=%d next_target_budget=%d error=%s",
                        self.agent_label,
                        attempt + 1,
                        self.oom_retry_max_attempts + 1,
                        max(1, int(round(base_soft * (self.oom_retry_shrink_ratio ** (attempt + 1))))),
                        max(1, int(round(base_target * (self.oom_retry_shrink_ratio ** (attempt + 1))))),
                        exc,
                    )
                    continue
                raise


class ToolPlanParseAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        auto_split_enabled: bool = True,
        soft_input_tokens: int = 1400,
        target_input_tokens: int = 850,
        max_completion_tokens: int = 384,
        oom_retry_max_attempts: int = 2,
        oom_retry_shrink_ratio: float = 0.6,
        tool_output_text_max_chars: int = 600,
    ):
        self._agent = _BudgetedFinalizeAgent(
            llm,
            rules=TOOL_PLAN_RULES,
            propose_task="Task: Create a tool plan to extract nodes such as functions/classes/routes/handlers/security primitives.",
            finalize_task="Produce FINAL nodes from the tool outputs.",
            auto_split_enabled=auto_split_enabled,
            soft_input_tokens=soft_input_tokens,
            target_input_tokens=target_input_tokens,
            max_completion_tokens=max_completion_tokens,
            oom_retry_max_attempts=oom_retry_max_attempts,
            oom_retry_shrink_ratio=oom_retry_shrink_ratio,
            tool_output_text_max_chars=tool_output_text_max_chars,
            agent_label="parser_fallback",
        )

    def propose_plan(self, file_path: str, language_hint: str, header_excerpt: str) -> Dict[str, Any]:
        return self._agent.propose_plan(file_path, language_hint, header_excerpt)

    def finalize(self, file_path: str, language_hint: str, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._agent.finalize(file_path, language_hint, tool_outputs)


class SecurityTagAgent:
    def __init__(
        self,
        llm: LLMClient,
        *,
        auto_split_enabled: bool = True,
        soft_input_tokens: int = 1400,
        target_input_tokens: int = 850,
        max_completion_tokens: int = 384,
        oom_retry_max_attempts: int = 2,
        oom_retry_shrink_ratio: float = 0.6,
        tool_output_text_max_chars: int = 600,
    ):
        self._agent = _BudgetedFinalizeAgent(
            llm,
            rules=SECURITY_TAG_RULES,
            propose_task="Task: Create a tool plan to extract only security nodes: auth_guard, policy_check, audit_log, sensitive_op.",
            finalize_task="Produce FINAL security nodes from the tool outputs.",
            auto_split_enabled=auto_split_enabled,
            soft_input_tokens=soft_input_tokens,
            target_input_tokens=target_input_tokens,
            max_completion_tokens=max_completion_tokens,
            oom_retry_max_attempts=oom_retry_max_attempts,
            oom_retry_shrink_ratio=oom_retry_shrink_ratio,
            tool_output_text_max_chars=tool_output_text_max_chars,
            agent_label="security_tagger",
        )

    def propose_plan(self, file_path: str, language_hint: str, header_excerpt: str) -> Dict[str, Any]:
        return self._agent.propose_plan(file_path, language_hint, header_excerpt)

    def finalize(self, file_path: str, language_hint: str, tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
        return self._agent.finalize(file_path, language_hint, tool_outputs)


def execute_tool_plan(
    repo: RepoTools,
    plan: Dict[str, Any],
    *,
    max_read_line_span: Optional[int] = None,
    max_output_text_chars: Optional[int] = None,
) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    output_char_limit = max_output_text_chars if isinstance(max_output_text_chars, int) else None
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
                    max_line_span=max_read_line_span,
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

        if output_char_limit and output_char_limit > 0:
            outputs[step_id] = _truncate_tool_output(outputs.get(step_id), output_char_limit)
    return outputs
