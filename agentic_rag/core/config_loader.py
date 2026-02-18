import json
import os
from typing import Any, Dict, List, Optional

import yaml

from agentic_rag.core.config import AgenticRagConfig


ENV_PREFIX = "AGENTIC_RAG__"


class ConfigLoaderError(ValueError):
    """Raised when YAML or environment overrides are invalid."""


def _read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise ConfigLoaderError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}

    if not isinstance(loaded, dict):
        raise ConfigLoaderError("Config file root must be a mapping/object")

    return loaded


def _deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _parse_extends_field(raw: Any, *, source_path: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        raise ConfigLoaderError(f"Invalid extends in {source_path}: expected string or list of strings")

    out: List[str] = []
    for idx, entry in enumerate(values):
        if not isinstance(entry, str) or not entry.strip():
            raise ConfigLoaderError(f"Invalid extends entry at index {idx} in {source_path}")
        out.append(entry.strip())
    return out


def _load_yaml_with_extends(path: str, *, ancestry: Optional[List[str]] = None) -> Dict[str, Any]:
    absolute_path = os.path.abspath(path)
    parent_chain = list(ancestry or [])
    if absolute_path in parent_chain:
        cycle = " -> ".join(parent_chain + [absolute_path])
        raise ConfigLoaderError(f"Config extends cycle detected: {cycle}")

    data = _read_yaml(absolute_path)
    extends_entries = _parse_extends_field(data.pop("extends", None), source_path=absolute_path)
    merged: Dict[str, Any] = {}
    config_dir = os.path.dirname(absolute_path)

    for entry in extends_entries:
        include_path = _resolve_path(entry, config_dir)
        included = _load_yaml_with_extends(include_path, ancestry=parent_chain + [absolute_path])
        merged = _deep_merge_dicts(merged, included)

    return _deep_merge_dicts(merged, data)


def _parse_bool(raw: str) -> bool:
    normalized = raw.strip().lower()
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}

    if normalized in truthy:
        return True
    if normalized in falsy:
        return False

    raise ConfigLoaderError(f"Invalid boolean override value: {raw}")


def _cast_env_value(raw: str, current_value: Any) -> Any:
    try:
        if isinstance(current_value, bool):
            return _parse_bool(raw)

        if isinstance(current_value, int) and not isinstance(current_value, bool):
            return int(raw)

        if isinstance(current_value, float):
            return float(raw)

        if isinstance(current_value, list):
            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ConfigLoaderError("Expected JSON array for list override")
            return parsed

        if isinstance(current_value, dict):
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ConfigLoaderError("Expected JSON object for dict override")
            return parsed

        return raw
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigLoaderError(f"Invalid env override value '{raw}' for type {type(current_value).__name__}") from exc


def _set_nested_override(config_data: Dict[str, Any], path_parts: List[str], raw_value: str) -> None:
    cursor: Any = config_data
    for key in path_parts[:-1]:
        if not isinstance(cursor, dict) or key not in cursor:
            dotted = ".".join(path_parts)
            raise ConfigLoaderError(f"Unknown env override path: {dotted}")
        cursor = cursor[key]

    leaf = path_parts[-1]
    if not isinstance(cursor, dict) or leaf not in cursor:
        dotted = ".".join(path_parts)
        raise ConfigLoaderError(f"Unknown env override path: {dotted}")

    current_value = cursor[leaf]
    cursor[leaf] = _cast_env_value(raw_value, current_value)


def _apply_env_overrides(config_data: Dict[str, Any]) -> None:
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue

        parts = [segment.strip().lower() for segment in env_key.split("__")[1:] if segment.strip()]
        if len(parts) < 2:
            raise ConfigLoaderError(f"Invalid env override key format: {env_key}")

        _set_nested_override(config_data, parts, env_value)


def _resolve_path(path_value: str, config_dir: str) -> str:
    expanded = os.path.expanduser(path_value)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(config_dir, expanded))


def _resolve_config_paths(config_data: Dict[str, Any], config_dir: str) -> None:
    paths = config_data.get("paths")
    if not isinstance(paths, dict):
        raise ConfigLoaderError("Config must contain a 'paths' object")

    for key in ("repo_path", "out_dir", "sqlite_path", "chroma_dir", "reports_dir"):
        raw = paths.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigLoaderError(f"paths.{key} must be a non-empty string")
        paths[key] = _resolve_path(raw, config_dir)

    chat = config_data.get("chat")
    if isinstance(chat, dict):
        memory = chat.get("memory")
        if isinstance(memory, dict):
            sqlite_path = memory.get("sqlite_path")
            if isinstance(sqlite_path, str) and sqlite_path.strip():
                memory["sqlite_path"] = _resolve_path(sqlite_path, config_dir)

    evaluation = config_data.get("evaluation")
    if isinstance(evaluation, dict):
        for key in ("rules_json_path", "jobs_dir", "reports_dir", "log_jsonl_path"):
            raw = evaluation.get(key)
            if isinstance(raw, str) and raw.strip():
                evaluation[key] = _resolve_path(raw, config_dir)


def load_agentic_rag_config(config_path: str) -> AgenticRagConfig:
    absolute_config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(absolute_config_path)

    config_data = _load_yaml_with_extends(absolute_config_path)
    _apply_env_overrides(config_data)
    _resolve_config_paths(config_data, config_dir)

    try:
        return AgenticRagConfig.model_validate(config_data)
    except Exception as exc:  # noqa: BLE001
        raise ConfigLoaderError(f"Invalid configuration: {exc}") from exc
