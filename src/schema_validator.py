"""Validate tool calls against the tools' JSON Schemas.

Two jobs:
  1. Data hygiene: every POSITIVE training target must be a schema-correct call. Malformed targets
     silently poison training, so we drop them at build time.
  2. Eval scoring: a predicted call is only "correct" if the tool exists AND its arguments validate.

We validate each call's `arguments` against the tool's `parameters` schema (standard OpenAI/JSON-Schema
"parameters" object). Missing `additionalProperties` is treated as False so hallucinated extra args fail.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from jsonschema import Draft7Validator


def _index_tools(tools: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {t["name"]: t for t in tools}


def _params_schema(tool: Dict[str, Any]) -> Dict[str, Any]:
    schema = dict(tool.get("parameters") or {"type": "object", "properties": {}})
    schema.setdefault("type", "object")
    # Reject arguments not defined in the schema unless the tool explicitly allows extras.
    schema.setdefault("additionalProperties", False)
    return schema


def validate_call(
    call: Dict[str, Any], tools: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """Return (ok, reason). ok=True means the call names a real tool with valid arguments."""
    name = call.get("name")
    if not name:
        return False, "call has no 'name'"
    index = _index_tools(tools)
    if name not in index:
        return False, f"tool '{name}' is not in the provided tool list"
    args = call.get("arguments", {})
    if not isinstance(args, dict):
        return False, "'arguments' is not an object"
    schema = _params_schema(index[name])
    errors = sorted(Draft7Validator(schema).iter_errors(args), key=lambda e: e.path)
    if errors:
        return False, "; ".join(e.message for e in errors[:3])
    return True, "ok"


def validate_answer(answer: Dict[str, Any], tools: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Validate a full example answer (used to filter positive training targets)."""
    if answer.get("type") != "tool_call":
        # refuse / clarify have no schema to validate
        return True, "ok"
    calls = answer.get("calls") or []
    if not calls:
        return False, "tool_call answer has no calls"
    for c in calls:
        ok, reason = validate_call(c, tools)
        if not ok:
            return False, reason
    return True, "ok"


def required_args(tool: Dict[str, Any]) -> List[str]:
    return list((tool.get("parameters") or {}).get("required", []))
