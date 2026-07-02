"""Schema-drift dataset slice — a distinctive, cheap originality contribution.

Research finding: nested/DAG calls and injection are already saturated in the literature, but there is
no dedicated *fine-tuning* dataset teaching a model to stay correct when a tool's schema CHANGES
(params renamed, retyped, or newly required). We synthesize that slice by mutating real tool schemas,
with labels correct by construction:

  add_required : a new REQUIRED param the query doesn't supply  -> clarify (ask for it)
  retype_enum  : a param constrained to an enum the query's value violates -> clarify (ask for valid value)
  rename       : a required param renamed; the query supplies the value under the OLD name
                 -> CALL using the CURRENT schema's field name (schema-aware remapping)

Every example carries meta.sd_kind. Deterministic given the seed.
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

_ENUMS = {
    "priority": ["low", "medium", "high"],
    "status": ["open", "closed", "pending"],
    "unit": ["celsius", "fahrenheit"],
    "mode": ["fast", "accurate"],
}


def _required(tool: Dict[str, Any]) -> List[str]:
    return list((tool.get("parameters") or {}).get("required", []))


def _props(tool: Dict[str, Any]) -> Dict[str, Any]:
    return dict((tool.get("parameters") or {}).get("properties", {}))


def _desc(tool: Dict[str, Any]) -> str:
    return (tool.get("description") or tool["name"]).rstrip(".")


def make_add_required(tool: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """Add a new required param the query won't provide -> clarify."""
    drifted = copy.deepcopy(tool)
    params = drifted.setdefault("parameters", {"type": "object", "properties": {}, "required": []})
    props = params.setdefault("properties", {})
    req = params.setdefault("required", [])
    new_field = rng.choice(["confirmation_token", "api_version", "region", "account_id"])
    props[new_field] = {"type": "string", "description": f"Required {new_field}"}
    req.append(new_field)
    return {
        "tools": [drifted],
        "query": f"Please {_desc(tool).lower()}.",
        "answer": {
            "type": "clarify",
            "content": f"`{tool['name']}` now requires `{new_field}`. Could you provide it?",
        },
        "meta": {"source": "schema_drift", "hn_kind": None, "sd_kind": "add_required"},
    }


def make_retype_enum(tool: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Constrain a required string param to an enum the query's value violates -> clarify."""
    req = _required(tool)
    props = _props(tool)
    candidates = [r for r in req if (props.get(r, {}) or {}).get("type", "string") == "string"]
    if not candidates:
        return None
    field = rng.choice(candidates)
    enum = _ENUMS.get(field) or rng.choice(list(_ENUMS.values()))
    drifted = copy.deepcopy(tool)
    drifted["parameters"]["properties"][field] = {"type": "string", "enum": list(enum)}
    bad_value = "urgent-ASAP"  # deliberately not in any enum
    return {
        "tools": [drifted],
        "query": f"{_desc(tool)} with {field} = \"{bad_value}\".",
        "answer": {
            "type": "clarify",
            "content": (
                f"`{field}` must now be one of {', '.join(enum)} — \"{bad_value}\" isn't valid. "
                f"Which would you like?"
            ),
        },
        "meta": {"source": "schema_drift", "hn_kind": None, "sd_kind": "retype_enum"},
    }


def make_rename(tool: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Rename a required param; query supplies the value under the OLD name -> call with NEW name."""
    req = _required(tool)
    if not req:
        return None
    old = rng.choice(req)
    new = f"{old}_id" if not old.endswith("_id") else f"{old}_value"
    drifted = copy.deepcopy(tool)
    props = drifted["parameters"]["properties"]
    if old in props:
        props[new] = props.pop(old)
    drifted["parameters"]["required"] = [new if r == old else r for r in req]

    # supply plausible values for all current required params; the query names the OLD field
    value = "ACME-42"
    args = {}
    for r in drifted["parameters"]["required"]:
        args[r] = value if r == new else "example"
    return {
        "tools": [drifted],
        "query": f"{_desc(tool)} — the {old} is {value}.",
        "answer": {"type": "tool_call", "calls": [{"name": tool["name"], "arguments": args}]},
        "meta": {"source": "schema_drift", "hn_kind": None, "sd_kind": "rename"},
    }


def generate(
    tool_pool: List[Dict[str, Any]],
    n: int,
    kind_weights: Dict[str, float],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 13)  # offset from other generators
    if not tool_pool:
        return []
    kinds = list(kind_weights.keys())
    weights = [kind_weights[k] for k in kinds]
    makers = {
        "add_required": make_add_required,
        "retype_enum": make_retype_enum,
        "rename": make_rename,
    }
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        tool = rng.choice(tool_pool)
        ex = makers[kind](tool, rng)
        if ex is not None:
            out.append(ex)
    return out
