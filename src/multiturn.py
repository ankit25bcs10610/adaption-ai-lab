"""Multi-turn example generator — the highest-leverage data for BFCL v3/v4.

Multi-turn is ~30% of the BFCL v4 aggregate and the split where small fine-tunes provably beat GPT-4o
(xLAM-2-8b 70.5 vs GPT-4o 51). We synthesize the three canonical BFCL multi-turn failure modes from real
tool schemas, deterministically (labels correct by construction):

  miss_param  -- user omits a required arg across the conversation -> assistant must CLARIFY, not guess.
  miss_func   -- the needed tool wasn't available earlier; once provided, assistant must CALL it.
  long_context-- the correct tool is buried among many distractor tools -> selection under load.

Each example carries a `history` list of {role, content} turns preceding the final user query, consumed
by format_utils.build_eval_prompt / to_prompt_completion.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


def _required_args(tool: Dict[str, Any]) -> List[str]:
    return list((tool.get("parameters") or {}).get("required", []))


def _properties(tool: Dict[str, Any]) -> Dict[str, Any]:
    return dict((tool.get("parameters") or {}).get("properties", {}))


def _sample_value(spec: Dict[str, Any], rng: random.Random) -> Any:
    if "enum" in spec and spec["enum"]:
        return rng.choice(spec["enum"])
    t = spec.get("type", "string")
    if t in ("integer", "number"):
        return rng.choice([1, 3, 7, 12, 30])
    if t == "boolean":
        return rng.choice([True, False])
    if t == "array":
        return []
    return rng.choice(["Mumbai", "2026-02-01", "report.pdf", "acme-corp", "en"])


def make_miss_param(tool: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """User references the task but never supplies a required arg -> clarify."""
    req = _required_args(tool)
    if not req:
        return None
    dropped = rng.choice(req)
    desc = (tool.get("description") or tool["name"]).rstrip(".")
    history = [
        {"role": "user", "content": f"Hi, I'd like to {desc.lower()}."},
        {"role": "assistant", "content": '{"action":"clarify","message":"Sure — I can help with that."}'},
    ]
    return {
        "tools": [tool],
        "history": history,
        "query": "Go ahead and do it now.",
        "answer": {
            "type": "clarify",
            "content": f"I still need the required `{dropped}` before I can call `{tool['name']}`. What should it be?",
        },
        "meta": {"source": "multiturn", "hn_kind": None, "mt_kind": "miss_param"},
    }


def make_miss_func(tool: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """The needed tool was absent earlier (assistant refused); now it's provided -> call it."""
    props = _properties(tool)
    args = {name: _sample_value(props.get(name, {}), rng) for name in _required_args(tool)}
    desc = (tool.get("description") or tool["name"]).rstrip(".")
    history = [
        {"role": "user", "content": f"Can you {desc.lower()}?"},
        {"role": "assistant", "content": '{"action":"refuse","message":"No available tool can do that yet."}'},
        {"role": "user", "content": f"I've just enabled the `{tool['name']}` tool for you."},
    ]
    return {
        "tools": [tool],
        "history": history,
        "query": f"Great — now {desc.lower()}.",
        "answer": {"type": "tool_call", "calls": [{"name": tool["name"], "arguments": args}]},
        "meta": {"source": "multiturn", "hn_kind": None, "mt_kind": "miss_func"},
    }


def make_long_context(
    tool: Dict[str, Any], distractors: List[Dict[str, Any]], rng: random.Random
) -> Dict[str, Any]:
    """Correct tool buried among many distractors -> must select the right one."""
    props = _properties(tool)
    args = {name: _sample_value(props.get(name, {}), rng) for name in _required_args(tool)}
    tools = distractors + [tool]
    rng.shuffle(tools)
    desc = (tool.get("description") or tool["name"]).rstrip(".")
    history = [
        {"role": "user", "content": "I have a lot of tools available; help me use the right one."},
        {"role": "assistant", "content": '{"action":"clarify","message":"Of course — what do you need?"}'},
    ]
    return {
        "tools": tools,
        "history": history,
        "query": f"Please {desc.lower()}.",
        "answer": {"type": "tool_call", "calls": [{"name": tool["name"], "arguments": args}]},
        "meta": {"source": "multiturn", "hn_kind": None, "mt_kind": "long_context"},
    }


def generate(
    tool_pool: List[Dict[str, Any]],
    n: int,
    kind_weights: Dict[str, float],
    seed: int = 42,
    long_context_size: int = 10,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 7)  # offset so it doesn't mirror hard_negatives' RNG
    if len(tool_pool) < 2:
        raise ValueError("need >=2 tools for multi-turn generation")
    kinds = list(kind_weights.keys())
    weights = [kind_weights[k] for k in kinds]
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        tool = rng.choice(tool_pool)
        if kind == "miss_param":
            ex = make_miss_param(tool, rng)
        elif kind == "miss_func":
            ex = make_miss_func(tool, rng)
        else:  # long_context
            k = min(long_context_size, len(tool_pool) - 1)
            distractors = [t for t in rng.sample(tool_pool, k + 1) if t["name"] != tool["name"]][:k]
            ex = make_long_context(tool, distractors, rng)
        if ex is not None:
            out.append(ex)
    return out
