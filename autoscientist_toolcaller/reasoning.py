"""Correct-by-construction reasoning traces — a short <think> block derived from the SAME gold answer.

Rendered from the gold (not model-guessed), so the trace can never disagree with the label. Kept short
(<~60 tokens). Wrapped as `<think> ... </think>` before the JSON envelope; the eval parser
(`format_utils.strip_think`) reads only what follows `</think>`, so grading is unaffected.

Config-gated: `dataset.reasoning_traces: false` by default → the build is byte-identical to today. Enable
to A/B a reasoning arm.

TRAP (documented): do NOT enable this AND the platform `reasoning_traces` recipe at the same time — that
double-applies (and un-verifies) reasoning. Pick one arm and A/B them. `build_dataset` warns if both are on.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

from .format_utils import target_to_json_str


def _tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t.get("name", "") for t in (tools or []) if t.get("name")]


def render_trace(example: Dict[str, Any], rng: random.Random) -> str:
    """A short, gold-derived reasoning string (no leading/trailing tags)."""
    ans = example.get("answer", {}) or {}
    meta = example.get("meta", {}) or {}
    t = ans.get("type")
    if t == "tool_call":
        names = [c.get("name") for c in ans.get("calls", []) if c.get("name")]
        nm = ", ".join(f"`{n}`" for n in names)
        if len(names) > 1:
            return f"The request has {len(names)} independent intents; {nm} each fit and their required args are present, so I'll issue all the calls."
        return rng.choice([
            f"{nm} matches this request and every required argument is present, so I'll call it.",
            f"This maps cleanly to {nm}; all required arguments are given, so a direct call is correct.",
        ])
    if t == "refuse":
        offered = ", ".join(f"`{n}`" for n in _tool_names(example.get("tools"))[:4]) or "the listed tools"
        return rng.choice([
            f"None of the available tools ({offered}) can satisfy this, so I should refuse rather than invent a call.",
            f"This request is outside {offered}; the right move is to decline, not hallucinate a tool.",
        ])
    # clarify
    dropped = meta.get("dropped")
    if meta.get("hn_kind") == "ambiguous" or meta.get("mt_kind") is None and dropped is None and t == "clarify":
        pass
    if dropped:
        return f"A tool fits, but the required `{dropped}` is missing — I should ask for it before calling, not guess."
    if meta.get("hn_kind") == "ambiguous":
        return "Two available tools could apply and the request doesn't say which, so I should ask before acting."
    return "The matching tool is missing a required argument (or the choice is ambiguous), so I should clarify before calling."


def with_trace(prompt_completion: Dict[str, str], example: Dict[str, Any], rng: random.Random) -> Dict[str, str]:
    """Prepend a `<think>…</think>` trace to the completion of a {prompt, completion} row."""
    trace = render_trace(example, rng)
    pc = dict(prompt_completion)
    pc["completion"] = f"<think>{trace}</think>\n{pc['completion']}"
    return pc


def apply(pc_rows: List[Dict[str, str]], examples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, str]]:
    """Add reasoning traces to a list of prompt/completion rows aligned with their source examples."""
    rng = random.Random(seed + 555)
    return [with_trace(pc, ex, rng) for pc, ex in zip(pc_rows, examples)]
