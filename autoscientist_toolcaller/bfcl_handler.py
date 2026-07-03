"""Envelope translation for scoring our model with the official BFCL v4 harness.

Our model emits ONE JSON envelope per decision:
    {"action": "call"|"refuse"|"clarify", "calls": [{"name","arguments"}, ...], "message": "..."}

BFCL's AST checker expects a *decoded* list of calls, each shaped `{func_name: {arg: value}}`, and an
EMPTY list when no function should be called (its relevance/irrelevance category). This module maps our
envelope onto that form so the official checker can score us fairly. It is intentionally dependency-free
and pure so it can be dropped into a `bfcl-eval` checkout (wrap it in a model handler whose
`decode_ast` / `decode_execute` returns `translate(raw_output)`).

Nothing here runs the harness — it only translates. See `autoscientist_toolcaller/export_bfcl.py` for the data export.
"""
from __future__ import annotations

from typing import Any, Dict, List

from .format_utils import parse_model_output


def translate(raw_output: str) -> List[Dict[str, Any]]:
    """Our raw envelope string -> BFCL decoded-AST list.

    call    -> [{func_name: {arg: value, ...}}, ...]
    refuse  -> []   (irrelevance: correct behavior is to make no call)
    clarify -> []   (same — no call emitted)
    unparseable / unknown action -> []  (scored as no-call, i.e. a miss on positive categories)
    """
    env = parse_model_output(raw_output)
    if not env or env.get("action") != "call":
        return []
    decoded: List[Dict[str, Any]] = []
    for call in env.get("calls", []) or []:
        name = call.get("name")
        if not name:
            continue
        decoded.append({name: dict(call.get("arguments") or {})})
    return decoded


def is_no_call(raw_output: str) -> bool:
    """True when the model correctly declined to call a tool (refuse/clarify or an empty call list).
    BFCL's irrelevance category is scored on exactly this predicate.
    """
    return len(translate(raw_output)) == 0
