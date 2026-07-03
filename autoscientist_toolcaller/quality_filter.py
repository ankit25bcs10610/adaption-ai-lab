"""Quality filtering for positive examples — the top data-quality lever (LIMA; Alpaca 52k->9k).

Two scorers:
  * heuristic_score  -- no-network, deterministic signals (default). Good enough to drop obvious junk.
  * llm_score_fn     -- optional LLM-as-judge via a user-supplied callable; higher fidelity.

`filter_examples` keeps the top `keep_frac` by score. Ties broken deterministically by index so runs
are reproducible.

The LLM judge is intentionally decoupled: pass any `score_fn(example)->float`. A ready-made prompt is in
JUDGE_PROMPT; wire it to Claude/any API in your own `score_fn`. We keep the default offline so the
pipeline runs without spending tokens, and you can turn the judge on when you want the extra lift.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .schema_validator import validate_answer

_WORD = re.compile(r"[a-z0-9]+")


JUDGE_PROMPT = """You are grading a single function-calling training example for quality.

Tools available (JSON Schema):
{tools}

User query:
{query}

Reference tool call(s):
{calls}

Score 0-10 on: (a) is the query realistic and unambiguous, (b) do the reference calls correctly and
completely answer it, (c) are all arguments present and plausible. Reply with ONLY a JSON object:
{{"score": <0-10>, "reason": "<short>"}}"""


def _tokens(s: str) -> List[str]:
    return _WORD.findall(s.lower())


def heuristic_score(example: Dict[str, Any]) -> float:
    """Cheap 0..1 quality proxy. Higher = keep."""
    tools = example.get("tools", [])
    query = example.get("query", "") or ""
    ans = example.get("answer", {})

    score = 0.0
    # 1. schema validity is table stakes
    ok, _ = validate_answer(ans, tools)
    if not ok:
        return 0.0
    score += 0.35

    # 2. query has reasonable length (not a fragment, not a wall of text)
    n = len(_tokens(query))
    if 4 <= n <= 60:
        score += 0.2
    elif n >= 3:
        score += 0.1

    # 3. arguments are non-empty and not placeholder-y
    calls = ans.get("calls", []) if ans.get("type") == "tool_call" else []
    if calls:
        vals = [v for c in calls for v in (c.get("arguments") or {}).values()]
        if vals and all(_nonplaceholder(v) for v in vals):
            score += 0.25
        elif vals:
            score += 0.1

    # 4. lexical grounding: query shares vocabulary with the chosen tool's name/description
    if calls:
        tool_by_name = {t["name"]: t for t in tools}
        qtok = set(_tokens(query))
        overlap = 0
        for c in calls:
            t = tool_by_name.get(c.get("name"))
            if t:
                ttok = set(_tokens(t.get("name", "") + " " + t.get("description", "")))
                overlap += len(qtok & ttok)
        if overlap >= 2:
            score += 0.2
        elif overlap == 1:
            score += 0.1

    return min(score, 1.0)


def _nonplaceholder(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        s = v.strip().lower()
        return s not in ("", "string", "value", "none", "null", "n/a", "todo", "xxx", "example")
    return True


def filter_examples(
    examples: List[Dict[str, Any]],
    keep_frac: float,
    score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Keep the top `keep_frac` of examples by score (deterministic)."""
    if keep_frac >= 1.0:
        return examples
    scorer = score_fn or heuristic_score
    scored = [(scorer(ex), i, ex) for i, ex in enumerate(examples)]
    # sort by score desc, then original index asc for stable ties
    scored.sort(key=lambda t: (-t[0], t[1]))
    k = max(1, int(len(examples) * keep_frac))
    kept = [ex for _, _, ex in scored[:k]]
    dropped = len(examples) - len(kept)
    print(f"[quality] kept {len(kept)}/{len(examples)} (dropped {dropped}) at keep_frac={keep_frac}")
    return kept


def make_llm_score_fn(call_llm: Callable[[str], str]) -> Callable[[Dict[str, Any]], float]:
    """Adapt a raw text-completion callable into a score_fn.

    `call_llm(prompt)->str` should return the judge's reply; we parse the JSON `score` (0-10 -> 0..1).
    Falls back to heuristic_score on parse failure so one bad judge reply can't zero out an example.
    """
    import json

    def _score(example: Dict[str, Any]) -> float:
        import json as _j

        calls = example["answer"].get("calls", []) if example["answer"].get("type") == "tool_call" else []
        prompt = JUDGE_PROMPT.format(
            tools=_j.dumps(example["tools"], indent=2),
            query=example["query"],
            calls=_j.dumps(calls, indent=2),
        )
        try:
            reply = call_llm(prompt)
            start = reply.find("{")
            obj = json.loads(reply[start : reply.rfind("}") + 1])
            return max(0.0, min(float(obj["score"]) / 10.0, 1.0))
        except Exception:
            return heuristic_score(example)

    return _score
