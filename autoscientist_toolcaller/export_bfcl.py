"""Bridge our test set to the OFFICIAL Berkeley Function-Calling Leaderboard (BFCL v4) harness.

Judges trust numbers from `pip install bfcl-eval` far more than a hand-rolled scorer. `eval_bfcl.py`
mirrors BFCL's structure for fast local iteration; for the final model-card table you should ALSO
report an official cross-check. This script exports our examples into BFCL's on-disk format.

BFCL uses **two parallel files** per test category (not one):
  - a PROMPT file:  {"id", "question": [[{role,content}, ...], ...], "function": [ {schema}, ... ]}
  - an ANSWER file: {"id", "ground_truth": [ {func_name: {arg: [acceptable, values]}}, ... ]}
`question` is a list of *turns*, each turn a list of messages (single-turn = one turn, one message).
Every argument's value is a **list of acceptable values** — the AST checker accepts any of them.

IMPORTANT — envelope translation. Our model emits a single JSON envelope
`{"action": "call"|"refuse"|"clarify", "calls": [...]}`, which is NOT BFCL's expected function-call
format. To score with the official harness you must register a model handler that maps our envelope to
BFCL's decoded-AST form: `action=="call"` -> the list of `{name: {arg: value}}` calls;
`action in ("refuse","clarify")` -> an empty call list (relevance/irrelevance). Without that handler
the official run scores ~0 regardless of model quality. A reference handler lives in
`autoscientist_toolcaller/bfcl_handler.py` (translation only; wire it into your local bfcl-eval checkout).

Usage:
  python -m autoscientist_toolcaller.export_bfcl --data data/out/test.jsonl --out-dir results/bfcl
Produces results/bfcl/BFCL_v4_<category>.json (prompts) and .._answer.json (ground truth).
Then run the official harness per https://github.com/ShishirPatil/gorilla.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

# Map our internal category (from eval_bfcl.categorize) to a BFCL v4 test-category slug.
_CATEGORY_MAP = {
    "simple": "simple",
    "multiple": "multiple",
    "parallel": "parallel",
    "parallel_multiple": "parallel_multiple",
    "irrelevance": "irrelevance",
    "multi_turn": "multi_turn_base",
}


def _functions(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in example["tools"]
    ]


def _question_turns(example: Dict[str, Any]) -> List[List[Dict[str, str]]]:
    """Build BFCL's list-of-turns. Each user message opens a new turn; assistant/tool messages
    attach to the current turn. Single-turn examples collapse to one turn with one user message.
    """
    turns: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    for msg in (example.get("history") or []):
        role = msg.get("role", "user")
        if role == "user" and current:
            turns.append(current)
            current = []
        current.append({"role": role, "content": msg.get("content", "")})
    if current:
        turns.append(current)
    turns.append([{"role": "user", "content": example["query"]}])
    return turns


def _ground_truth(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """BFCL possible-answer: a list of {func_name: {arg: [acceptable values]}}. Each value is wrapped
    in a single-element list (our labels are exact; the checker accepts any element). Refuse/clarify
    (irrelevance) expect NO call, so ground_truth is empty.
    """
    ans = example["answer"]
    if ans.get("type") != "tool_call":
        return []
    out = []
    for c in ans.get("calls", []):
        args = {k: [v] for k, v in (c.get("arguments") or {}).items()}
        out.append({c["name"]: args})
    return out


def to_bfcl_prompt(example: Dict[str, Any], entry_id: str) -> Dict[str, Any]:
    return {"id": entry_id, "question": _question_turns(example), "function": _functions(example)}


def to_bfcl_answer(example: Dict[str, Any], entry_id: str) -> Dict[str, Any]:
    return {"id": entry_id, "ground_truth": _ground_truth(example)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out-dir", default="results/bfcl")
    args = ap.parse_args()

    # Reuse the exact category logic the local scorer uses, for consistency.
    from .eval_bfcl import categorize

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    by_cat_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_cat_answer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_cat_idx: Dict[str, int] = defaultdict(int)

    for ex in rows:
        internal = categorize(ex)
        if (ex.get("history")):
            internal = "multi_turn"
        cat = _CATEGORY_MAP.get(internal, "simple")
        i = per_cat_idx[cat]
        per_cat_idx[cat] += 1
        entry_id = f"BFCL_v4_{cat}_{i}"
        by_cat_prompt[cat].append(to_bfcl_prompt(ex, entry_id))
        by_cat_answer[cat].append(to_bfcl_answer(ex, entry_id))

    for cat, prompts in by_cat_prompt.items():
        p_path = os.path.join(args.out_dir, f"BFCL_v4_{cat}.json")
        a_path = os.path.join(args.out_dir, f"BFCL_v4_{cat}_answer.json")
        with open(p_path, "w", encoding="utf-8") as f:
            for row in prompts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(a_path, "w", encoding="utf-8") as f:
            for row in by_cat_answer[cat]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[bfcl] {cat}: {len(prompts)} -> {p_path} (+ answer file)")

    print(
        f"[bfcl] exported {len(rows)} examples across {len(by_cat_prompt)} categories -> {args.out_dir}\n"
        "Register the envelope handler (autoscientist_toolcaller/bfcl_handler.py) in your bfcl-eval checkout, then run the "
        "official harness: https://github.com/ShishirPatil/gorilla (pip install bfcl-eval)."
    )


if __name__ == "__main__":
    main()
