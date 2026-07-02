"""BFCL-aligned evaluation: category breakdown + lenient (AST-style) argument matching.

Adaption's hidden test set is unknown, but the Berkeley Function-Calling Leaderboard (BFCL) is the
de-facto standard for this domain, so mirroring its structure is our best proxy. This module adds two
things the base harness (eval_harness.py) doesn't:

  1. CATEGORY BREAKDOWN, BFCL-style:
       simple       -> one tool available, one expected call
       multiple     -> several tools available, one correct call (tool-selection test)
       parallel     -> one query, multiple expected calls
       irrelevance  -> no applicable tool (must refuse)   [our no_tool hard negative]
       clarify      -> missing arg / ambiguous (must clarify) [our other hard negatives]

  2. LENIENT MATCHING: BFCL allows multiple acceptable argument values. We normalize values
     (case/whitespace/number/boolean/known aliases) and accept a gold `acceptable` alternatives list
     when present, so semantically-correct-but-differently-formatted calls aren't unfairly failed.

Report BOTH strict (from eval_harness) and lenient numbers in the model card — it reads as honest.

CLI: python -m src.eval_bfcl --model <id> [--adapter path] --data data/out/test.jsonl \
       --out results/eval_bfcl.json [--strict]
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List

from .format_utils import build_eval_prompt, parse_model_output
from .schema_validator import validate_call

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


# --------------------------------------------------------------------------------------
# Categorization
# --------------------------------------------------------------------------------------
def categorize(example: Dict[str, Any]) -> str:
    # multi-turn (BFCL v3/v4's highest-weight AST bucket) takes precedence
    if example.get("history"):
        return "multi_turn"
    ans = example["answer"]
    if ans["type"] == "refuse":
        return "irrelevance"
    if ans["type"] == "clarify":
        return "clarify"
    calls = ans.get("calls", [])
    if len(calls) > 1:
        return "parallel"
    return "multiple" if len(example.get("tools", [])) > 1 else "simple"


# --------------------------------------------------------------------------------------
# Lenient value normalization
# --------------------------------------------------------------------------------------
def normalize_value(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if _NUM_RE.match(s):
            return float(s)
        if s in ("true", "false"):
            return s == "true"
        # collapse internal whitespace and strip trailing punctuation
        return re.sub(r"\s+", " ", s).rstrip(".!?")
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in v.items()}
    return v


def args_match_lenient(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    if set(pred.keys()) != set(gold.keys()):
        return False
    for k in gold:
        gv = gold[k]
        # gold value may be a list of acceptable alternatives wrapped as {"_acceptable": [...]}
        if isinstance(gv, dict) and "_acceptable" in gv:
            accepted = [normalize_value(a) for a in gv["_acceptable"]]
            if normalize_value(pred[k]) not in accepted:
                return False
        elif normalize_value(pred[k]) != normalize_value(gv):
            return False
    return True


# --------------------------------------------------------------------------------------
# Judging (lenient by default; set lenient=False to fall back to strict equality)
# --------------------------------------------------------------------------------------
def judge_bfcl(example: Dict[str, Any], output_text: str, lenient: bool = True) -> Dict[str, Any]:
    tools = example["tools"]
    gold = example["answer"]
    gold_type = gold["type"]
    parsed = parse_model_output(output_text)
    res = {
        "category": categorize(example),
        "parsed_ok": parsed is not None,
        "correct": False,
        "hallucinated_call": False,
    }
    if parsed is None:
        return res
    action = parsed.get("action")

    if gold_type == "tool_call":
        if action != "call":
            return res
        pred_calls = parsed.get("calls") or []
        gold_calls = gold["calls"]
        if len(pred_calls) != len(gold_calls):
            return res
        remaining = list(gold_calls)
        for pc in pred_calls:
            ok, _ = validate_call(pc, tools)
            if not ok:
                return res
            matcher = args_match_lenient if lenient else (lambda a, b: a == b)
            hit = next(
                (g for g in remaining
                 if g["name"] == pc.get("name")
                 and matcher(pc.get("arguments", {}), g.get("arguments", {}))),
                None,
            )
            if hit is None:
                return res
            remaining.remove(hit)
        res["correct"] = not remaining
        return res

    # hard negative
    if action == "call":
        res["hallucinated_call"] = True
        return res
    res["correct"] = action == ("refuse" if gold_type == "refuse" else "clarify")
    return res


def evaluate_bfcl(
    records: List[Dict[str, Any]],
    generate_fn,
    lenient: bool = True,
) -> Dict[str, Any]:
    from .eval_harness import _bootstrap_se

    cats = ["simple", "multiple", "parallel", "multi_turn", "irrelevance", "clarify"]
    buckets: Dict[str, List[Dict[str, Any]]] = {c: [] for c in cats}
    all_bits: List[int] = []
    hard_halluc = 0
    hard_total = 0

    for ex in records:
        prompt = build_eval_prompt(ex)
        v = judge_bfcl(ex, generate_fn(prompt), lenient=lenient)
        buckets[v["category"]].append(v)
        all_bits.append(int(v["correct"]))
        if v["category"] in ("irrelevance", "clarify"):
            hard_total += 1
            hard_halluc += int(v["hallucinated_call"])

    per_cat = {
        c: {
            "n": len(b),
            "accuracy": (sum(x["correct"] for x in b) / len(b)) if b else None,
        }
        for c, b in buckets.items()
    }
    return {
        "mode": "lenient" if lenient else "strict",
        "n": len(records),
        "overall_accuracy": (sum(all_bits) / len(all_bits)) if all_bits else 0.0,
        "overall_stderr": _bootstrap_se(all_bits),
        "hallucination_rate": (hard_halluc / hard_total) if hard_total else 0.0,
        "by_category": per_cat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out", default="results/eval_bfcl.json")
    ap.add_argument("--strict", action="store_true", help="use strict equality instead of lenient")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    import os
    from .eval_harness import hf_generate_fn, load_jsonl

    records = load_jsonl(args.data)
    gen = hf_generate_fn(args.model, args.max_new_tokens, args.temperature, args.adapter)
    metrics = evaluate_bfcl(records, gen, lenient=not args.strict)
    metrics["model"], metrics["adapter"] = args.model, args.adapter
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
