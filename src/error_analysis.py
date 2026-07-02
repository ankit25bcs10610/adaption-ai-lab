"""Error analysis — the engine of the data-centric iteration loop.

Runs the eval capturing every model output, then groups FAILURES by BFCL category and hard-negative kind
so you can see *what* the model gets wrong and fix the data accordingly. Writes:
  results/errors.jsonl  -- every failing case (example + output + why)
  results/errors.md     -- a human-readable breakdown with a few examples per bucket

Loop: build -> train -> error_analysis -> patch data -> retrain.

Usage:
  python -m src.error_analysis --model <id> [--adapter path] --data data/out/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .eval_bfcl import categorize, judge_bfcl
from .format_utils import build_eval_prompt, parse_model_output

_GOLD_ACTION = {"tool_call": "call", "refuse": "refuse", "clarify": "clarify"}


def _pred_action(output_text: str) -> str:
    parsed = parse_model_output(output_text)
    if parsed is None:
        return "parse_fail"
    action = parsed.get("action")
    return action if action in ("call", "refuse", "clarify") else "parse_fail"


def analyze(records: List[Dict[str, Any]], generate_fn, lenient: bool = True) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    by_cat_total = Counter()
    by_cat_fail = Counter()
    fail_reason = Counter()

    for ex in records:
        cat = categorize(ex)
        by_cat_total[cat] += 1
        prompt = build_eval_prompt(ex)
        out = generate_fn(prompt)
        v = judge_bfcl(ex, out, lenient=lenient)
        predictions.append(
            {
                "gold": _GOLD_ACTION.get(ex["answer"]["type"], "call"),
                "pred": _pred_action(out),
                "category": cat,
                "correct": v["correct"],
            }
        )
        if not v["correct"]:
            by_cat_fail[cat] += 1
            reason = (
                "parse_failure" if not v["parsed_ok"]
                else "hallucinated_call" if v["hallucinated_call"]
                else "wrong_call_or_args"
            )
            fail_reason[reason] += 1
            failures.append(
                {
                    "category": cat,
                    "hn_kind": ex["meta"].get("hn_kind"),
                    "reason": reason,
                    "query": ex["query"],
                    "gold": ex["answer"],
                    "output": out,
                }
            )

    return {
        "n": len(records),
        "n_failures": len(failures),
        "by_category": {c: {"total": by_cat_total[c], "fail": by_cat_fail[c]} for c in by_cat_total},
        "fail_reasons": dict(fail_reason),
        "failures": failures,
        "predictions": predictions,
    }


def write_reports(result: Dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    jl = os.path.join(out_dir, "errors.jsonl")
    with open(jl, "w", encoding="utf-8") as f:
        for row in result["failures"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # predictions.jsonl feeds the confusion matrix in src.eval_report
    preds = os.path.join(out_dir, "predictions.jsonl")
    with open(preds, "w", encoding="utf-8") as f:
        for row in result.get("predictions", []):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    md = [f"# Error analysis\n",
          f"- Total: {result['n']}  |  Failures: {result['n_failures']} "
          f"({result['n_failures'] / max(result['n'], 1):.1%})\n",
          "## By category\n",
          "| Category | Fail / Total | Fail rate |", "|---|---|---|"]
    for c, d in sorted(result["by_category"].items()):
        rate = d["fail"] / d["total"] if d["total"] else 0.0
        md.append(f"| {c} | {d['fail']} / {d['total']} | {rate:.1%} |")
    md += ["\n## Failure reasons\n", "| Reason | Count |", "|---|---|"]
    for r, n in sorted(result["fail_reasons"].items(), key=lambda x: -x[1]):
        md.append(f"| {r} | {n} |")

    # a few example failures per category (for eyeballing)
    md.append("\n## Sample failures\n")
    per_cat = defaultdict(list)
    for row in result["failures"]:
        if len(per_cat[row["category"]]) < 3:
            per_cat[row["category"]].append(row)
    for cat, rows in per_cat.items():
        md.append(f"### {cat}")
        for row in rows:
            md.append(f"- **query:** {row['query']}")
            md.append(f"  - reason: `{row['reason']}`")
            md.append(f"  - gold: `{json.dumps(row['gold'], ensure_ascii=False)[:200]}`")
            md.append(f"  - output: `{row['output'][:200]}`")
    mdp = os.path.join(out_dir, "errors.md")
    open(mdp, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"[error-analysis] wrote {jl} and {mdp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    from .eval_harness import hf_generate_fn, load_jsonl

    records = load_jsonl(args.data)
    gen = hf_generate_fn(args.model, args.max_new_tokens, args.temperature, args.adapter)
    result = analyze(records, gen, lenient=not args.strict)
    write_reports(result, args.out_dir)
    print(json.dumps({k: result[k] for k in ("n", "n_failures", "by_category", "fail_reasons")}, indent=2))


if __name__ == "__main__":
    main()
