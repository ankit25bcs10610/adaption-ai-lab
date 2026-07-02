"""Bridge our test set to the OFFICIAL Berkeley Function-Calling Leaderboard harness.

Judges trust numbers from `pip install bfcl-eval` far more than a hand-rolled scorer. Our eval_bfcl.py
mirrors BFCL's structure for fast iteration, but for the final model-card table you should ALSO report
official numbers. This script exports our examples into BFCL's prompt schema so you can cross-check.

BFCL entry shape (simplified): {"id", "question": [[{"role","content"}]], "function": [ {schema}, ... ]}
with a parallel possible-answers file. Multi-turn uses nested question turns.

Usage:
  python -m src.export_bfcl --data data/out/test.jsonl --out results/bfcl_export.jsonl
Then run the official harness per https://github.com/ShishirPatil/gorilla (bfcl generate / bfcl evaluate).
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def to_bfcl(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
    turns: List[List[Dict[str, str]]] = []
    for turn in example.get("history", []) or []:
        turns.append([{"role": turn.get("role", "user"), "content": turn.get("content", "")}])
    turns.append([{"role": "user", "content": example["query"]}])

    functions = [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }
        for t in example["tools"]
    ]
    ans = example["answer"]
    if ans["type"] == "tool_call":
        possible = [{c["name"]: c.get("arguments", {})} for c in ans["calls"]]
    else:
        # BFCL relevance/irrelevance: no function should be called
        possible = []
    return {
        "id": f"autoscientist_{idx}",
        "question": turns,
        "function": functions,
        "ground_truth": possible,
        "category": ans["type"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out", default="results/bfcl_export.jsonl")
    args = ap.parse_args()

    import os

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(rows):
            f.write(json.dumps(to_bfcl(ex, i), ensure_ascii=False) + "\n")
    print(f"[bfcl] exported {len(rows)} examples -> {args.out}")
    print("Run the official harness: https://github.com/ShishirPatil/gorilla (pip install bfcl-eval)")


if __name__ == "__main__":
    main()
