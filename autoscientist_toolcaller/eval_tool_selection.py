"""Tool-selection-under-many-tools eval — retrieval recall@k + accuracy vs #tools (advanced).

Pads each example's tool list with distractor tools from the pool (simulating a large tool registry),
then measures (a) retrieval recall@k — did the gold tool survive TF-IDF retrieval — and (b) end-to-end
accuracy with the FULL padded context vs. only the RETRIEVED top-k. The gap is the degradation real
systems hit as the tool set grows. Deterministic + offline (recall is correct-by-construction).

CLI: python -m autoscientist_toolcaller.eval_tool_selection --model <id> --data data/out/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Callable, Dict, List

from .eval_harness import judge
from .format_utils import build_eval_prompt
from .tool_retrieval import ToolRetriever


def _gold_tool_names(rec: Dict[str, Any]) -> set:
    ans = rec.get("answer", {}) or {}
    if ans.get("type") == "tool_call":
        return {c.get("name") for c in ans.get("calls", [])}
    return set()


def pad_with_distractors(rec: Dict[str, Any], pool: List[Dict[str, Any]], m: int, rng: random.Random) -> List[Dict[str, Any]]:
    """Return the example's tools + up to m distractor tools from `pool` (no name collisions), shuffled."""
    seen = {t["name"] for t in rec["tools"]}
    distractors: List[Dict[str, Any]] = []
    shuffled = pool[:]
    rng.shuffle(shuffled)
    for t in shuffled:
        if len(distractors) >= m:
            break
        if t.get("name") in seen:
            continue
        seen.add(t["name"])
        distractors.append(t)
    tools = rec["tools"] + distractors
    rng.shuffle(tools)
    return tools


def retrieval_recall(records: List[Dict[str, Any]], pool: List[Dict[str, Any]],
                     k: int = 5, m: int = 20, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    hits = total = 0
    for rec in records:
        gold = _gold_tool_names(rec)
        if not gold:
            continue
        tools = pad_with_distractors(rec, pool, m, rng)
        top = {t["name"] for t in ToolRetriever(tools).retrieve(rec["query"], k)}
        total += 1
        if gold <= top:  # all gold tools survived retrieval
            hits += 1
    return {"recall_at_k": (hits / total) if total else 0.0, "k": k, "m": m, "n": total}


def evaluate_tool_selection(records: List[Dict[str, Any]], generate_fn: Callable[[str], str],
                            pool: List[Dict[str, Any]], k: int = 5, m: int = 20, seed: int = 42) -> Dict[str, Any]:
    rng = random.Random(seed)
    full_bits: List[int] = []
    retr_bits: List[int] = []
    for rec in records:
        tools = pad_with_distractors(rec, pool, m, rng)
        ex_full = {**rec, "tools": tools}
        full_bits.append(int(judge(ex_full, generate_fn(build_eval_prompt(ex_full)))["correct"]))
        top = ToolRetriever(tools).retrieve(rec["query"], k)
        ex_r = {**rec, "tools": top}
        retr_bits.append(int(judge(ex_r, generate_fn(build_eval_prompt(ex_r)))["correct"]))

    def acc(b: List[int]) -> float:
        return sum(b) / len(b) if b else 0.0

    rec_k = retrieval_recall(records, pool, k, m, seed)
    return {
        "n": len(records), "k": k, "distractors": m,
        "recall_at_k": rec_k["recall_at_k"],
        "accuracy_full_context": acc(full_bits),
        "accuracy_retrieved_topk": acc(retr_bits),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out", default="results/eval_tool_selection.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--distractors", type=int, default=20)
    args = ap.parse_args()
    import os
    from .build_dataset import harvest_tool_pool
    from .eval_harness import hf_generate_fn, load_jsonl

    records = load_jsonl(args.data)
    pool = harvest_tool_pool(records)
    gen = hf_generate_fn(args.model, 512, 0.0, args.adapter)
    metrics = evaluate_tool_selection(records, gen, pool, k=args.k, m=args.distractors)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
