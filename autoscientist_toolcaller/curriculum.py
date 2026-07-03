"""Difficulty tagging + curriculum ordering — a criterion-4 depth artifact.

Scores each example's difficulty (answer type, tool count, required-arg count, slice) and can emit a
curriculum-ordered `train_pc` (easy → hard) plus a difficulty histogram for the model card. Pure stdlib,
deterministic, offline.

CLI:
  python -m autoscientist_toolcaller.curriculum --data data/out/train.jsonl \
      --out-pc data/out/train_pc_curriculum.jsonl --hist results/difficulty.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Dict, List

from .format_utils import to_prompt_completion


def difficulty(ex: Dict[str, Any]) -> int:
    """A 1..~10 difficulty score — higher = harder decision / more to get right."""
    ans = ex.get("answer", {}) or {}
    meta = ex.get("meta", {}) or {}
    tools = ex.get("tools", []) or []
    t = ans.get("type")
    score = 1
    # abstention is harder to learn than a plain call
    if t == "refuse":
        score += 2
    elif t == "clarify":
        score += 2
    # selection under load: more tools on offer = harder
    score += min(len(tools) // 2, 3)
    if t == "tool_call":
        calls = ans.get("calls", [])
        score += max(len(calls) - 1, 0) * 2  # multi-call completeness
        req = sum(len((c.get("arguments") or {})) for c in calls)
        score += min(req, 3)
    # slice-specific difficulty
    if meta.get("sd_kind"):
        score += 2  # schema drift
    if meta.get("hn_kind") in ("over_refusal", "partial_parallel"):
        score += 2
    if meta.get("mt_kind") or ex.get("history"):
        score += 1  # multi-turn context
    if (meta.get("lang") or "en") != "en":
        score += 1  # non-English query
    return score


def band(score: int) -> str:
    return "easy" if score <= 3 else "medium" if score <= 6 else "hard"


def tag(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach meta.difficulty + meta.difficulty_band (returns the same list, mutated)."""
    for ex in examples:
        d = difficulty(ex)
        ex.setdefault("meta", {})["difficulty"] = d
        ex["meta"]["difficulty_band"] = band(d)
    return examples


def histogram(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [difficulty(ex) for ex in examples]
    bands = Counter(band(s) for s in scores)
    return {
        "n": len(scores),
        "mean": round(sum(scores) / len(scores), 2) if scores else 0.0,
        "min": min(scores) if scores else 0,
        "max": max(scores) if scores else 0,
        "by_band": dict(bands),
        "by_score": dict(sorted(Counter(scores).items())),
    }


def curriculum_order(examples: List[Dict[str, Any]]) -> List[int]:
    """Return indices ordering examples easy → hard (stable)."""
    return sorted(range(len(examples)), key=lambda i: (difficulty(examples[i]), i))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/train.jsonl")
    ap.add_argument("--out-pc", default="data/out/train_pc_curriculum.jsonl")
    ap.add_argument("--hist", default="results/difficulty.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    order = curriculum_order(rows)
    pc = [to_prompt_completion(rows[i]) for i in order]
    os.makedirs(os.path.dirname(args.out_pc) or ".", exist_ok=True)
    with open(args.out_pc, "w", encoding="utf-8") as f:
        for r in pc:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = histogram(rows)
    os.makedirs(os.path.dirname(args.hist) or ".", exist_ok=True)
    json.dump(hist, open(args.hist, "w"), indent=2)
    print(f"[curriculum] {len(rows)} rows easy→hard -> {args.out_pc}")
    print(f"[curriculum] difficulty: {hist['by_band']} (mean {hist['mean']}) -> {args.hist}")


if __name__ == "__main__":
    main()
