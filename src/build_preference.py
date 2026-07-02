"""Build DPO preference pairs that train the exact behavior our submission is about.

For each example we emit (prompt, chosen, rejected) where:
  hard negatives (refuse/clarify):  chosen = the correct refuse/clarify envelope
                                     rejected = a PLAUSIBLE hallucinated tool call (the mistake baselines make)
  positives (tool_call):            chosen = the correct call
                                     rejected = a corrupted call (wrong tool, or a required arg dropped)

The rejected samples are the failure modes we want the model to stop doing, so DPO after SFT pushes
directly on the moat (Berkeley's irrelevance category). Fully deterministic given the seed.

Output: data/out/pref.jsonl with columns {prompt, chosen, rejected} (TRL DPO format).
Run: python -m src.build_preference --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Optional

import yaml

from .format_utils import build_system_prompt, target_to_json_str


def _prompt(example: Dict[str, Any]) -> str:
    system = build_system_prompt(example["tools"])
    return f"{system}\n\nUser request:\n{example['query']}"


def _plausible_hallucinated_call(example: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    """A tool call that LOOKS reasonable but is wrong (the baseline's mistake on a hard negative)."""
    tools = example["tools"]
    tool = rng.choice(tools) if tools else {"name": "do_action", "parameters": {}}
    props = (tool.get("parameters") or {}).get("properties", {})
    required = (tool.get("parameters") or {}).get("required", list(props.keys()))
    args = {}
    for name in required:
        spec = props.get(name, {})
        args[name] = _fake_value(spec, name, rng)
    return {"action": "call", "calls": [{"name": tool["name"], "arguments": args}]}


def _fake_value(spec: Dict[str, Any], name: str, rng: random.Random) -> Any:
    t = spec.get("type", "string")
    if "enum" in spec and spec["enum"]:
        return rng.choice(spec["enum"])
    if t in ("integer", "number"):
        return rng.choice([1, 2, 5, 10, 42])
    if t == "boolean":
        return rng.choice([True, False])
    if t == "array":
        return []
    return {"city": "New York", "date": "2026-01-01", "query": "example"}.get(name, "example")


def _corrupt_positive(example: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Corrupt a correct call into a wrong one: drop a required arg, or swap to another tool."""
    calls = copy.deepcopy(example["answer"]["calls"])
    if not calls:
        return None
    mode = rng.choice(["drop_arg", "swap_tool"])
    c = calls[0]
    if mode == "drop_arg" and c.get("arguments"):
        keys = list(c["arguments"].keys())
        if keys:
            del c["arguments"][rng.choice(keys)]
            return {"action": "call", "calls": calls}
    # swap_tool: pick a different available tool
    others = [t for t in example["tools"] if t["name"] != c["name"]]
    if others:
        c["name"] = rng.choice(others)["name"]
        return {"action": "call", "calls": calls}
    # fallback to drop_arg if swap impossible
    if c.get("arguments"):
        keys = list(c["arguments"].keys())
        if keys:
            del c["arguments"][rng.choice(keys)]
            return {"action": "call", "calls": calls}
    return None


def build_pairs(examples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    for ex in examples:
        ans = ex["answer"]
        chosen = target_to_json_str(ans)
        if ans["type"] in ("refuse", "clarify"):
            rejected_obj = _plausible_hallucinated_call(ex, rng)
        else:
            rejected_obj = _corrupt_positive(ex, rng)
            if rejected_obj is None:
                continue
        pairs.append(
            {
                "prompt": _prompt(ex),
                "chosen": chosen,
                "rejected": json.dumps(rejected_obj, sort_keys=True, ensure_ascii=False),
            }
        )
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="train", help="which split to build pairs from")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["paths"]["out_dir"]

    src = os.path.join(out_dir, f"{args.split}.jsonl")
    examples = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    pairs = build_pairs(examples, seed=cfg["seed"])

    out = os.path.join(out_dir, "pref.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_hn = sum(1 for e in examples if e["answer"]["type"] in ("refuse", "clarify"))
    print(f"[pref] wrote {len(pairs)} pairs -> {out} ({n_hn} from hard negatives)")


if __name__ == "__main__":
    main()
