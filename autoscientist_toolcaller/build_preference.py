"""Build DPO preference pairs that train the exact behavior our submission is about.

For each example we emit (prompt, chosen, rejected) where:
  hard negatives (refuse/clarify):  chosen = the correct refuse/clarify envelope
                                     rejected = a PLAUSIBLE hallucinated tool call (the mistake baselines make)
  positives (tool_call):            chosen = the correct call
                                     rejected = a corrupted call (wrong tool, or a required arg dropped)

The rejected samples are the failure modes we want the model to stop doing, so DPO after SFT pushes
directly on the moat (Berkeley's irrelevance category). Fully deterministic given the seed.

Output: data/out/pref.jsonl with columns {prompt, chosen, rejected} (TRL DPO format).
Run: python -m autoscientist_toolcaller.build_preference --config config.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any, Dict, List, Optional

import yaml

from .format_utils import answer_to_target, build_system_prompt, target_to_json_str


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


def _required_of(example: Dict[str, Any], tool_name: str) -> List[str]:
    tool = next((t for t in example["tools"] if t.get("name") == tool_name), None)
    if not tool:
        return []
    return list((tool.get("parameters") or {}).get("required", []))


def _corrupt_positive(example: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Corrupt a correct call into a *provably* wrong one: swap to another tool, or drop a REQUIRED
    arg. We never drop a merely-optional arg — that can leave the call still-correct (a poison pair).
    """
    calls = copy.deepcopy(example["answer"]["calls"])
    if not calls:
        return None
    c = calls[0]
    required = _required_of(example, c.get("name", ""))
    present_required = [k for k in required if k in (c.get("arguments") or {})]
    modes = []
    others = [t for t in example["tools"] if t["name"] != c["name"]]
    if others:
        modes.append("swap_tool")
    if present_required:
        modes.append("drop_required")
    if len(calls) >= 2:
        modes.append("drop_call")  # parallel completeness: emit one fewer call than required
    if not modes:
        return None
    mode = rng.choice(modes)
    if mode == "drop_call":
        calls.pop(rng.randrange(len(calls)))
    elif mode == "swap_tool":
        c["name"] = rng.choice(others)["name"]
    else:  # drop_required -> the call is now missing a required arg (schema-invalid)
        del c["arguments"][rng.choice(present_required)]
    return {"action": "call", "calls": calls}


def _axis_pairs(example: Dict[str, Any], rng: random.Random) -> List[Dict[str, str]]:
    """Extra preference pairs that target specific moat axes, guaranteed (not left to rejection
    sampling): over-refusal (chosen=call vs rejected=refuse) and partial-parallel (both calls vs one)."""
    ans = example["answer"]
    if ans["type"] != "tool_call":
        return []
    meta = example.get("meta", {}) or {}
    hk = meta.get("hn_kind")
    chosen = target_to_json_str(ans)
    out: List[Dict[str, str]] = []
    if hk == "over_refusal":
        rej = {"action": "refuse", "message": "I don't think any available tool can do that."}
        if _confirmed_wrong(example, rej):
            out.append({"prompt": _prompt(example), "chosen": chosen, "rejected": _canon(rej)})
    if hk == "partial_parallel" and len(ans.get("calls") or []) >= 2:
        calls = copy.deepcopy(ans["calls"])
        calls.pop(rng.randrange(len(calls)))
        rej = {"action": "call", "calls": calls}
        if _confirmed_wrong(example, rej):
            out.append({"prompt": _prompt(example), "chosen": chosen, "rejected": _canon(rej)})
    return out


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _confirmed_wrong(example: Dict[str, Any], rejected_obj: Dict[str, Any]) -> bool:
    """Gate against poison pairs: a `rejected` is usable only if it is provably NOT the correct
    answer. If the correct behavior is refuse/clarify, any tool call is wrong. For a positive, the
    rejected must differ from gold by tool name, a missing required arg, or a changed required value.
    """
    ans = example["answer"]
    if ans["type"] in ("refuse", "clarify"):
        return rejected_obj.get("action") == "call" and bool(rejected_obj.get("calls"))
    # gold is a tool_call:
    if rejected_obj.get("action") in ("refuse", "clarify"):
        return True  # abstaining when a call is required is wrong (the over-refusal negative)
    gold_calls = ans.get("calls") or []
    rej_calls = rejected_obj.get("calls") or []
    if _canon(rejected_obj) == _canon({"action": "call", "calls": gold_calls}):
        return False  # identical to gold -> poison
    if len(rej_calls) != len(gold_calls):
        return True  # wrong number of calls (dropped/added a call — the partial-parallel negative)
    rc = (rej_calls or [{}])[0]
    gc = (gold_calls or [{}])[0]
    if rc.get("name") != gc.get("name"):
        return True  # different tool
    required = _required_of(example, rc.get("name", ""))
    rargs = rc.get("arguments") or {}
    gargs = gc.get("arguments") or {}
    if any(k not in rargs for k in required):
        return True  # dropped a required arg
    return rargs != gargs  # required args present but a value changed


def _hardness(chosen_obj: Dict[str, Any], cand_obj: Dict[str, Any]) -> int:
    """Structural distance from chosen to a rejected candidate — SMALLER = a harder near-miss.

    A near-miss (one changed required value) teaches a sharper preference boundary than an obviously
    wrong tool swap. Pure + deterministic so selection is reproducible and unit-testable.
    """
    cc = (chosen_obj.get("calls") or [{}])[0]
    rc = (cand_obj.get("calls") or [{}])[0]
    dist = 0
    if cc.get("name") != rc.get("name"):
        dist += 10  # a tool-name change is a large, easy-to-spot difference
    ca, ra = cc.get("arguments") or {}, rc.get("arguments") or {}
    for k in set(ca) | set(ra):
        if ca.get(k) != ra.get(k):
            dist += 1
    return dist


def build_pairs(examples: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    pairs: List[Dict[str, str]] = []
    skipped = 0
    for ex in examples:
        ans = ex["answer"]
        chosen = target_to_json_str(ans)
        chosen_obj = answer_to_target(ans)
        # Rejection sampling: draw 4 candidates, keep ALL confirmed-wrong, then pick the HARDEST
        # (smallest structural distance to chosen) so DPO trains on the sharpest near-miss.
        cands = []
        for _ in range(4):
            cand = (
                _plausible_hallucinated_call(ex, rng)
                if ans["type"] in ("refuse", "clarify")
                else _corrupt_positive(ex, rng)
            )
            if cand is not None and _confirmed_wrong(ex, cand):
                cands.append(cand)
        if not cands:
            skipped += 1
            continue
        rejected_obj = min(
            cands, key=lambda c: (_hardness(chosen_obj, c), json.dumps(c, sort_keys=True))
        )
        pairs.append(
            {
                "prompt": _prompt(ex),
                "chosen": chosen,
                "rejected": json.dumps(rejected_obj, sort_keys=True, ensure_ascii=False),
            }
        )
        pairs += _axis_pairs(ex, rng)  # extra over-refusal / partial-parallel contrasts
    if skipped:
        print(f"[pref] skipped {skipped} examples (no confirmed-wrong negative — poison guard)")
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
    n_sft = len(pairs)

    # Merge the execution-labeled env DPO pairs — the repo's single highest-quality preference signal
    # (chosen = checker-verified call, rejected = checker-PROVEN-wrong). Previously generated but never
    # consumed. Generate them on demand if configured and not yet materialized.
    n_env = 0
    env_dpo_n = cfg["dataset"].get("env_dpo", 0)
    if env_dpo_n:
        env_path = os.path.join(out_dir, "env_dpo.jsonl")
        if not os.path.exists(env_path):
            from .envs import generate_dpo
            with open(env_path, "w", encoding="utf-8") as f:
                for d in generate_dpo(env_dpo_n, seed=cfg["seed"]):
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        env_pairs = [json.loads(l) for l in open(env_path, encoding="utf-8") if l.strip()]
        pairs += env_pairs
        n_env = len(env_pairs)

    # Merge agentic trajectory-STEP DPO pairs (chosen = gold step, rejected = checker-proven-wrong
    # next call on that step's state) — the same execution-labeled quality as env_dpo, but multi-step.
    n_ag = 0
    ag_dpo_n = cfg["dataset"].get("agentic_dpo", 0)
    if ag_dpo_n:
        ag_path = os.path.join(out_dir, "agentic_dpo.jsonl")
        if not os.path.exists(ag_path):
            from .agentic import generate_dpo as _ag_dpo
            with open(ag_path, "w", encoding="utf-8") as f:
                for d in _ag_dpo(ag_dpo_n, seed=cfg["seed"]):
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        ag_pairs = [json.loads(l) for l in open(ag_path, encoding="utf-8") if l.strip()]
        pairs += ag_pairs
        n_ag = len(ag_pairs)

    out = os.path.join(out_dir, "pref.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_hn = sum(1 for e in examples if e["answer"]["type"] in ("refuse", "clarify"))
    print(f"[pref] wrote {len(pairs)} pairs -> {out} "
          f"({n_sft} SFT-derived incl. {n_hn} hard-neg, {n_env} execution-labeled env, {n_ag} agentic-step)")

    # Refresh the reproducibility manifest so pref.jsonl's hash is current. build_dataset writes the
    # manifest BEFORE pref.jsonl exists, so without this the manifest carries a stale pref hash and
    # release preflight fails ("artifact changed"). Refreshing here keeps the pref build self-consistent
    # (this was the exact drift that once blocked every upload).
    try:
        from . import manifest as _manifest
        results_dir = cfg.get("paths", {}).get("results_dir", "results")
        _manifest.write(out_dir=out_dir, config_path=args.config,
                        manifest_path=os.path.join(results_dir, "manifest.json"))
        print(f"[pref] refreshed {os.path.join(results_dir, 'manifest.json')} (pref.jsonl hash now current)")
    except Exception as e:
        print(f"[pref] manifest refresh skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
