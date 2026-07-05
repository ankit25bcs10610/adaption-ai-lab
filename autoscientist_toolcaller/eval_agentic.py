"""Agentic trajectory evaluation — free rollout success + per-step accuracy (advanced #1).

Given a model `generate_fn`, we ROLL OUT each trajectory: at every step we build the prompt from the
actual interaction so far, get the model's action, **apply it to the environment**, and feed the real
observation forward. We report:
  * trajectory_success_rate — did the rollout reach the goal state (and, for RECOVERY trajectories,
    correctly abstain on the impossible final step instead of blindly calling)?
  * per_step_accuracy       — fraction of individual steps whose action matched the gold step.

The environment (envs.apply) is the oracle, so this needs no labels beyond the generated trajectory.
Offline-testable: an oracle generate_fn (returns each gold action) scores 1.0; always-refuse scores 0.

CLI: python -m autoscientist_toolcaller.eval_agentic --model <id> --data data/out/agentic_trajectories.jsonl
"""
from __future__ import annotations

import argparse
import copy
import json
from typing import Any, Callable, Dict, List

from .agentic import _obs, env_by_name
from .eval_stats import bootstrap_ci
from .format_utils import build_eval_prompt, parse_model_output


def _step_matches(parsed: Dict[str, Any] | None, step: Dict[str, Any]) -> bool:
    """Did the model's action match this gold step (call name+args, or the abstention type)?"""
    if parsed is None:
        return False
    gold = step["gold_action"]
    action = parsed.get("action")
    if gold == "call":
        if action != "call":
            return False
        pc = (parsed.get("calls") or [{}])[0]
        gc = step["answer"]["calls"][0]
        return pc.get("name") == gc.get("name") and (pc.get("arguments") or {}) == (gc.get("arguments") or {})
    # clarify / refuse abstention step
    return action == gold


def rollout(traj: Dict[str, Any], generate_fn: Callable[[str], str]) -> Dict[str, Any]:
    """Free rollout: advance the env by the MODEL's own actions, observation-in-the-loop."""
    env = env_by_name(traj["env"])
    state = copy.deepcopy(traj["init_state"])
    hist: List[Dict[str, Any]] = []
    per_step: List[bool] = []
    for step in traj["steps"]:
        ex = {"tools": traj["tools"], "query": traj["goal"], "history": hist or None}
        out = generate_fn(build_eval_prompt(ex))
        parsed = parse_model_output(out)
        per_step.append(_step_matches(parsed, step))
        if parsed and parsed.get("action") == "call" and parsed.get("calls"):
            state, ok, reason = env.apply(state, parsed["calls"][0])
            obs = _obs(env, state, ok, reason)
        elif parsed and parsed.get("action") in ("refuse", "clarify"):
            obs = "OK — noted; awaiting your input."
        else:
            obs = "ERROR: unparseable action."
        hist.append({"role": "assistant", "content": out})
        hist.append({"role": "tool", "content": obs})

    reached = state == traj["final_state"]
    # RECOVERY success additionally requires the last (impossible) step was abstained, not called.
    success = reached and (traj.get("kind") != "recovery" or (bool(per_step) and per_step[-1]))
    return {"success": bool(success), "per_step": per_step, "steps": len(traj["steps"])}


def evaluate_agentic(trajectories: List[Dict[str, Any]], generate_fn: Callable[[str], str]) -> Dict[str, Any]:
    results = [rollout(t, generate_fn) for t in trajectories]
    succ = [int(r["success"]) for r in results]
    step_bits = [int(b) for r in results for b in r["per_step"]]
    ci = bootstrap_ci(succ)
    by_env: Dict[str, List[int]] = {}
    for t, r in zip(trajectories, results):
        by_env.setdefault(t["env"], []).append(int(r["success"]))
    return {
        "n": len(trajectories),
        "trajectory_success_rate": (sum(succ) / len(succ)) if succ else 0.0,
        "success_lo": ci["lo"], "success_hi": ci["hi"],
        "per_step_accuracy": (sum(step_bits) / len(step_bits)) if step_bits else 0.0,
        "avg_steps": (sum(r["steps"] for r in results) / len(results)) if results else 0.0,
        "by_env": {k: {"n": len(v), "success_rate": sum(v) / len(v)} for k, v in by_env.items()},
    }


def to_markdown(m: Dict[str, Any]) -> str:
    if not m.get("n"):
        return ""
    lines = [
        "### Agentic trajectory eval",
        "",
        f"- **Trajectory success:** {m['trajectory_success_rate']:.3f} "
        f"(95% CI {m['success_lo']:.3f}–{m['success_hi']:.3f}, n={m['n']})",
        f"- **Per-step accuracy:** {m['per_step_accuracy']:.3f}  ·  avg steps {m['avg_steps']:.2f}",
    ]
    for env, v in sorted(m["by_env"].items()):
        lines.append(f"  - {env}: {v['success_rate']:.3f} (n={v['n']})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/out/agentic_trajectories.jsonl")
    ap.add_argument("--out", default="results/eval_agentic.json")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()
    import os
    from .eval_harness import hf_generate_fn

    trajs = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    gen = hf_generate_fn(args.model, args.max_new_tokens, args.temperature, args.adapter)
    metrics = evaluate_agentic(trajs, gen)
    metrics["model"], metrics["adapter"] = args.model, args.adapter
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("\n" + to_markdown(metrics))


if __name__ == "__main__":
    main()
