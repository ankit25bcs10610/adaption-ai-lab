"""Agentic multi-step tool-use trajectories — observation-in-the-loop rollouts (advanced #1).

`envs.py` verifies a SINGLE next call; here we build multi-STEP trajectories where each gold action
follows the tool OBSERVATION from the previous step, ending in a terminal action — plus a RECOVERY
variant where a requested step is impossible in the reached state, so the gold is to **clarify** rather
than blindly call. The environment (`envs.apply`) is the oracle, so every step is correct by
construction. Trajectories explode into standard multi-turn SFT examples (history = prior assistant
actions + tool observations), so training/eval reuse the existing format unchanged.

Deterministic + offline. CLI: python -m autoscientist_toolcaller.agentic --n 20 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from typing import Any, Dict, List, Optional

from .envs import CalendarEnv, CartEnv, FRUITS, TITLES, _intent_text, _valid_next_call, _perturb
from .format_utils import target_to_json_str

_ENV_BY_NAME = {"cart": CartEnv, "calendar": CalendarEnv}


def env_by_name(name: str):
    return _ENV_BY_NAME[name]()


def _obs(env, state: Dict[str, Any], ok: bool, reason: str) -> str:
    """Tool observation fed back to the model after an action."""
    if ok:
        return f"OK — {env.describe(state)}."
    return f"ERROR: {reason} — {env.describe(state)}."


def _impossible_call(env, state: Dict[str, Any], rng: random.Random):
    """A call that FAILS on `state`, plus a templated clarify message (for the recovery step)."""
    if isinstance(env, CartEnv):
        if state["items"]:
            item = rng.choice(list(state["items"]))
            have = state["items"][item]
            call = {"name": "remove_item", "arguments": {"item": item, "quantity": have + rng.randint(1, 5)}}
            msg = f"You only have {have} {item}; how many should I remove?"
        else:
            item = rng.choice(FRUITS)
            call = {"name": "remove_item", "arguments": {"item": item, "quantity": rng.randint(1, 3)}}
            msg = f"There are no {item} in the cart to remove — did you mean to add them?"
        return call, msg
    # calendar: cancel/move an event that doesn't exist
    absent = [t for t in TITLES if t not in state["events"]] or TITLES
    title = rng.choice(absent)
    call = {"name": "cancel_event", "arguments": {"title": title}}
    msg = f"There's no event titled {title} to cancel — which event did you mean?"
    return call, msg


def build_trajectory(env, rng: random.Random, traj_id: str) -> Optional[Dict[str, Any]]:
    """One agentic trajectory: 2–4 sequential verified calls (+ maybe a recovery clarify step)."""
    k = rng.randint(2, 4)
    plan: List[Dict[str, Any]] = []
    cur = env.blank()
    for _ in range(k):
        c = _valid_next_call(env, cur, rng)
        if c is None:
            break
        nxt, ok, _ = env.apply(cur, c)
        if not ok:
            break
        plan.append(c)
        cur = nxt
    if len(plan) < 2:
        return None

    recovery = rng.random() < 0.35
    intents = [_intent_text(env, c) for c in plan]
    bad_call = clarify_msg = None
    if recovery:
        bad_call, clarify_msg = _impossible_call(env, cur, rng)  # impossible in the FINAL reached state
        intents.append(_intent_text(env, bad_call))
    goal = " ".join(intents)

    steps: List[Dict[str, Any]] = []
    state = env.blank()
    hist: List[Dict[str, Any]] = []
    for c in plan:
        pre = copy.deepcopy(state)
        answer = {"type": "tool_call", "calls": [c]}
        steps.append({
            "history": copy.deepcopy(hist), "query": goal, "answer": answer,
            "gold_action": "call", "pre_state": pre,
        })
        nstate, ok, reason = env.apply(state, c)
        hist.append({"role": "assistant", "content": target_to_json_str(answer)})
        hist.append({"role": "tool", "content": _obs(env, nstate, ok, reason)})
        state = nstate
    final_state = copy.deepcopy(state)
    # Reject degenerate net-zero plans (final == blank): otherwise "do nothing" trivially "succeeds"
    # and trajectory-success stops discriminating.
    if final_state == env.blank():
        return None
    if recovery:
        steps.append({
            "history": copy.deepcopy(hist), "query": goal,
            "answer": {"type": "clarify", "content": clarify_msg},
            "gold_action": "clarify", "pre_state": copy.deepcopy(state),
        })

    return {
        "traj_id": traj_id, "env": env.name, "tools": env.tools(), "goal": goal,
        "init_state": env.blank(), "steps": steps, "final_state": final_state,
        "kind": "recovery" if recovery else "clean",
    }


def to_examples(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Explode a trajectory into per-step canonical multi-turn SFT examples."""
    out = []
    n = len(traj["steps"])
    for i, step in enumerate(traj["steps"]):
        out.append({
            "tools": traj["tools"],
            "query": step["query"],
            "answer": step["answer"],
            "history": step["history"] or None,
            "meta": {"source": "agentic", "env": traj["env"], "hn_kind": None, "verified": True,
                     "pair_id": traj["traj_id"], "step": i, "n_steps": n, "traj_kind": traj["kind"]},
        })
    return out


def generate_trajectories(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """`n` full trajectory objects (for eval + the agentic_trajectories.jsonl artifact)."""
    rng = random.Random(seed + 505)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        env = rng.choice([CartEnv, CalendarEnv])()
        traj = build_trajectory(env, rng, traj_id=f"traj_{len(out):05d}")
        if traj is not None:
            out.append(traj)
    return out


def generate(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Per-step SFT examples (flattened trajectories); whole trajectories are kept (grouped by pair_id)."""
    out: List[Dict[str, Any]] = []
    tid = 0
    rng = random.Random(seed + 505)
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        env = rng.choice([CartEnv, CalendarEnv])()
        traj = build_trajectory(env, rng, traj_id=f"traj_{tid:05d}")
        if traj is None:
            continue
        tid += 1
        out.extend(to_examples(traj))
    return out


def generate_dpo(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Execution-labeled trajectory-STEP preference pairs: chosen = gold call, rejected = a call the
    checker proves wrong on that step's state. Reuses envs._perturb. {prompt, chosen, rejected}."""
    from .format_utils import build_eval_prompt

    rng = random.Random(seed + 606)
    out: List[Dict[str, Any]] = []
    trajs = generate_trajectories(max(n, 8), seed)
    for traj in trajs:
        env = env_by_name(traj["env"])
        for step in traj["steps"]:
            if len(out) >= n:
                return out
            if step["gold_action"] != "call":
                continue
            gold = step["answer"]["calls"][0]
            bad = _perturb(env, step["pre_state"], gold, rng)
            if bad is None:
                continue
            ex = {"tools": traj["tools"], "query": step["query"], "history": step["history"] or None}
            out.append({
                "prompt": build_eval_prompt(ex),
                "chosen": target_to_json_str({"type": "tool_call", "calls": [gold]}),
                "rejected": json.dumps({"action": "call", "calls": [bad]}, sort_keys=True, ensure_ascii=False),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/out")
    ap.add_argument("--n", type=int, default=200, help="per-step SFT examples")
    ap.add_argument("--n-traj", type=int, default=60, help="full trajectories for eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    ex = generate(args.n, args.seed)
    trajs = generate_trajectories(args.n_traj, args.seed)
    with open(os.path.join(args.out_dir, "agentic_steps.jsonl"), "w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "agentic_trajectories.jsonl"), "w", encoding="utf-8") as f:
        for t in trajs:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    kinds = {}
    for t in trajs:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"[agentic] {len(ex)} per-step examples, {len(trajs)} trajectories {kinds}")


if __name__ == "__main__":
    main()
