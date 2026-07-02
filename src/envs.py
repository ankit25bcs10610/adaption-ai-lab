"""Execution-verified tool environments — the highest-leverage data work (research upgrade #3).

Deterministic toy domains (cart, calendar) with tool stubs + state-diff checkers. We generate a target
tool call for a given state+intent, then EXECUTE it against the environment and keep the example only if
the resulting state matches the expected state — so every positive is *correct by construction and
execution-verified*, not LLM-guessed. The failed perturbations become execution-labeled DPO pairs
(chosen = verified call, rejected = a call the checker proves wrong on the same prompt) — the data-centric
stand-in for RL-with-verifiable-rewards (the platform can't run online RL).

Outputs canonical function-calling examples (tools/query/answer/meta) compatible with src.format_utils,
plus DPO pairs compatible with src.build_preference. Pure stdlib; seeded; offline-testable.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Environment protocol: tools() -> schemas ; apply(state, call) -> (new_state, ok, reason)
# --------------------------------------------------------------------------------------
FRUITS = ["apples", "bananas", "oranges", "grapes", "mangoes"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TITLES = ["Standup", "Review", "1:1", "Planning", "Demo"]


class CartEnv:
    name = "cart"

    def tools(self) -> List[Dict[str, Any]]:
        item = {"type": "string", "enum": FRUITS}
        qty = {"type": "integer"}
        return [
            {"name": "add_item", "description": "Add a quantity of an item to the cart",
             "parameters": {"type": "object", "properties": {"item": item, "quantity": qty},
                            "required": ["item", "quantity"]}},
            {"name": "remove_item", "description": "Remove a quantity of an item from the cart",
             "parameters": {"type": "object", "properties": {"item": item, "quantity": qty},
                            "required": ["item", "quantity"]}},
            {"name": "checkout", "description": "Check out the current cart",
             "parameters": {"type": "object", "properties": {}, "required": []}},
        ]

    def blank(self) -> Dict[str, Any]:
        return {"items": {}, "checked_out": False}

    def apply(self, state: Dict[str, Any], call: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
        s = copy.deepcopy(state)
        n, a = call.get("name"), call.get("arguments", {})
        if s["checked_out"]:
            return s, False, "cart already checked out"
        if n == "add_item":
            if a.get("item") not in FRUITS or not isinstance(a.get("quantity"), int) or a["quantity"] <= 0:
                return s, False, "bad add args"
            s["items"][a["item"]] = s["items"].get(a["item"], 0) + a["quantity"]
            return s, True, "ok"
        if n == "remove_item":
            have = s["items"].get(a.get("item"), 0)
            if a.get("item") not in FRUITS or not isinstance(a.get("quantity"), int) or a["quantity"] <= 0:
                return s, False, "bad remove args"
            if a["quantity"] > have:
                return s, False, "removing more than present"
            s["items"][a["item"]] = have - a["quantity"]
            if s["items"][a["item"]] == 0:
                del s["items"][a["item"]]
            return s, True, "ok"
        if n == "checkout":
            if not s["items"]:
                return s, False, "cannot check out empty cart"
            s["checked_out"] = True
            return s, True, "ok"
        return s, False, f"unknown tool {n}"

    def describe(self, state: Dict[str, Any]) -> str:
        if not state["items"]:
            return "the cart is empty"
        return "cart: " + ", ".join(f"{q} {i}" for i, q in state["items"].items())


class CalendarEnv:
    name = "calendar"

    def tools(self) -> List[Dict[str, Any]]:
        title = {"type": "string", "enum": TITLES}
        day = {"type": "string", "enum": DAYS}
        return [
            {"name": "create_event", "description": "Create an event on a day",
             "parameters": {"type": "object", "properties": {"title": title, "day": day},
                            "required": ["title", "day"]}},
            {"name": "cancel_event", "description": "Cancel an event by title",
             "parameters": {"type": "object", "properties": {"title": title}, "required": ["title"]}},
            {"name": "move_event", "description": "Move an event to a new day",
             "parameters": {"type": "object", "properties": {"title": title, "day": day},
                            "required": ["title", "day"]}},
        ]

    def blank(self) -> Dict[str, Any]:
        return {"events": {}}

    def apply(self, state: Dict[str, Any], call: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, str]:
        s = copy.deepcopy(state)
        n, a = call.get("name"), call.get("arguments", {})
        if n == "create_event":
            if a.get("title") not in TITLES or a.get("day") not in DAYS:
                return s, False, "bad create args"
            if a["title"] in s["events"]:
                return s, False, "event already exists"
            s["events"][a["title"]] = a["day"]
            return s, True, "ok"
        if n == "cancel_event":
            if a.get("title") not in s["events"]:
                return s, False, "no such event"
            del s["events"][a["title"]]
            return s, True, "ok"
        if n == "move_event":
            if a.get("title") not in s["events"] or a.get("day") not in DAYS:
                return s, False, "cannot move"
            s["events"][a["title"]] = a["day"]
            return s, True, "ok"
        return s, False, f"unknown tool {n}"

    def describe(self, state: Dict[str, Any]) -> str:
        if not state["events"]:
            return "the calendar is empty"
        return "calendar: " + ", ".join(f"{t} on {d}" for t, d in state["events"].items())


ENVS = [CartEnv, CalendarEnv]


# --------------------------------------------------------------------------------------
# Scenario generation: build a state via a random valid history, then a verified next call
# --------------------------------------------------------------------------------------
def _valid_next_call(env, state: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    """Propose a call that applies successfully to `state` (execution-verified)."""
    for _ in range(12):
        call = _random_call(env, state, rng, valid_bias=True)
        _, ok, _ = env.apply(state, call)
        if ok:
            return call
    return None


def _random_call(env, state, rng, valid_bias: bool) -> Dict[str, Any]:
    if isinstance(env, CartEnv):
        kind = rng.choice(["add", "add", "remove", "checkout"])
        if kind == "add":
            return {"name": "add_item", "arguments": {"item": rng.choice(FRUITS), "quantity": rng.randint(1, 5)}}
        if kind == "remove" and state["items"] and valid_bias:
            it = rng.choice(list(state["items"]))
            return {"name": "remove_item", "arguments": {"item": it, "quantity": rng.randint(1, state["items"][it])}}
        if kind == "remove":
            return {"name": "remove_item", "arguments": {"item": rng.choice(FRUITS), "quantity": rng.randint(1, 5)}}
        return {"name": "checkout", "arguments": {}}
    # calendar
    kind = rng.choice(["create", "create", "cancel", "move"])
    if kind == "create":
        avail = [t for t in TITLES if t not in state["events"]] or TITLES
        return {"name": "create_event", "arguments": {"title": rng.choice(avail), "day": rng.choice(DAYS)}}
    if kind == "cancel" and state["events"] and valid_bias:
        return {"name": "cancel_event", "arguments": {"title": rng.choice(list(state["events"]))}}
    if kind == "cancel":
        return {"name": "cancel_event", "arguments": {"title": rng.choice(TITLES)}}
    if state["events"] and valid_bias:
        return {"name": "move_event", "arguments": {"title": rng.choice(list(state["events"])), "day": rng.choice(DAYS)}}
    return {"name": "move_event", "arguments": {"title": rng.choice(TITLES), "day": rng.choice(DAYS)}}


def _intent_text(env, call: Dict[str, Any]) -> str:
    a = call["arguments"]
    if call["name"] == "add_item":
        return f"Add {a['quantity']} {a['item']} to my cart."
    if call["name"] == "remove_item":
        return f"Remove {a['quantity']} {a['item']} from my cart."
    if call["name"] == "checkout":
        return "Check out my cart."
    if call["name"] == "create_event":
        return f"Schedule {a['title']} on {a['day']}."
    if call["name"] == "cancel_event":
        return f"Cancel {a['title']}."
    if call["name"] == "move_event":
        return f"Move {a['title']} to {a['day']}."
    return "Do it."


def _perturb(env, state, call, rng) -> Optional[Dict[str, Any]]:
    """A plausible-but-wrong call that the checker proves fails on the same state (for DPO rejected)."""
    for _ in range(12):
        bad = copy.deepcopy(call)
        mode = rng.choice(["wrong_arg", "wrong_tool", "invalid"])
        if mode == "wrong_arg" and bad["arguments"]:
            k = rng.choice(list(bad["arguments"]))
            if isinstance(bad["arguments"][k], int):
                bad["arguments"][k] = bad["arguments"][k] + rng.randint(5, 20)  # over-remove / bad qty
            else:
                pool = FRUITS if k in ("item",) else DAYS if k == "day" else TITLES
                bad["arguments"][k] = rng.choice([x for x in pool if x != bad["arguments"][k]] or pool)
        elif mode == "wrong_tool":
            others = [t["name"] for t in env.tools() if t["name"] != bad["name"]]
            bad["name"] = rng.choice(others)
        else:
            bad = _random_call(env, state, rng, valid_bias=False)
        _, ok_bad, _ = env.apply(state, bad)
        if not ok_bad and bad != call:
            return bad  # verified to fail -> a legitimate DPO 'rejected'
    return None


def _build_state(env, rng: random.Random, steps: int) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
    state = env.blank()
    hist: List[Tuple[str, Dict[str, Any]]] = []
    for _ in range(steps):
        call = _valid_next_call(env, state, rng)
        if call is None:
            break
        new, ok, _ = env.apply(state, call)
        if not ok:
            break
        hist.append((_intent_text(env, call), call))
        state = new
    return state, hist


def generate(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Execution-verified function-calling examples (canonical format, with history + current state)."""
    rng = random.Random(seed + 202)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        env = rng.choice(ENVS)()
        state, hist = _build_state(env, rng, rng.randint(0, 3))
        call = _valid_next_call(env, state, rng)
        if call is None:
            continue
        new_state, ok, _ = env.apply(state, call)
        if not ok:  # execution check (must pass by construction, but verify)
            continue
        history = []
        for intent, c in hist:
            history.append({"role": "user", "content": intent})
            history.append({"role": "assistant", "content": json.dumps({"action": "call", "calls": [c]})})
        out.append({
            "tools": env.tools(),
            "query": f"({env.describe(state)}) {_intent_text(env, call)}",
            "answer": {"type": "tool_call", "calls": [call]},
            "history": history or None,
            "meta": {"source": "env", "env": env.name, "hn_kind": None,
                     "verified": True, "expected_state": new_state},
        })
    return out


def _apply_seq(env, state: Dict[str, Any], calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Apply a sequence of calls; return the final state, or None if any step fails."""
    s = state
    for c in calls:
        s, ok, _ = env.apply(s, c)
        if not ok:
            return None
    return s


def generate_multicall(n: int, seed: int = 42, k_range: Tuple[int, int] = (2, 3)) -> List[Dict[str, Any]]:
    """Execution-verified MULTI-CALL trajectories: 2-3 calls the model must emit together.

    Each call is verified by replaying it against the environment (execution-verified). We keep only
    ORDER-INDEPENDENT compositions — every permutation of the k calls reaches the same final state — so
    the gold set of calls is unambiguous under the eval's order-insensitive matcher. This is the only
    slice that trains multi-call COMPLETENESS from verified state.
    """
    from itertools import permutations

    rng = random.Random(seed + 404)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        env = rng.choice(ENVS)()
        state, hist = _build_state(env, rng, rng.randint(0, 2))
        k = rng.randint(k_range[0], k_range[1])
        calls: List[Dict[str, Any]] = []
        cur = state
        for _ in range(k):
            c = _valid_next_call(env, cur, rng)
            if c is None:
                break
            nxt, ok, _ = env.apply(cur, c)
            if not ok:
                break
            calls.append(c)
            cur = nxt
        if len(calls) < k:
            continue
        # reject exact-duplicate calls (unnatural request) and non-commuting sets
        if len({json.dumps(c, sort_keys=True) for c in calls}) != len(calls):
            continue
        final = _apply_seq(env, state, calls)
        if final is None:
            continue
        if any(_apply_seq(env, state, list(p)) != final for p in permutations(calls)):
            continue  # order matters -> ambiguous gold; skip
        history = []
        for intent, c in hist:
            history.append({"role": "user", "content": intent})
            history.append({"role": "assistant", "content": json.dumps({"action": "call", "calls": [c]})})
        query = f"({env.describe(state)}) " + " ".join(_intent_text(env, c) for c in calls)
        out.append({
            "tools": env.tools(),
            "query": query,
            "answer": {"type": "tool_call", "calls": calls},
            "history": history or None,
            "meta": {"source": "env", "env": env.name, "hn_kind": None,
                     "verified": True, "multicall": True, "expected_state": final},
        })
    return out


def generate_dpo(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Execution-labeled DPO pairs: chosen = verified call, rejected = checker-proven-wrong call."""
    from .format_utils import build_system_prompt, target_to_json_str

    rng = random.Random(seed + 303)
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        env = rng.choice(ENVS)()
        state, _ = _build_state(env, rng, rng.randint(0, 2))
        call = _valid_next_call(env, state, rng)
        if call is None:
            continue
        bad = _perturb(env, state, call, rng)
        if bad is None:
            continue
        ex = {"tools": env.tools(), "query": f"({env.describe(state)}) {_intent_text(env, call)}"}
        prompt = f"{build_system_prompt(ex['tools'])}\n\nUser request:\n{ex['query']}"
        out.append({
            "prompt": prompt,
            "chosen": target_to_json_str({"type": "tool_call", "calls": [call]}),
            "rejected": json.dumps({"action": "call", "calls": [bad]}, sort_keys=True, ensure_ascii=False),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/out")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-dpo", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ex = generate(args.n, args.seed)
    dpo = generate_dpo(args.n_dpo, args.seed)
    with open(os.path.join(args.out_dir, "env_verified.jsonl"), "w") as f:
        for e in ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "env_dpo.jsonl"), "w") as f:
        for d in dpo:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"[envs] {len(ex)} execution-verified examples, {len(dpo)} execution-labeled DPO pairs")


if __name__ == "__main__":
    main()
