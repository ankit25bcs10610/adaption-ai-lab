"""Live agent runtime — goal → plan → CALL a real tool → observe → repeat (advanced #5).

The trajectories/eval teach and score the behavior; THIS is the live loop that puts it to work. Give the
agent a goal and a ToolRegistry of REAL callables; it drives a model (the offline sim, Anthropic, or your
fine-tuned tool-caller) through the same call / refuse / clarify envelope, executes each call, feeds the
observation back, and stops when the model calls `finish`, abstains (refuse/clarify), or hits the step
budget. The dataset's discipline (don't hallucinate a tool; ask when unsure) is exactly what makes such
an agent safe to run autonomously.

Bounded, not magic: the agent can only do what its registered tools allow, and only as well as its model.
Everything is offline-testable with an injected model_fn + safe stdlib tools.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .format_utils import build_eval_prompt, parse_model_output

ModelFn = Callable[[str], str]
ToolFn = Callable[..., Any]


class ToolRegistry:
    """Maps tool name -> (JSON-Schema, python callable). `call` executes safely and returns (ok, result)."""

    def __init__(self) -> None:
        self.schemas: List[Dict[str, Any]] = []
        self._fns: Dict[str, ToolFn] = {}

    def register(self, schema: Dict[str, Any], fn: ToolFn) -> "ToolRegistry":
        self.schemas.append(schema)
        self._fns[schema["name"]] = fn
        return self

    def call(self, name: str, args: Dict[str, Any]) -> Tuple[bool, Any]:
        if name not in self._fns:
            return False, f"unknown tool '{name}'"
        try:
            return True, self._fns[name](**(args or {}))
        except Exception as e:  # a tool that raises is an observation, not a crash
            return False, f"{type(e).__name__}: {e}"


def _structural_score(registry: ToolRegistry, parsed: Dict[str, Any] | None) -> float:
    """Cheap, side-effect-free verifier for best-of-N when no env oracle is available: prefer a
    well-formed call to REGISTERED tools > a valid abstention > an unknown-tool call > unparseable."""
    if parsed is None:
        return -1.0
    action = parsed.get("action")
    if action in ("refuse", "clarify"):
        return 1.0
    if action == "call":
        calls = parsed.get("calls") or []
        if not calls:
            return -1.0
        return 2.0 if all(c.get("name") in registry._fns for c in calls) else 0.0
    return -1.0


def run_agent(
    goal: str,
    registry: ToolRegistry,
    model_fn: ModelFn,
    max_steps: int = 8,
    best_of_n: int = 1,
    verify_fn=None,
) -> Dict[str, Any]:
    """Run the agent loop. Returns {status, steps, result, transcript, history}.

    status: 'done' (model called finish) · 'refuse'/'clarify' (model abstained) · 'max_steps' · 'parse_error'.
    best_of_n>1 draws K candidates per step and commits the best per `verify_fn` (default: structural).
    """
    history: List[Dict[str, str]] = []
    transcript: List[Dict[str, Any]] = []
    status, result = "max_steps", None
    verify = verify_fn or (lambda p: _structural_score(registry, p))
    for step in range(max_steps):
        prompt = build_eval_prompt({"tools": registry.schemas, "query": goal, "history": history or None})
        cands = [model_fn(prompt) for _ in range(max(1, best_of_n))]
        plist = [parse_model_output(c) for c in cands]
        best_i = max(range(len(cands)), key=lambda i: (verify(plist[i]), -i))
        raw, parsed = cands[best_i], plist[best_i]
        if parsed is None:
            status = "parse_error"
            transcript.append({"step": step, "action": None, "raw": raw})
            break
        action = parsed.get("action")
        if action != "call":
            status = action if action in ("refuse", "clarify") else "parse_error"
            transcript.append({"step": step, "action": action, "message": parsed.get("message")})
            break
        # execute each requested call and build one observation
        obs_parts, calls = [], parsed.get("calls") or []
        finished = False
        for c in calls:
            name, cargs = c.get("name"), c.get("arguments", {}) or {}
            if name == "finish":
                finished, result = True, cargs.get("answer", "")
                obs_parts.append("finish: done")
                continue
            ok, res = registry.call(name, cargs)
            obs_parts.append(f"{name}: {'OK ' + str(res) if ok else 'ERROR ' + str(res)}")
        transcript.append({"step": step, "action": "call", "calls": calls, "observation": "; ".join(obs_parts)})
        history.append({"role": "assistant", "content": raw})
        history.append({"role": "tool", "content": "; ".join(obs_parts)})
        if finished:
            status = "done"
            break
    return {"status": status, "steps": len(transcript), "result": result,
            "transcript": transcript, "history": history}


# --------------------------------------------------------------------------------------
# Tool registries
# --------------------------------------------------------------------------------------
def _num(p="a number"):
    return {"type": "number", "description": p}


_FINISH_SCHEMA = {
    "name": "finish", "description": "Call this when the goal is complete, with the final answer.",
    "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
}


def safe_tools_registry() -> ToolRegistry:
    """Deterministic, offline, side-effect-free real tools — safe to run anywhere (great for a demo)."""
    r = ToolRegistry()
    r.register({"name": "add", "description": "Add two numbers",
                "parameters": {"type": "object", "properties": {"a": _num(), "b": _num()}, "required": ["a", "b"]}},
               lambda a, b: a + b)
    r.register({"name": "multiply", "description": "Multiply two numbers",
                "parameters": {"type": "object", "properties": {"a": _num(), "b": _num()}, "required": ["a", "b"]}},
               lambda a, b: a * b)
    r.register({"name": "word_count", "description": "Count the words in a piece of text",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
               lambda text: len(str(text).split()))
    r.register(_FINISH_SCHEMA, lambda answer="": answer)
    return r


def env_registry(env) -> ToolRegistry:
    """Wrap a stateful autoscientist_toolcaller.envs environment (Cart/Calendar) as agent tools that mutate its state."""
    r = ToolRegistry()
    state = {"s": env.blank()}

    def _mk(tool_name):
        def _fn(**args):
            new, ok, reason = env.apply(state["s"], {"name": tool_name, "arguments": args})
            if ok:
                state["s"] = new
                return env.describe(new)
            raise ValueError(reason)
        return _fn

    for schema in env.tools():
        r.register(schema, _mk(schema["name"]))
    r.register(_FINISH_SCHEMA, lambda answer="": answer)
    r._state = state  # exposed for inspection/tests
    return r


def register_mcp(registry: ToolRegistry, session) -> ToolRegistry:
    """Optional: register the tools of a connected MCP session (lazy; needs the `mcp` SDK + a running
    server). `session` is an initialized MCP client session; we list its tools and wrap each `call_tool`.
    Documented hook — not exercised offline."""
    tools = session.list_tools()  # SDK-specific; wrap per your MCP client
    for t in getattr(tools, "tools", tools):
        schema = {"name": t.name, "description": getattr(t, "description", ""),
                  "parameters": getattr(t, "inputSchema", {"type": "object", "properties": {}})}
        registry.register(schema, (lambda _n: (lambda **a: session.call_tool(_n, a)))(t.name))
    return registry


# --------------------------------------------------------------------------------------
# Model backends (lazy)
# --------------------------------------------------------------------------------------
def hf_model_fn(model_id: str, adapter: Optional[str] = None, max_new_tokens: int = 512) -> ModelFn:
    from .eval_harness import hf_generate_fn
    return hf_generate_fn(model_id, max_new_tokens, 0.0, adapter)


def anthropic_model_fn(model: str = "claude-opus-4-8") -> ModelFn:
    """Drive the agent with Claude (the whole prompt — incl. the envelope instructions — is the user msg)."""
    from .synth_llm import _default_complete_fn
    cf = _default_complete_fn(model)
    return lambda prompt: cf("", prompt)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the tool-calling agent on a goal.")
    ap.add_argument("--goal", required=True)
    ap.add_argument("--backend", choices=["anthropic", "hf"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--sandbox", default=None, help="dir to expose read/write file tools (sandboxed)")
    ap.add_argument("--http", action="store_true", help="also expose a read-only http_get tool")
    args = ap.parse_args()
    model_fn = anthropic_model_fn(args.model) if args.backend == "anthropic" else hf_model_fn(args.model)
    if args.sandbox:
        from .agent_tools import register_http, sandbox_fs_registry
        registry = sandbox_fs_registry(args.sandbox, allow_write=True)
        if args.http:
            register_http(registry)
    else:
        registry = safe_tools_registry()
    result = run_agent(args.goal, registry, model_fn, max_steps=args.max_steps)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
