"""LLM-judge data synthesis + filter loop (advanced #4).

Mints NEW hard function-calling cases with an LLM, then GATES each one: an LLM critique pass, a
programmatic schema-verify, and a dedup. The LLM is reached through an INJECTABLE
`complete_fn(system, user) -> str`, so the whole loop unit-tests offline with a fake; the default
builds a real Anthropic client (lazy, gated on ANTHROPIC_API_KEY). Nothing here fabricates numbers — it
produces *candidate training rows*, each of which must pass the same `schema_validator` gate as every
other slice, so a hallucinated/invalid gold is dropped, not shipped.

Offline: `synthesize(n, tools_pool, complete_fn=fake)`. Live: set ANTHROPIC_API_KEY, then
`python -m autoscientist_toolcaller.synth_llm --n 50` (or wire dataset.llm_synth_examples in config).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Callable, Dict, List, Optional

from .schema_validator import validate_answer

CompleteFn = Callable[[str, str], str]

_GEN_SYSTEM = (
    "You generate ONE hard function-calling TRAINING case as strict JSON. Given a list of tool JSON "
    "Schemas, invent a realistic user request plus the gold decision, using exactly this shape:\n"
    '{"query": "<user request>", "answer": {"type": "tool_call"|"refuse"|"clarify", '
    '"calls": [{"name": "<tool>", "arguments": {...}}], "content": "<message for refuse/clarify>"}}\n'
    "Make it a HARD case: prefer requests where the right decision is to REFUSE (no listed tool fits) "
    "or CLARIFY (a required argument is missing or two tools are plausible), not just an easy call. "
    "Only use listed tools; include every required argument when calling. Output JSON only."
)
_CRITIQUE_SYSTEM = (
    "You are a strict data-quality judge (critique). Given the tools and a candidate case, decide if "
    "the gold answer is CORRECT for the query and genuinely useful (a real hard negative or a clean "
    'call), not trivial or mislabeled. Reply as strict JSON: {"ok": true|false, "reason": "<why>"}.'
)


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Recover the first balanced {...} JSON object from an LLM reply (tolerates fences/prose)."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s[:4].lower() == "json":
            s = s[4:]
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _default_complete_fn(model: str = "claude-opus-4-8") -> CompleteFn:
    """Real Anthropic client (lazy import, gated on ANTHROPIC_API_KEY)."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set — pass a complete_fn or set the key")
    import anthropic  # lazy: never imported on the offline path

    client = anthropic.Anthropic(api_key=key)

    def _fn(system: str, user: str) -> str:
        msg = client.messages.create(
            model=model, max_tokens=1500, thinking={"type": "adaptive"},
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")

    return _fn


def _norm_q(q: Any) -> str:
    return re.sub(r"\s+", " ", str(q).strip().lower())


def synthesize(
    n: int,
    tools_pool: List[Dict[str, Any]],
    complete_fn: Optional[CompleteFn] = None,
    seed: int = 42,
    model: str = "claude-opus-4-8",
    max_attempts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate → schema-verify → dedup → LLM-critique. Returns canonical examples (source=llm_synth)."""
    rng = random.Random(seed + 717)
    cf = complete_fn or _default_complete_fn(model)
    out: List[Dict[str, Any]] = []
    seen: set = set()
    attempts = 0
    cap = max_attempts if max_attempts is not None else n * 8
    while len(out) < n and attempts < cap:
        attempts += 1
        tools = rng.sample(tools_pool, min(len(tools_pool), rng.randint(1, 3)))
        gen = _extract_json(cf(_GEN_SYSTEM, "Tools:\n" + json.dumps(tools, ensure_ascii=False)))
        if not gen or "query" not in gen or not isinstance(gen.get("answer"), dict):
            continue
        query, answer = gen["query"], gen["answer"]
        if answer.get("type") not in ("tool_call", "refuse", "clarify"):
            continue
        # programmatic schema gate — a hallucinated/invalid gold is dropped here, not shipped
        okv, _ = validate_answer(answer, tools)
        if not okv:
            continue
        key = _norm_q(query)
        if key in seen:
            continue
        # LLM critique gate
        crit = _extract_json(cf(_CRITIQUE_SYSTEM, json.dumps(
            {"tools": tools, "query": query, "answer": answer}, ensure_ascii=False)))
        if not (crit and crit.get("ok") is True):
            continue
        seen.add(key)
        out.append({
            "tools": tools, "query": query, "answer": answer,
            "meta": {"source": "llm_synth", "hn_kind": answer.get("hn_kind"),
                     "answer_type": answer.get("type")},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="data/out/llm_synth.jsonl")
    args = ap.parse_args()
    import yaml
    from .build_dataset import harvest_tool_pool

    cfg = yaml.safe_load(open(args.config))
    # tool pool from the built train split, else a tiny fallback
    src = os.path.join(cfg["paths"]["out_dir"], "train.jsonl")
    pool: List[Dict[str, Any]] = []
    if os.path.exists(src):
        exs = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
        pool = harvest_tool_pool(exs)
    if not pool:
        raise SystemExit("no tool pool found — build the dataset first (train.jsonl)")
    rows = synthesize(args.n, pool, model=cfg["dataset"].get("quality_judge_model", "claude-opus-4-8"),
                      seed=cfg["seed"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[llm_synth] wrote {len(rows)} generated+critiqued+verified cases -> {args.out}")


if __name__ == "__main__":
    main()
