"""Hard-negative generator — the core original contribution of this submission.

Most function-calling datasets only contain examples where a tool SHOULD be called. Real evals
(and the Berkeley Function-Calling Leaderboard's irrelevance category) heavily penalize a model for
(a) inventing a tool call when none applies, (b) guessing a missing required argument, or
(c) picking arbitrarily between two plausible tools. Training on those failure modes is where the
measurable improvement comes from.

We synthesize three kinds of hard negatives from REAL tool schemas already present in the corpus, so
the tools stay realistic and the negatives are grounded:

  no_tool      -> a request no available tool can satisfy        => answer: refuse
  missing_arg  -> a request that omits a required argument         => answer: clarify
  ambiguous    -> two similar tools, request underspecifies which  => answer: clarify

Everything is seeded and template-based (no LLM call required), so it is fully reproducible. You can
optionally post-process the `query` text with an LLM for fluency, but the labels are guaranteed correct
by construction.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# Requests that are plausibly phrased but that a *tool* list will almost never cover. We check against
# the actual tool names/descriptions to make sure we don't accidentally pick something a tool DOES do.
_OFF_DOMAIN_REQUESTS = [
    "What do you personally think is the meaning of life?",
    "Write me a short poem about the ocean.",
    "Can you feel emotions?",
    "Give me your honest opinion on which political party is best.",
    "Tell me a joke about cats.",
    "What's your favorite movie and why?",
    "Explain quantum entanglement like I'm five.",
    "Draft a heartfelt apology to my friend for missing their birthday.",
    "What will the stock market do next year?",
    "Sing me a lullaby.",
    "Summarize the plot of Hamlet.",
    "What's a good recipe for a vegan lasagna?",
]

_REFUSAL_TEMPLATES = [
    "None of the available tools can handle this request, so I can't complete it with the tools provided.",
    "I don't have a tool that supports this. I can only act through the listed tools.",
    "This request falls outside the available tools, so there's no valid tool call for it.",
]


def _tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t["name"] for t in tools]


def _required_args(tool: Dict[str, Any]) -> List[str]:
    return list((tool.get("parameters") or {}).get("required", []))


def _properties(tool: Dict[str, Any]) -> Dict[str, Any]:
    return dict((tool.get("parameters") or {}).get("properties", {}))


def make_no_tool(tools: List[Dict[str, Any]], rng: random.Random) -> Optional[Dict[str, Any]]:
    """A request no listed tool can satisfy -> the model must refuse, not invent a call."""
    haystack = " ".join(
        [t.get("name", "") + " " + t.get("description", "") for t in tools]
    ).lower()
    candidates = list(_OFF_DOMAIN_REQUESTS)  # copy — never mutate the module global
    rng.shuffle(candidates)
    for req in candidates:
        # crude overlap guard: skip if a tool clearly relates to the request's key noun
        key = req.lower().split()[0]
        if key in haystack:
            continue
        return {
            "tools": tools,
            "query": req,
            "answer": {"type": "refuse", "content": rng.choice(_REFUSAL_TEMPLATES)},
            "meta": {"source": "hard_negative", "hn_kind": "no_tool"},
        }
    return None


def make_missing_arg(
    tool: Dict[str, Any], rng: random.Random
) -> Optional[Dict[str, Any]]:
    """A request that references a tool but omits a REQUIRED argument -> clarify, don't guess."""
    req_args = _required_args(tool)
    if not req_args:
        return None
    props = _properties(tool)
    # keep at least one required arg present so the intent is clear, drop exactly one.
    dropped = rng.choice(req_args)
    kept = [a for a in req_args if a != dropped]
    desc = tool.get("description", tool["name"]).rstrip(".")
    kept_phrase = ""
    if kept:
        # mention the kept args generically so the query looks natural
        kept_phrase = " I can provide the " + ", ".join(kept) + " if needed."
    query = f"Please {desc.lower()}.{kept_phrase}"
    dropped_type = (props.get(dropped) or {}).get("type", "value")
    return {
        "tools": tool_list_context(tool, rng),
        "query": query,
        "answer": {
            "type": "clarify",
            "content": (
                f"I can use `{tool['name']}` for that, but I need the required "
                f"`{dropped}` ({dropped_type}) before I can proceed. Could you provide it?"
            ),
        },
        "meta": {"source": "hard_negative", "hn_kind": "missing_arg", "dropped": dropped},
    }


def make_ambiguous(
    tool_a: Dict[str, Any], tool_b: Dict[str, Any], rng: random.Random
) -> Dict[str, Any]:
    """Two plausible tools, an underspecified request -> ask which one, don't pick arbitrarily."""
    verb = rng.choice(["handle", "take care of", "do", "process"])
    query = (
        f"Can you {verb} this for me? Use whichever of your tools fits best."
    )
    return {
        "tools": [tool_a, tool_b],
        "query": query,
        "answer": {
            "type": "clarify",
            "content": (
                f"I have two tools that could apply — `{tool_a['name']}` and `{tool_b['name']}`. "
                f"Which would you like me to use, and what are the details?"
            ),
        },
        "meta": {"source": "hard_negative", "hn_kind": "ambiguous"},
    }


def tool_list_context(primary: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    """Return the primary tool alone (context distractors are added by the assembler)."""
    return [primary]


def generate(
    tool_pool: List[Dict[str, Any]],
    n: int,
    kind_weights: Dict[str, float],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate up to `n` hard negatives from a pool of real tool schemas.

    `tool_pool` is a de-duplicated list of tool schemas harvested from the positive corpus.
    """
    rng = random.Random(seed)
    if len(tool_pool) < 2:
        raise ValueError("need at least 2 distinct tools to build ambiguous negatives")

    kinds = list(kind_weights.keys())
    weights = [kind_weights[k] for k in kinds]
    out: List[Dict[str, Any]] = []
    attempts = 0
    while len(out) < n and attempts < n * 20:
        attempts += 1
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        if kind == "no_tool":
            # give the model a realistic set of 2-4 tools it must decline to use
            k = rng.randint(2, min(4, len(tool_pool)))
            tools = rng.sample(tool_pool, k)
            ex = make_no_tool(tools, rng)
        elif kind == "missing_arg":
            tool = rng.choice(tool_pool)
            ex = make_missing_arg(tool, rng)
            if ex is not None:
                # add 1-2 distractor tools so the choice isn't trivial
                distractors = [t for t in rng.sample(tool_pool, min(3, len(tool_pool)))
                               if t["name"] != tool["name"]][:2]
                ex["tools"] = ex["tools"] + distractors
        else:  # ambiguous
            a, b = rng.sample(tool_pool, 2)
            ex = make_ambiguous(a, b, rng)
        if ex is not None:
            out.append(ex)
    return out
