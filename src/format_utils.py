"""Canonical example format + rendering to Adaption's prompt/completion columns.

An "example" is a plain dict:

    {
      "tools":  [ {"name","description","parameters"(JSON Schema)}, ... ],
      "query":  "user request text",
      "answer": {"type": "tool_call"|"refuse"|"clarify", "calls": [...], "content": "..."},
      "meta":   {"source": "...", "hn_kind": "no_tool|missing_arg|ambiguous"|None},
    }

We render `answer` into a single JSON object the model must emit. Forcing a JSON envelope for *every*
turn (including refusals/clarifications) is deliberate: it makes hard-negative behavior machine-checkable
in the eval harness, and it teaches the model that "no call" is a first-class, structured decision.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# The assistant must always answer with ONE JSON object matching this envelope.
#   {"action": "call",     "calls": [{"name": "...", "arguments": {...}}, ...]}
#   {"action": "refuse",   "message": "..."}
#   {"action": "clarify",  "message": "..."}
SYSTEM_TEMPLATE = """You are a precise function-calling assistant.

You are given a list of available tools as JSON Schemas. For the user's request, respond with a SINGLE \
JSON object and nothing else, using exactly one of these shapes:

- To call one or more tools:
  {{"action": "call", "calls": [{{"name": "<tool_name>", "arguments": {{...}}}}]}}
- If NO available tool can satisfy the request, do not invent one:
  {{"action": "refuse", "message": "<brief explanation>"}}
- If a required argument is missing or the request is ambiguous between tools:
  {{"action": "clarify", "message": "<what you need to proceed>"}}

Rules:
- Only use tools from the provided list. Never call a tool that is not listed.
- Only include arguments defined in a tool's schema; include every required argument.
- If a required argument's value is not present in the request, ask for it with "clarify" — never guess.
- Output JSON only. No prose, no markdown fences.

Available tools:
{tools_json}"""


def render_tools(tools: List[Dict[str, Any]]) -> str:
    """Pretty, stable JSON for the tool list (sorted keys => deterministic prompts)."""
    return json.dumps(tools, indent=2, sort_keys=True, ensure_ascii=False)


def build_system_prompt(tools: List[Dict[str, Any]]) -> str:
    return SYSTEM_TEMPLATE.format(tools_json=render_tools(tools))


def answer_to_target(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize an example's `answer` into the exact JSON envelope the model must produce."""
    t = answer.get("type")
    if t == "tool_call":
        return {"action": "call", "calls": answer["calls"]}
    if t == "refuse":
        return {"action": "refuse", "message": answer.get("content", "")}
    if t == "clarify":
        return {"action": "clarify", "message": answer.get("content", "")}
    raise ValueError(f"Unknown answer type: {t!r}")


def target_to_json_str(answer: Dict[str, Any]) -> str:
    """Deterministic serialization of the target completion."""
    return json.dumps(answer_to_target(answer), sort_keys=True, ensure_ascii=False)


def render_history(history: List[Dict[str, Any]] | None) -> str:
    """Flatten prior conversation turns into a readable transcript for template-free prompts."""
    if not history:
        return ""
    lines = []
    for turn in history:
        role = str(turn.get("role", "user")).upper()
        lines.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(lines)


def build_eval_prompt(example: Dict[str, Any]) -> str:
    """Single source of truth for the inference prompt (single- AND multi-turn).

    Used by every eval path so base and fine-tuned models see byte-identical prompts, and so multi-turn
    history is rendered consistently. Training (to_prompt_completion) mirrors this.
    """
    system = build_system_prompt(example["tools"])
    hist = render_history(example.get("history"))
    if hist:
        return f"{system}\n\nConversation so far:\n{hist}\n\nUser request:\n{example['query']}"
    return f"{system}\n\nUser request:\n{example['query']}"


def to_prompt_completion(example: Dict[str, Any], tokenizer=None) -> Dict[str, str]:
    """Render an example into {"prompt", "completion"}.

    If a tokenizer with a chat template is supplied, the prompt is built with
    `apply_chat_template(..., add_generation_prompt=True)` so it matches inference byte-for-byte
    (the #1 silent cause of fine-tunes underperforming). Otherwise we fall back to a plain
    role-tagged string, which Adaption's preprocessing can also consume. Multi-turn examples carry a
    `history` list of {role, content} turns that precede the final user query.
    """
    system = build_system_prompt(example["tools"])
    user = example["query"]
    history = example.get("history") or []
    completion = target_to_json_str(example["answer"])

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system}]
        for turn in history:
            messages.append({"role": turn.get("role", "user"), "content": turn.get("content", "")})
        messages.append({"role": "user", "content": user})
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        hist = render_history(history)
        convo = f"{hist}\n" if hist else ""
        prompt = f"<|system|>\n{system}\n{convo}<|user|>\n{user}\n<|assistant|>\n"

    return {"prompt": prompt, "completion": completion}


def strip_think(text: str) -> str | None:
    """Return only the answer portion after a reasoning block.

    Reasoning traces (``<think> ... </think>``) frequently contain brace-y prose that would
    otherwise be captured by the first-``{`` scan below and silently mis-graded. We parse ONLY
    what follows ``</think>``. A ``<think>`` opened but never closed is malformed output, so we
    signal a parse failure (None) rather than grading truncated reasoning as an answer.
    """
    lower = text.lower()
    open_i = lower.find("<think>")
    if open_i == -1:
        return text
    close = lower.find("</think>", open_i)
    if close == -1:
        return None  # dangling reasoning block -> unparseable
    return text[close + len("</think>"):]


def parse_model_output(text: str) -> Dict[str, Any] | None:
    """Best-effort parse of a model's raw output into the envelope dict.

    Tolerates markdown fences and leading/trailing prose so eval isn't unfairly strict about
    formatting. Returns None if no JSON object can be recovered.
    """
    stripped = strip_think(text)
    if stripped is None:
        return None
    s = stripped.strip()
    if s.startswith("```"):
        s = s.strip("`")
        # drop an optional leading "json" language tag
        if s[:4].lower() == "json":
            s = s[4:]
        s = s.strip()
    # find the first balanced {...} block
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
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None
