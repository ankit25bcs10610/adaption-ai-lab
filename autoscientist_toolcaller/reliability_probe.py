"""Reliability probe — a small, hand-authored, human-legible benchmark that DEMONSTRATES the moat.

The test set measures the model; this probe *shows* it. It is ~20 curated cases, each annotated with
exactly what tool-use reliability skill it exercises, spanning the decisions ordinary function-calling
data ignores:

  call · parallel-call · refuse (no applicable tool) · clarify (missing required arg) ·
  disambiguate (two plausible tools) · enum-aware clarify · long-context selection ·
  schema-drift remap · and — critically — OVER-REFUSAL TRAPS: requests that look underspecified but
  are actually satisfiable, where the correct answer is to CALL, not to refuse/clarify.

The over-refusal traps matter because the training set carries a large refuse/clarify slice; a probe
that a model can pass only by *both* abstaining when it should AND acting when it should keeps the
model honest and makes over-refusal visible.

Everything here is correct-by-construction: each case's gold answer, echoed back, scores 100% under the
same `judge_bfcl` the eval harness uses. Plug a real model's `generate_fn` in later to get a legible
pass/fail table (markdown + HTML); the DATA and scorer are fully offline now.

Usage (with a model):
  python -m autoscientist_toolcaller.reliability_probe --model <id> [--adapter path] --out results/reliability_probe.md
Offline self-check (oracle must score 100%):
  python -m autoscientist_toolcaller.reliability_probe --self-check
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List

from .eval_bfcl import judge_bfcl
from .format_utils import build_eval_prompt, target_to_json_str

# --- tool schemas used across the probe (realistic, minimal) ---------------------------------------
_WEATHER = {
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]},
}
_FLIGHT = {
    "name": "book_flight",
    "description": "Book a flight between two cities on a date",
    "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]},
}
_EMAIL = {
    "name": "send_email",
    "description": "Send an email to a recipient",
    "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]},
}
_TRANSLATE = {
    "name": "translate_text",
    "description": "Translate text into a target language",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string", "enum": ["en", "hi", "fr", "es"]}}, "required": ["text", "target_lang"]},
}
_CONVERT = {
    "name": "convert_currency",
    "description": "Convert an amount from one currency to another",
    "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}}, "required": ["amount", "from_ccy", "to_ccy"]},
}
_STOCK = {
    "name": "get_stock_price",
    "description": "Get the latest price for a stock ticker",
    "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]},
}
_REMINDER = {
    "name": "set_reminder",
    "description": "Set a reminder at a given time",
    "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]},
}


def _case(id, skill, tools, query, answer, category, history=None) -> Dict[str, Any]:
    ex: Dict[str, Any] = {"tools": tools, "query": query, "answer": answer,
                          "meta": {"source": "reliability_probe", "probe_id": id, "skill": skill}, "category": category}
    if history:
        ex["history"] = history
    return ex


# --- the probe -------------------------------------------------------------------------------------
def probe_cases() -> List[Dict[str, Any]]:
    return [
        _case("simple_call", "Call the right tool with the given argument",
              [_WEATHER, _STOCK], "What's the weather in Mumbai?",
              {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "simple"),
        _case("call_with_enum", "Fill an enum argument stated in the request",
              [_WEATHER], "Give me the weather in Delhi in fahrenheit.",
              {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Delhi", "units": "fahrenheit"}}]}, "simple"),
        _case("parallel_call", "Emit two calls when two independent actions are requested",
              [_WEATHER, _STOCK], "What's the weather in Pune and the price of AAPL?",
              {"type": "tool_call", "calls": [
                  {"name": "get_weather", "arguments": {"city": "Pune"}},
                  {"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]}, "parallel"),
        _case("refuse_no_tool", "Refuse when no available tool can satisfy the request",
              [_WEATHER, _STOCK], "Write me a haiku about the monsoon.",
              {"type": "refuse", "content": "None of the available tools can write poetry, so I can't do that with the tools provided."}, "irrelevance"),
        _case("refuse_wrong_domain", "Refuse a plausible-sounding but unsupported request",
              [_EMAIL, _REMINDER], "Please book me a taxi to the airport.",
              {"type": "refuse", "content": "I don't have a tool that can book a taxi; I can only send email or set reminders."}, "irrelevance"),
        _case("clarify_missing_arg", "Clarify instead of guessing a missing required arg",
              [_FLIGHT], "Book me a flight to Bangalore next Friday.",
              {"type": "clarify", "content": "I can book that flight, but I need the origin city before I can proceed. Where are you flying from?"}, "clarify"),
        _case("clarify_missing_recipient", "Clarify when the required recipient is absent",
              [_EMAIL], "Send an email saying the meeting is moved to 3pm.",
              {"type": "clarify", "content": "I can send that, but who should it go to? I need the recipient address."}, "clarify"),
        _case("disambiguate", "Ask which tool when two plausibly apply",
              [_EMAIL, _REMINDER], "Can you remind them about the deadline?",
              {"type": "clarify", "content": "Do you want me to send an email or set a reminder, and with what details?"}, "clarify"),
        _case("long_context_select", "Pick the one right tool among many distractors",
              [_WEATHER, _FLIGHT, _EMAIL, _TRANSLATE, _CONVERT, _STOCK, _REMINDER],
              "Convert 100 USD to INR.",
              {"type": "tool_call", "calls": [{"name": "convert_currency", "arguments": {"amount": 100, "from_ccy": "USD", "to_ccy": "INR"}}]}, "multiple"),
        _case("translate_enum", "Use an enum-constrained target language present in the request",
              [_TRANSLATE], "Translate 'good morning' to Hindi.",
              {"type": "tool_call", "calls": [{"name": "translate_text", "arguments": {"text": "good morning", "target_lang": "hi"}}]}, "simple"),
        _case("enum_violation_clarify", "Clarify when the requested option is outside the enum",
              [_TRANSLATE], "Translate 'hello' into Japanese.",
              {"type": "clarify", "content": "I can only translate into en, hi, fr, or es — Japanese isn't supported. Which of those would you like?"}, "clarify"),
        # --- OVER-REFUSAL TRAPS: satisfiable requests that look underspecified -> must CALL ----------
        _case("over_refusal_enough_info", "Do NOT over-refuse: all required args are actually present",
              [_WEATHER, _EMAIL], "email alice@example.com that I'll be late",
              {"type": "tool_call", "calls": [{"name": "send_email", "arguments": {"to": "alice@example.com", "body": "I'll be late"}}]}, "simple"),
        _case("over_refusal_casual", "Do NOT clarify when a casual phrasing still has the required arg",
              [_STOCK], "how's tesla stock doing? ticker TSLA",
              {"type": "tool_call", "calls": [{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}]}, "simple"),
        _case("over_refusal_defaults_optional", "Do NOT clarify for a MISSING OPTIONAL arg — just call",
              [_WEATHER], "weather in Chennai",
              {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Chennai"}}]}, "simple"),
        _case("over_refusal_all_present", "Do NOT refuse a fully-specified booking",
              [_FLIGHT], "Book a flight from Mumbai to Goa on 2026-08-01.",
              {"type": "tool_call", "calls": [{"name": "book_flight", "arguments": {"origin": "Mumbai", "destination": "Goa", "date": "2026-08-01"}}]}, "simple"),
        # --- multi-turn: clarify was already given; now the arg arrives -> CALL ----------------------
        _case("followup_now_call", "After clarifying, call once the missing arg is supplied",
              [_FLIGHT], "From Hyderabad.",
              {"type": "tool_call", "calls": [{"name": "book_flight", "arguments": {"origin": "Hyderabad", "destination": "Bangalore", "date": "2026-07-10"}}]}, "multi_turn",
              history=[
                  {"role": "user", "content": "Book me a flight to Bangalore on 2026-07-10."},
                  {"role": "assistant", "content": '{"action":"clarify","message":"Where are you flying from?"}'}]),
        _case("multi_step_reminder", "Set a reminder with both required args from the request",
              [_REMINDER, _EMAIL], "Remind me to call the bank at 9am tomorrow.",
              {"type": "tool_call", "calls": [{"name": "set_reminder", "arguments": {"text": "call the bank", "time": "9am tomorrow"}}]}, "simple"),
        _case("refuse_capability", "Refuse a capability question with no matching tool",
              [_WEATHER, _CONVERT], "Can you feel emotions?",
              {"type": "refuse", "content": "That's outside what my tools can do — I can only fetch weather or convert currency."}, "irrelevance"),
    ]


def score(generate_fn: Callable[[str], str], lenient: bool = True) -> Dict[str, Any]:
    """Run the probe through a model's generate_fn and score each case with the eval judge."""
    cases = probe_cases()
    results = []
    n_ok = 0
    for ex in cases:
        out = generate_fn(build_eval_prompt(ex))
        v = judge_bfcl(ex, out, lenient=lenient)
        ok = bool(v["correct"])
        n_ok += int(ok)
        results.append({
            "probe_id": ex["meta"]["probe_id"], "skill": ex["meta"]["skill"],
            "category": ex["category"], "correct": ok,
            "gold": target_to_json_str(ex["answer"]), "output": out[:240],
        })
    return {"n": len(cases), "passed": n_ok, "pass_rate": n_ok / len(cases) if cases else 0.0, "results": results}


def to_markdown(scored: Dict[str, Any]) -> str:
    lines = [
        f"# Reliability probe — {scored['passed']}/{scored['n']} passed "
        f"({scored['pass_rate']*100:.0f}%)",
        "",
        "A curated demonstration of tool-use reliability, including over-refusal traps.",
        "",
        "| ✓ | Probe | Skill exercised | Category |",
        "|---|---|---|---|",
    ]
    for r in scored["results"]:
        mark = "✅" if r["correct"] else "❌"
        lines.append(f"| {mark} | `{r['probe_id']}` | {r['skill']} | {r['category']} |")
    return "\n".join(lines)


def to_html(scored: Dict[str, Any]) -> str:
    head = "".join(f"<th>{h}</th>" for h in ["✓", "Probe", "Skill", "Category"])
    body = ""
    for r in scored["results"]:
        mark = "✅" if r["correct"] else "❌"
        body += f"<tr><td>{mark}</td><td><code>{r['probe_id']}</code></td><td>{r['skill']}</td><td>{r['category']}</td></tr>"
    return (f"<h2>Reliability probe — {scored['passed']}/{scored['n']} "
            f"({scored['pass_rate']*100:.0f}%)</h2>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def _oracle(ex_by_prompt: Dict[str, str]) -> Callable[[str], str]:
    """A perfect predictor: returns each case's gold envelope. Used by --self-check."""
    return lambda prompt: ex_by_prompt.get(prompt, '{"action":"refuse","message":"?"}')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", default="results/reliability_probe.md")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--self-check", action="store_true", help="run the oracle; it must fully pass (100 pct)")
    args = ap.parse_args()

    if args.self_check or not args.model:
        cases = probe_cases()
        gold_by_prompt = {build_eval_prompt(ex): target_to_json_str(ex["answer"]) for ex in cases}
        scored = score(_oracle(gold_by_prompt), lenient=not args.strict)
        print(to_markdown(scored))
        if scored["pass_rate"] < 1.0:
            raise SystemExit(f"self-check FAILED: oracle scored {scored['pass_rate']:.0%} (probe is not correct-by-construction)")
        print("\n[probe] self-check OK (oracle 100%).")
        return

    from .eval_harness import hf_generate_fn
    gen = hf_generate_fn(args.model, 512, 0.0, args.adapter)
    scored = score(gen, lenient=not args.strict)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(to_markdown(scored) + "\n")
    json.dump(scored, open(args.out.replace(".md", ".json"), "w"), indent=2)
    print(to_markdown(scored))
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
