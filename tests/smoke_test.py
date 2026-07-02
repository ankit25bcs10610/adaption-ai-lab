"""No-network smoke test for the core logic (format, validation, hard negatives, eval judging).

Run:  python -m tests.smoke_test
Exercises everything that does NOT need model downloads or the Adaption API, so bugs in the data/eval
path surface before you spend credits.
"""
from __future__ import annotations

import json
import sys

from src import hard_negatives
from src.eval_harness import evaluate, judge
from src.format_utils import (
    build_system_prompt,
    parse_model_output,
    target_to_json_str,
    to_prompt_completion,
)
from src.schema_validator import validate_answer, validate_call

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a flight between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
]


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main() -> int:
    ok = True

    print("format_utils:")
    pc = to_prompt_completion(
        {
            "tools": TOOLS,
            "query": "weather in Mumbai?",
            "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]},
        }
    )
    ok &= check("prompt has tools", "get_weather" in pc["prompt"])
    ok &= check("completion is json envelope", json.loads(pc["completion"])["action"] == "call")

    print("parse_model_output tolerance:")
    fenced = '```json\n{"action": "refuse", "message": "no tool"}\n```'
    ok &= check("parses fenced json", parse_model_output(fenced)["action"] == "refuse")
    ok &= check("parses trailing prose", parse_model_output('Sure! {"action":"clarify","message":"x"} ok')["action"] == "clarify")

    print("schema_validator:")
    good, _ = validate_call({"name": "get_weather", "arguments": {"city": "Mumbai"}}, TOOLS)
    ok &= check("valid call passes", good)
    bad_extra, _ = validate_call({"name": "get_weather", "arguments": {"city": "X", "foo": 1}}, TOOLS)
    ok &= check("extra arg rejected", not bad_extra)
    bad_missing, _ = validate_call({"name": "book_flight", "arguments": {"origin": "A"}}, TOOLS)
    ok &= check("missing required rejected", not bad_missing)
    bad_name, _ = validate_call({"name": "nope", "arguments": {}}, TOOLS)
    ok &= check("unknown tool rejected", not bad_name)
    a_ok, _ = validate_answer({"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "X"}}]}, TOOLS)
    ok &= check("valid answer passes", a_ok)

    print("hard_negatives:")
    hn = hard_negatives.generate(TOOLS, 12, {"no_tool": 0.4, "missing_arg": 0.35, "ambiguous": 0.25}, seed=1)
    ok &= check("generated some", len(hn) > 0)
    ok &= check("all are hard negatives", all(e["meta"]["source"] == "hard_negative" for e in hn))
    ok &= check("kinds valid", all(e["answer"]["type"] in ("refuse", "clarify") for e in hn))
    ok &= check("determinism", hard_negatives.generate(TOOLS, 12, {"no_tool": 0.4, "missing_arg": 0.35, "ambiguous": 0.25}, seed=1) == hn)

    print("eval judge:")
    pos = {"tools": TOOLS, "query": "weather?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {"hn_kind": None}}
    ok &= check("correct call judged correct", judge(pos, target_to_json_str(pos["answer"]))["correct"])
    ok &= check("hallucinated call flagged", judge(
        {"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "no tool"}, "meta": {"hn_kind": "no_tool"}},
        '{"action":"call","calls":[{"name":"get_weather","arguments":{"city":"X"}}]}',
    )["hallucinated_call"])
    ok &= check("correct refusal judged correct", judge(
        {"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "no tool"}, "meta": {"hn_kind": "no_tool"}},
        '{"action":"refuse","message":"nothing fits"}',
    )["correct"])

    print("evaluate() aggregation:")
    records = [pos,
               {"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "x"}, "meta": {"hn_kind": "no_tool"}}]
    # a "model" that always refuses: right on the negative, wrong on the positive
    metrics = evaluate(records, lambda p: '{"action":"refuse","message":"x"}')
    ok &= check("overall acc = 0.5", abs(metrics["overall_accuracy"] - 0.5) < 1e-9)
    ok &= check("hallucination rate = 0", metrics["hallucination_rate"] == 0.0)
    detailed = evaluate(records, lambda p: '{"action":"refuse","message":"x"}', return_records=True)
    ok &= check("return_records shape", "metrics" in detailed and len(detailed["records"]) == 2)

    print("eval_bfcl:")
    from src.eval_bfcl import args_match_lenient, categorize, judge_bfcl, normalize_value
    ok &= check("categorize simple", categorize({"tools": [TOOLS[0]], "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "simple")
    ok &= check("categorize multiple", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "multiple")
    ok &= check("categorize parallel", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]}}) == "parallel")
    ok &= check("categorize irrelevance", categorize({"tools": TOOLS, "answer": {"type": "refuse", "content": "x"}}) == "irrelevance")
    ok &= check("normalize num string", normalize_value("42") == 42.0)
    ok &= check("lenient case/space match", args_match_lenient({"city": " New York "}, {"city": "new york"}))
    ok &= check("lenient acceptable list", args_match_lenient({"city": "NYC"}, {"city": {"_acceptable": ["nyc", "new york city"]}}))
    ok &= check("lenient judge correct", judge_bfcl(pos, target_to_json_str(pos["answer"]))["correct"])

    print("quality_filter:")
    from src.quality_filter import filter_examples, heuristic_score
    good_ex = {"tools": TOOLS, "query": "what is the weather in Mumbai city today", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}}
    junk_ex = {"tools": TOOLS, "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "string"}}]}}
    ok &= check("good scores higher than junk", heuristic_score(good_ex) > heuristic_score(junk_ex))
    kept = filter_examples([good_ex, junk_ex], 0.5)
    ok &= check("filter keeps the good one", kept == [good_ex])

    print("build_preference:")
    from src.build_preference import build_pairs
    hn_ex = {"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "no tool"}, "meta": {"hn_kind": "no_tool"}}
    pairs = build_pairs([pos, hn_ex], seed=1)
    ok &= check("pairs built", len(pairs) == 2)
    ok &= check("hard-neg rejected is a call", json.loads(pairs[1]["rejected"])["action"] == "call")
    ok &= check("hard-neg chosen is refuse", json.loads(pairs[1]["chosen"])["action"] == "refuse")
    ok &= check("preference determinism", build_pairs([pos, hn_ex], seed=1) == pairs)

    print("multiturn:")
    from src import multiturn as mt
    from src.format_utils import build_eval_prompt
    mts = mt.generate(TOOLS, 12, {"miss_param": 0.4, "miss_func": 0.3, "long_context": 0.3}, seed=2)
    ok &= check("multiturn generated", len(mts) > 0)
    ok &= check("all have history", all(e.get("history") for e in mts))
    ok &= check("mt determinism", mt.generate(TOOLS, 12, {"miss_param": 0.4, "miss_func": 0.3, "long_context": 0.3}, seed=2) == mts)
    from src.eval_bfcl import categorize as cat2
    ok &= check("categorized multi_turn", all(cat2(e) == "multi_turn" for e in mts))
    ok &= check("eval prompt includes history", "Conversation so far" in build_eval_prompt(mts[0]))
    # a miss_func example's gold is a call and should judge correct when echoed back
    mf = next((e for e in mts if e["meta"]["mt_kind"] == "miss_func"), None)
    if mf:
        ok &= check("miss_func judged correct", judge(mf, target_to_json_str(mf["answer"]))["correct"])

    print("export_bfcl:")
    from src.export_bfcl import to_bfcl
    b = to_bfcl(pos, 0)
    ok &= check("bfcl has function list", len(b["function"]) == len(pos["tools"]))
    ok &= check("bfcl question is nested turns", isinstance(b["question"], list) and isinstance(b["question"][0], list))
    b_neg = to_bfcl({"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "x"}, "meta": {}}, 1)
    ok &= check("bfcl refuse -> empty ground_truth", b_neg["ground_truth"] == [])

    print("schema_drift:")
    from src import schema_drift as sd
    sds = sd.generate(TOOLS, 12, {"add_required": 0.4, "retype_enum": 0.3, "rename": 0.3}, seed=3)
    ok &= check("schema_drift generated", len(sds) > 0)
    ok &= check("sd determinism", sd.generate(TOOLS, 12, {"add_required": 0.4, "retype_enum": 0.3, "rename": 0.3}, seed=3) == sds)
    ok &= check("sd kinds valid", all(e["meta"]["sd_kind"] in ("add_required", "retype_enum", "rename") for e in sds))
    # a rename example's gold call must validate against its DRIFTED tool schema
    rn = next((e for e in sds if e["meta"]["sd_kind"] == "rename"), None)
    if rn:
        okc, _ = validate_answer(rn["answer"], rn["tools"])
        ok &= check("rename call valid vs drifted schema", okc)
    ar = next((e for e in sds if e["meta"]["sd_kind"] == "add_required"), None)
    if ar:
        ok &= check("add_required -> clarify", ar["answer"]["type"] == "clarify")

    print("eval_report:")
    from src.eval_report import confusion_from_predictions, render_html
    preds = [
        {"gold": "call", "pred": "call"}, {"gold": "call", "pred": "call"},
        {"gold": "refuse", "pred": "call"}, {"gold": "refuse", "pred": "refuse"},
        {"gold": "clarify", "pred": "parse_fail"},
    ]
    cm = confusion_from_predictions(preds)
    ok &= check("confusion counts diagonal", cm.get(("call", "call")) == 2)
    ok &= check("confusion counts off-diagonal", cm.get(("refuse", "call")) == 1)
    html = render_html(
        {"overall_accuracy": 0.41, "hallucination_rate": 0.6},
        {"overall_accuracy": 0.83, "hallucination_rate": 0.08},
        {"by_category": {"simple": {"n": 10, "accuracy": 0.9}}},
        preds,
    )
    ok &= check("html has confusion matrix", "confusion" in html and "+42.0 pp" in html)
    ok &= check("html has category bar", "Per-category" in html)

    print("fill_model_card:")
    from src.fill_model_card import build_metrics_block
    blk = build_metrics_block({"overall_accuracy": 0.4, "overall_stderr": 0.02, "hallucination_rate": 0.6},
                              {"overall_accuracy": 0.8, "overall_stderr": 0.02, "hallucination_rate": 0.1},
                              {"evaluation_summary": {"improvement_percent": 37}})
    ok &= check("metrics block has both cols", "0.400" in blk and "0.800" in blk)
    ok &= check("metrics block has improvement", "37" in blk)

    print("eval_stats:")
    from src.eval_stats import bootstrap_ci, mcnemar, paired_gap
    ci = bootstrap_ci([1] * 80 + [0] * 20)
    ok &= check("bootstrap_ci mean=0.8", abs(ci["mean"] - 0.8) < 1e-9)
    ok &= check("bootstrap_ci brackets mean", ci["lo"] <= ci["mean"] <= ci["hi"])
    # base 60% vs ft 90% on 100 aligned examples -> clear, significant improvement
    base = [1] * 60 + [0] * 40
    ft = [1] * 90 + [0] * 10
    g = paired_gap(base, ft)
    ok &= check("paired gap ~ +0.30", abs(g["gap"] - 0.30) < 1e-9)
    ok &= check("gap CI excludes 0", g["lo"] > 0)
    ok &= check("bootstrap p<0.05", g["p_value"] < 0.05)
    mc = mcnemar(base, ft)
    ok &= check("mcnemar p<0.05 on clear win", mc["p_value"] < 0.05)
    ok &= check("no gap -> p=1", paired_gap([1, 0, 1, 0], [1, 0, 1, 0])["p_value"] == 1.0)

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
