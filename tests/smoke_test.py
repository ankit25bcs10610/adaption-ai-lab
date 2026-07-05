"""No-network smoke test for the core logic (format, validation, hard negatives, eval judging).

Run:  python -m tests.smoke_test
Exercises everything that does NOT need model downloads or the Adaption API, so bugs in the data/eval
path surface before you spend credits.
"""
from __future__ import annotations

import json
import sys

from autoscientist_toolcaller import hard_negatives
from autoscientist_toolcaller.eval_harness import evaluate, judge
from autoscientist_toolcaller.format_utils import (
    build_system_prompt,
    parse_model_output,
    target_to_json_str,
    to_prompt_completion,
)
from autoscientist_toolcaller.schema_validator import validate_answer, validate_call

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
    # <think> firewall: brace-y reasoning must not be mistaken for the answer JSON.
    thinky = '<think>Maybe I should {call foo} but no tool fits</think>{"action":"refuse","message":"no tool"}'
    ok &= check("think block ignored, answer parsed", parse_model_output(thinky)["action"] == "refuse")
    ok &= check("dangling think -> parse fail", parse_model_output('<think>reasoning {a:1} with no close') is None)

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
    # Hammer no_tool: from a real positive, the offered tools must EXCLUDE the gold tool -> refuse.
    _hammer_pool = TOOLS + [
        {"name": "send_email", "description": "Send an email", "parameters": {"type": "object", "properties": {"to": {"type": "string"}}, "required": ["to"]}},
        {"name": "create_event", "description": "Create a calendar event", "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}},
    ]
    _pos = {"tools": [TOOLS[0]], "query": "What's the weather in Mumbai?",
            "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]},
            "meta": {"source": "toolace"}}
    hn_pos = hard_negatives.generate(_hammer_pool, 40, {"no_tool": 1.0}, seed=3, positives=[_pos])
    nt = [e for e in hn_pos if e["meta"].get("hn_kind") == "no_tool" and "removed_tool" in e["meta"]]
    ok &= check("hammer no_tool generated", len(nt) > 0)
    ok &= check("hammer excludes gold tool", all("get_weather" not in [t["name"] for t in e["tools"]] for e in nt))
    ok &= check("hammer answer is refuse", all(e["answer"]["type"] == "refuse" for e in nt))

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
    from autoscientist_toolcaller.eval_bfcl import args_match_lenient, categorize, judge_bfcl, normalize_value
    ok &= check("categorize simple", categorize({"tools": [TOOLS[0]], "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "simple")
    ok &= check("categorize multiple", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "multiple")
    ok &= check("categorize parallel", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]}}) == "parallel")
    ok &= check("categorize irrelevance", categorize({"tools": TOOLS, "answer": {"type": "refuse", "content": "x"}}) == "irrelevance")
    ok &= check("normalize num string", normalize_value("42") == 42.0)
    ok &= check("lenient case/space match", args_match_lenient({"city": " New York "}, {"city": "new york"}))
    ok &= check("lenient acceptable list", args_match_lenient({"city": "NYC"}, {"city": {"_acceptable": ["nyc", "new york city"]}}))
    ok &= check("lenient judge correct", judge_bfcl(pos, target_to_json_str(pos["answer"]))["correct"])

    print("bfcl_handler (envelope translation):")
    from autoscientist_toolcaller.bfcl_handler import translate, is_no_call
    ok &= check("call -> decoded ast", translate('{"action":"call","calls":[{"name":"f","arguments":{"a":1}}]}') == [{"f": {"a": 1}}])
    ok &= check("refuse -> empty (no call)", translate('{"action":"refuse","message":"x"}') == [])
    ok &= check("is_no_call on clarify", is_no_call('{"action":"clarify","message":"x"}'))
    ok &= check("think block firewalled in handler", translate('<think>{z}</think>{"action":"call","calls":[{"name":"g","arguments":{}}]}') == [{"g": {}}])

    print("export_bfcl:")
    from autoscientist_toolcaller.export_bfcl import to_bfcl_prompt, to_bfcl_answer
    _ex = {"tools": TOOLS, "query": "weather?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}}
    _p = to_bfcl_prompt(_ex, "BFCL_v4_simple_0")
    _a = to_bfcl_answer(_ex, "BFCL_v4_simple_0")
    ok &= check("prompt has question turns", _p["question"] == [[{"role": "user", "content": "weather?"}]])
    ok &= check("answer wraps values in lists", _a["ground_truth"] == [{"get_weather": {"city": ["Mumbai"]}}])
    ok &= check("irrelevance answer empty", to_bfcl_answer({"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "x"}}, "i")["ground_truth"] == [])

    print("quality_filter:")
    from autoscientist_toolcaller.quality_filter import filter_examples, heuristic_score
    good_ex = {"tools": TOOLS, "query": "what is the weather in Mumbai city today", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}}
    junk_ex = {"tools": TOOLS, "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "string"}}]}}
    ok &= check("good scores higher than junk", heuristic_score(good_ex) > heuristic_score(junk_ex))
    kept = filter_examples([good_ex, junk_ex], 0.5)
    ok &= check("filter keeps the good one", kept == [good_ex])

    print("build_preference:")
    from autoscientist_toolcaller.build_preference import build_pairs
    hn_ex = {"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "no tool"}, "meta": {"hn_kind": "no_tool"}}
    pairs = build_pairs([pos, hn_ex], seed=1)
    ok &= check("pairs built", len(pairs) == 2)
    ok &= check("hard-neg rejected is a call", json.loads(pairs[1]["rejected"])["action"] == "call")
    ok &= check("hard-neg chosen is refuse", json.loads(pairs[1]["chosen"])["action"] == "refuse")
    ok &= check("preference determinism", build_pairs([pos, hn_ex], seed=1) == pairs)
    # Poison guard: no pair may have rejected == chosen, and every positive's rejected is a real error.
    ok &= check("no poison pair (chosen==rejected)", all(p["chosen"] != p["rejected"] for p in pairs))
    from autoscientist_toolcaller.build_preference import _confirmed_wrong
    ok &= check("positive rejected is confirmed wrong", _confirmed_wrong(pos, json.loads(pairs[0]["rejected"])))

    print("multiturn:")
    from autoscientist_toolcaller import multiturn as mt
    from autoscientist_toolcaller.format_utils import build_eval_prompt
    mts = mt.generate(TOOLS, 12, {"miss_param": 0.4, "miss_func": 0.3, "long_context": 0.3}, seed=2)
    ok &= check("multiturn generated", len(mts) > 0)
    ok &= check("all have history", all(e.get("history") for e in mts))
    ok &= check("mt determinism", mt.generate(TOOLS, 12, {"miss_param": 0.4, "miss_func": 0.3, "long_context": 0.3}, seed=2) == mts)
    from autoscientist_toolcaller.eval_bfcl import categorize as cat2
    ok &= check("categorized multi_turn", all(cat2(e) == "multi_turn" for e in mts))
    ok &= check("eval prompt includes history", "Conversation so far" in build_eval_prompt(mts[0]))
    # a miss_func example's gold is a call and should judge correct when echoed back
    mf = next((e for e in mts if e["meta"]["mt_kind"] == "miss_func"), None)
    if mf:
        ok &= check("miss_func judged correct", judge(mf, target_to_json_str(mf["answer"]))["correct"])

    print("schema_drift:")
    from autoscientist_toolcaller import schema_drift as sd
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
    from autoscientist_toolcaller.eval_report import confusion_from_predictions, render_html
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
    from autoscientist_toolcaller.fill_model_card import build_metrics_block
    blk = build_metrics_block({"overall_accuracy": 0.4, "overall_stderr": 0.02, "hallucination_rate": 0.6},
                              {"overall_accuracy": 0.8, "overall_stderr": 0.02, "hallucination_rate": 0.1},
                              {"evaluation_summary": {"improvement_percent": 37}})
    ok &= check("metrics block has both cols", "0.400" in blk and "0.800" in blk)
    ok &= check("metrics block has improvement", "37" in blk)

    print("eval_stats:")
    from autoscientist_toolcaller.eval_stats import bootstrap_ci, mcnemar, paired_gap
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

    print("robustness_table:")
    from autoscientist_toolcaller.robustness_table import robustness
    base_p = ([{"category": "simple", "correct": True}] * 9 + [{"category": "simple", "correct": False}]
              + [{"category": "multi_turn", "correct": True}] * 4 + [{"category": "multi_turn", "correct": False}] * 6)
    ft_p = ([{"category": "simple", "correct": True}] * 9 + [{"category": "simple", "correct": False}]
            + [{"category": "multi_turn", "correct": True}] * 8 + [{"category": "multi_turn", "correct": False}] * 2)
    r = robustness(base_p, ft_p)
    ok &= check("robustness computes drops", r["mean_base_drop"] is not None and r["mean_ft_drop"] is not None)
    ok &= check("ft more robust (smaller drop)", r["mean_ft_drop"] < r["mean_base_drop"])

    print("eval_decompose (gap decomposition):")
    from autoscientist_toolcaller.eval_decompose import decompose, aggregate_seeds, to_html as dec_html
    drows = ([{"category": "simple", "base": 1, "ft": 1}] * 8 + [{"category": "simple", "base": 0, "ft": 1}] * 2
             + [{"category": "irrelevance", "base": 0, "ft": 1}] * 6 + [{"category": "irrelevance", "base": 0, "ft": 0}] * 4
             + [{"category": "multi_turn", "base": 1, "ft": 1}] * 3 + [{"category": "multi_turn", "base": 0, "ft": 1}] * 5)
    dec = decompose(drows, n_boot=300)
    ok &= check("decomposition identity holds (Σ contrib == gap)", dec["identity_ok"])
    ok &= check("sum contributions == overall gap", abs(dec["sum_contributions"] - dec["overall"]["gap"]) < 1e-9)
    ok &= check("per-condition contributions present", len(dec["by_condition"]) == 3)
    ok &= check("irrelevance carries positive contribution", any(c["condition"] == "irrelevance" and c["contribution"] > 0 for c in dec["by_condition"]))
    ok &= check("decomposition html renders", "<table>" in dec_html(dec))
    # multi-seed aggregation: 3 identical seeds -> zero std
    import tempfile, os as _os
    _paths = []
    for s in range(3):
        bp = _os.path.join(tempfile.gettempdir(), f"_dec_base_{s}.jsonl")
        fp = _os.path.join(tempfile.gettempdir(), f"_dec_ft_{s}.jsonl")
        open(bp, "w").write("\n".join(json.dumps({"category": r2["category"], "correct": bool(r2["base"])}) for r2 in drows))
        open(fp, "w").write("\n".join(json.dumps({"category": r2["category"], "correct": bool(r2["ft"])}) for r2 in drows))
        _paths.append((bp, fp))
    agg = aggregate_seeds(_paths, n_boot=100)
    ok &= check("multiseed aggregates 3 seeds", agg["seeds"] == 3)
    ok &= check("identical seeds -> zero gap std", agg["gap"]["std"] < 1e-9)

    print("multilingual:")
    from autoscientist_toolcaller import multilingual as _ml
    _mlr = _ml.generate(30, seed=1)
    ok &= check("multilingual generated", len(_mlr) > 0)
    ok &= check("multilingual all correct-by-construction", all(judge(r, target_to_json_str(r["answer"]))["correct"] for r in _mlr))
    ok &= check("multilingual has en + hi + hi-rom + es + fr", {r["meta"]["lang"] for r in _mlr} >= {"en", "hi", "hi-rom", "es", "fr"})
    ok &= check("multilingual matched twins (pair_id shared across langs)", sum(1 for r in _mlr if r["meta"]["pair_id"] == _mlr[0]["meta"]["pair_id"]) == 5)
    ok &= check("multilingual determinism", _ml.generate(30, seed=1) == _mlr)

    print("reasoning traces:")
    from autoscientist_toolcaller import reasoning as _rz
    import random as _rr
    _rng2 = _rr.Random(0)
    _rzW = {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    _rzex = {"tools": [_rzW], "query": "weather?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {}}
    _tr = _rz.render_trace(_rzex, _rng2)
    ok &= check("trace is short (<60 tokens)", 0 < len(_tr.split()) <= 60)
    _pc = _rz.with_trace(to_prompt_completion(_rzex), _rzex, _rng2)
    ok &= check("completion wrapped with <think>", _pc["completion"].startswith("<think>") and "</think>" in _pc["completion"])
    ok &= check("firewall recovers envelope after trace", parse_model_output(_pc["completion"])["action"] == "call")

    print("toucan parser (#8, offline):")
    from autoscientist_toolcaller.build_dataset import _toucan_row_to_example
    _trow = {"tools": [{"type": "function", "function": {"name": "ping", "description": "p", "parameters": {"type": "object", "properties": {"host": {"type": "string"}}, "required": ["host"]}}}],
             "messages": [{"role": "user", "content": "ping it"},
                          {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "ping", "arguments": '{"host":"example.com"}'}}]}]}
    _tex = _toucan_row_to_example(_trow)
    ok &= check("toucan row parses to example", _tex is not None and _tex["answer"]["calls"][0]["name"] == "ping")
    ok &= check("toucan unwraps function tools", _tex["tools"][0]["name"] == "ping" and "function" not in _tex["tools"][0])
    ok &= check("toucan gold valid vs schema", validate_answer(_tex["answer"], _tex["tools"])[0])
    ok &= check("toucan handles JSON-string fields", _toucan_row_to_example({"tools": json.dumps(_trow["tools"]), "messages": json.dumps(_trow["messages"])}) is not None)

    print("curriculum (difficulty + ordering):")
    from autoscientist_toolcaller import curriculum as _cur
    _easy = {"tools": [_rzW], "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {}}
    _hard = {"tools": [_rzW, _rzW, _rzW, _rzW], "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "a", "arguments": {"p": 1}}, {"name": "b", "arguments": {"q": 2}}]}, "meta": {"sd_kind": "rename"}}
    ok &= check("harder example scores higher", _cur.difficulty(_hard) > _cur.difficulty(_easy))
    ok &= check("curriculum orders easy->hard", _cur.curriculum_order([_hard, _easy]) == [1, 0])
    _h = _cur.histogram([_easy, _hard])
    ok &= check("histogram has bands + mean", "by_band" in _h and _h["n"] == 2)

    print("error explorer:")
    from autoscientist_toolcaller.eval_report import render_error_explorer
    _errs = [{"category": "irrelevance", "reason": "hallucinated_call", "query": "write a poem",
              "gold": {"type": "refuse"}, "output": '{"action":"call","calls":[{"name":"x"}]}'}]
    _eh = render_error_explorer(_errs)
    ok &= check("error explorer renders failures", "Error explorer" in _eh and "hallucinated_call" in _eh)
    ok &= check("error explorer empty -> empty string", render_error_explorer([]) == "")

    print("reliability_probe:")
    from autoscientist_toolcaller.reliability_probe import probe_cases, score, to_markdown as probe_md
    from autoscientist_toolcaller.format_utils import build_eval_prompt as _bep, target_to_json_str as _tjs
    _cases = probe_cases()
    _gold = {_bep(ex): _tjs(ex["answer"]) for ex in _cases}
    oracle = score(lambda p: _gold.get(p, '{"action":"refuse","message":"?"}'))
    ok &= check("probe correct-by-construction (oracle 100%)", oracle["pass_rate"] == 1.0)
    ok &= check("probe has over-refusal traps", any("over_refusal" in r["probe_id"] for r in oracle["results"]))
    # a naive 'always call the first tool' predictor MUST fail the refuse/clarify cases -> probe discriminates
    naive = score(lambda p: '{"action":"call","calls":[{"name":"get_weather","arguments":{"city":"X"}}]}')
    ok &= check("probe discriminates (naive < 100%)", naive["pass_rate"] < 1.0)
    ok &= check("probe markdown renders", "Reliability probe" in probe_md(oracle))

    print("envs (execution-verified):")
    from autoscientist_toolcaller.envs import CalendarEnv, CartEnv
    from autoscientist_toolcaller.envs import generate as env_gen, generate_dpo as env_dpo
    cart = CartEnv()
    s1, aok, _ = cart.apply(cart.blank(), {"name": "add_item", "arguments": {"item": "apples", "quantity": 3}})
    ok &= check("cart add verified", aok and s1["items"]["apples"] == 3)
    _, rok, _ = cart.apply(s1, {"name": "remove_item", "arguments": {"item": "apples", "quantity": 10}})
    ok &= check("cart over-remove rejected", not rok)
    _, cok, _ = cart.apply(cart.blank(), {"name": "checkout", "arguments": {}})
    ok &= check("checkout empty cart rejected", not cok)
    _, eok, _ = CalendarEnv().apply(CalendarEnv().blank(), {"name": "cancel_event", "arguments": {"title": "Standup"}})
    ok &= check("cancel missing event rejected", not eok)
    exs = env_gen(20, seed=1)
    ok &= check("env examples generated", len(exs) > 0)
    ok &= check("env examples verified", all(e["meta"]["verified"] and e["answer"]["type"] == "tool_call" for e in exs))
    # execution consistency: the answer call must name a real tool in that example's tool list
    ok &= check("env answer uses a real tool", all(
        e["answer"]["calls"][0]["name"] in {t["name"] for t in e["tools"]} for e in exs))
    ok &= check("env determinism", env_gen(20, seed=1) == exs)
    dpo = env_dpo(20, seed=1)
    ok &= check("env DPO pairs built", len(dpo) > 0 and all(d["chosen"] != d["rejected"] for d in dpo))

    # ---- round-2 advanced upgrades -------------------------------------------------------------
    print("schema_drift poison guard (#1):")
    from autoscientist_toolcaller.format_utils import sample_value
    import random as _r
    _rng = _r.Random(0)
    ok &= check("sample_value enum -> member", sample_value({"type": "string", "enum": ["a", "b"]}, _rng) in ("a", "b"))
    ok &= check("sample_value integer -> int", isinstance(sample_value({"type": "integer"}, _rng), int))
    _tool = {"name": "mk", "description": "make", "parameters": {"type": "object", "properties": {
        "count": {"type": "integer"}, "priority": {"type": "string", "enum": ["low", "high"]}, "title": {"type": "string"}},
        "required": ["count", "priority", "title"]}}
    _sd = sd.generate([_tool], 30, {"rename": 1.0}, seed=1)
    ok &= check("all rename golds valid (no poison)", all(validate_answer(e["answer"], e["tools"])[0] for e in _sd if e["answer"]["type"] == "tool_call"))
    ok &= check("rename int arg is int", all(isinstance(list(e["answer"]["calls"][0]["arguments"].values())[0], (int, str)) for e in _sd))

    print("hard-negative taxonomy (#2/#3):")
    _W = {"name": "get_weather", "description": "weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    _S = {"name": "get_stock", "description": "stock", "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}
    _pA = {"tools": [_W], "query": "What is the weather in Mumbai?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {}}
    _pB = {"tools": [_S], "query": "Price of AAPL?", "answer": {"type": "tool_call", "calls": [{"name": "get_stock", "arguments": {"ticker": "AAPL"}}]}, "meta": {}}
    _ort = hard_negatives.generate([_W, _S], 8, {"over_refusal": 1.0}, seed=1, positives=[_pA, _pB])
    ok &= check("over_refusal -> tool_call", _ort and all(e["answer"]["type"] == "tool_call" and e["meta"]["hn_kind"] == "over_refusal" for e in _ort))
    ok &= check("over_refusal keeps gold tool + hedges query", all(e["query"] != _pA["query"] and e["query"] != _pB["query"] for e in _ort))
    ok &= check("over_refusal judged: refuse is WRONG", not judge(_ort[0], '{"action":"refuse","message":"x"}')["correct"] and judge(_ort[0], target_to_json_str(_ort[0]["answer"]))["correct"])
    _pp = hard_negatives.generate([_W, _S], 8, {"partial_parallel": 1.0}, seed=2, positives=[_pA, _pB])
    ok &= check("partial_parallel -> 2 distinct calls", _pp and all(len(e["answer"]["calls"]) == 2 for e in _pp))
    ok &= check("partial_parallel single-call is WRONG", not judge(_pp[0], target_to_json_str({"type": "tool_call", "calls": [_pp[0]["answer"]["calls"][0]]}))["correct"])
    ok &= check("partial_parallel skips with <2 positives", hard_negatives.generate([_W, _S], 5, {"partial_parallel": 1.0}, seed=1, positives=[_pA]) == [])

    print("env multicall (#4):")
    from autoscientist_toolcaller.envs import generate_multicall
    _mc = generate_multicall(8, seed=1)
    ok &= check("multicall 2-3 verified calls", _mc and all(2 <= len(e["answer"]["calls"]) <= 3 and validate_answer(e["answer"], e["tools"])[0] for e in _mc))
    ok &= check("multicall drop-one is WRONG", not judge(_mc[0], target_to_json_str({"type": "tool_call", "calls": _mc[0]["answer"]["calls"][:-1]}))["correct"])
    ok &= check("multicall determinism", generate_multicall(8, seed=1) == _mc)

    print("pref hardness + env merge (#5):")
    from autoscientist_toolcaller.build_preference import _hardness
    _chosen = {"action": "call", "calls": [{"name": "f", "arguments": {"a": 1, "b": 2}}]}
    _nearmiss = {"action": "call", "calls": [{"name": "f", "arguments": {"a": 1, "b": 9}}]}
    _swap = {"action": "call", "calls": [{"name": "g", "arguments": {"a": 1, "b": 2}}]}
    ok &= check("near-miss harder (smaller dist) than tool swap", _hardness(_chosen, _nearmiss) < _hardness(_chosen, _swap))

    print("decontamination (#6):")
    from autoscientist_toolcaller.decontaminate import decontaminate, DEFAULT_PROBES
    _rows = [{"query": DEFAULT_PROBES[0], "meta": {"source": "toolace"}},
             {"query": "Completely unrelated: parse this XML config file for me.", "meta": {"source": "toolace"}}]
    _kept, _dropped = decontaminate(_rows, DEFAULT_PROBES, ngram_threshold=0.5, cos_threshold=0.95)
    ok &= check("decontam drops the contaminated row", len(_dropped) == 1 and len(_kept) == 1)
    ok &= check("decontam keeps the unrelated row", _kept[0]["query"].startswith("Completely unrelated"))
    from autoscientist_toolcaller.dedup import _embed_model
    ok &= check("embed model cached (same object)", _embed_model() is _embed_model())

    print("schema-drift eval + worst-seed (#7):")
    _rec = [{"tools": [_W], "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {"sd_kind": "rename"}}]
    _m = evaluate(_rec, lambda p: target_to_json_str(_rec[0]["answer"]))
    ok &= check("evaluate reports by_sd_kind", "by_sd_kind" in _m and "rename" in _m["by_sd_kind"])
    ok &= check("schema_drift_accuracy present", _m.get("schema_drift_accuracy") == 1.0)
    _agg = {"seeds": 2, "base_acc": {"mean": 0.5, "std": 0.0}, "ft_acc": {"mean": 0.7, "std": 0.0}, "gap": {"mean": 0.2, "std": 0.05}, "worst_seed_overall_gap": 0.15, "worst_seed_index": 1}
    from autoscientist_toolcaller.eval_decompose import seeds_to_markdown as _sm
    ok &= check("worst-seed in banner", "worst seed" in _sm(_agg))

    print("release preflight (#8):")
    from autoscientist_toolcaller.release import preflight
    import tempfile as _tf, os as _os2
    _td = _tf.mkdtemp()
    for _n in ("train.jsonl", "val.jsonl", "test.jsonl", "stats.json"):
        open(_os2.path.join(_td, _n), "w").write("{}\n")
    ok &= check("preflight flags placeholder card", any("placeholder" in p for p in preflight(_td, card_paths=["clean text with YOUR_USERNAME here\n"], check_manifest=False)))
    ok &= check("preflight passes clean card + artifacts", preflight(_td, card_paths=["all real numbers, no markers\n"], check_manifest=False) == [])
    ok &= check("preflight flags unfilled __PENDING__ card", any("__PENDING__" in p for p in preflight(_td, card_paths=["overall_accuracy value: __PENDING__\n"], check_manifest=False)))
    ok &= check("preflight allows a real 0.000 metric (no false-positive)", preflight(_td, card_paths=["Hallucination rate 0.000 after fine-tuning; overall 0.912\n"], check_manifest=False) == [])
    _td2 = _tf.mkdtemp()  # missing artifacts
    ok &= check("preflight flags missing artifact", any("missing artifact" in p for p in preflight(_td2, card_paths=["clean\n"], check_manifest=False)))

    print("reproducibility manifest:")
    from autoscientist_toolcaller.manifest import write as _mwrite, verify as _mverify
    _mpath = _os2.path.join(_td, "manifest.json")
    _mwrite(out_dir=_td, config_path="config.yaml", manifest_path=_mpath)
    ok &= check("manifest verifies clean", _mverify(_td, _mpath) == [])
    open(_os2.path.join(_td, "train.jsonl"), "w").write('{"tampered":1}\n')
    ok &= check("manifest detects tampered artifact", any("changed" in p for p in _mverify(_td, _mpath)))
    ok &= check("manifest missing -> flagged", _mverify(_td, _mpath + ".nope") != [])
    ok &= check("preflight blocks on manifest mismatch", any("changed" in p or "manifest" in p for p in preflight(_td, card_paths=["clean\n"], manifest_path=_mpath)))

    print("fetch_improved (platform deliverable):")
    from autoscientist_toolcaller.fetch_improved import extract_enhanced
    _raw = [{"prompt": "P", "completion": "C", "enhanced_prompt": "EP", "enhanced_completion": "EC", "row_embedding": [0.1]},
            {"prompt": "P2", "completion": "C2"}]  # second row has no enhanced fields
    _enh = list(extract_enhanced(_raw, prefer_enhanced=True))
    ok &= check("uses enhanced pair when present", _enh[0] == {"prompt": "EP", "completion": "EC"})
    ok &= check("falls back to original when no enhanced", _enh[1] == {"prompt": "P2", "completion": "C2"})
    ok &= check("enhanced extract drops embeddings", "row_embedding" not in _enh[0])
    _orig = list(extract_enhanced(_raw, prefer_enhanced=False))
    ok &= check("original mode ignores enhanced", _orig[0] == {"prompt": "P", "completion": "C"})

    print("multilingual split leak-guard (pair_id grouping):")
    from autoscientist_toolcaller.build_dataset import split as _bsplit
    _ml_rows = _ml.generate(50, seed=3)  # matched twins, one pair_id per 5-language set
    _solo = [{"tools": [_W], "query": f"q{i}", "answer": {"type": "refuse", "content": "x"},
              "meta": {"source": "toolace", "hn_kind": None}} for i in range(40)]
    _ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
    _parts = _bsplit(_ml_rows + _solo, _ratios, seed=7)
    _placement: dict = {}
    for _name, _rows in _parts.items():
        for _r in _rows:
            _pid = _r["meta"].get("pair_id")
            if _pid is not None:
                _placement.setdefault(_pid, set()).add(_name)
    ok &= check("no multilingual pair_id straddles splits", bool(_placement) and all(len(v) == 1 for v in _placement.values()))
    ok &= check("split preserves every row", sum(len(v) for v in _parts.values()) == len(_ml_rows) + len(_solo))
    ok &= check("split deterministic", {k: len(v) for k, v in _bsplit(_ml_rows + _solo, _ratios, seed=7).items()}
                == {k: len(v) for k, v in _parts.items()})

    print("BFCL parallel bijection matcher:")
    _f = {"name": "f", "description": "f", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}
    # gold: G1 accepts x in {1,2}; G2 requires x==1. pred [x=1, x=2] has a valid 1-to-1 (P->G2, P->G1)
    # that GREEDY first-hit would miss -> the bijection matcher must accept it.
    _bex = {"tools": [_f], "query": "do both", "answer": {"type": "tool_call", "calls": [
        {"name": "f", "arguments": {"x": {"_acceptable": [1, 2]}}}, {"name": "f", "arguments": {"x": 1}}]}, "meta": {}}
    _bpred = target_to_json_str({"type": "tool_call", "calls": [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]})
    ok &= check("bijection accepts valid match greedy would miss", judge_bfcl(_bex, _bpred)["correct"])
    # over-credit guard: two DISTINCT golds, but a DUPLICATED prediction must NOT satisfy both.
    _bex2 = {"tools": [_f], "query": "two", "answer": {"type": "tool_call", "calls": [
        {"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": {"_acceptable": [1, 2]}}}]}, "meta": {}}
    _dup = target_to_json_str({"type": "tool_call", "calls": [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 1}}]})
    ok &= check("duplicated call not credited to two distinct golds", not judge_bfcl(_bex2, _dup)["correct"])
    _ok2 = target_to_json_str({"type": "tool_call", "calls": [{"name": "f", "arguments": {"x": 1}}, {"name": "f", "arguments": {"x": 2}}]})
    ok &= check("distinct correct parallel still passes", judge_bfcl(_bex2, _ok2)["correct"])

    print("multilingual Δaccuracy evaluator:")
    from autoscientist_toolcaller.eval_multilingual import _delta_metrics, evaluate_multilingual
    _oracle_bits = {(p, l): 1 for p in range(1, 6) for l in ("en", "hi", "es")}
    _dm = _delta_metrics(_oracle_bits)
    ok &= check("Δ metrics: languages present", set(_dm["languages"]) == {"en", "hi", "es"})
    ok &= check("Δ metrics: oracle Δ(hi−en) == 0", _dm["matched_pair_delta_vs_en"]["hi"]["delta_vs_en"] == 0.0)
    _fail_hi = {(p, l): (0 if l == "hi" else 1) for p in range(1, 6) for l in ("en", "hi")}
    _dmf = _delta_metrics(_fail_hi)
    ok &= check("Δ metrics: fails-only-hi -> Δ(hi−en) == -1", _dmf["matched_pair_delta_vs_en"]["hi"]["delta_vs_en"] == -1.0)
    ok &= check("Δ metrics: hi accuracy 0.0", _dmf["by_lang"]["hi"]["accuracy"] == 0.0)
    _mlrecs = _ml.generate(30, seed=5)
    _outs = iter([target_to_json_str(r["answer"]) for r in _mlrecs])
    _mm = evaluate_multilingual(_mlrecs, lambda p: next(_outs))
    ok &= check("evaluate_multilingual: 5 languages measured", set(_mm["languages"]) >= {"en", "hi", "hi-rom", "es", "fr"})
    ok &= check("evaluate_multilingual: oracle -> every lang acc 1.0", all(v["accuracy"] == 1.0 for v in _mm["by_lang"].values()))
    ok &= check("evaluate_multilingual: oracle -> Δ == 0 all langs", all(d["delta_vs_en"] == 0.0 for d in _mm["matched_pair_delta_vs_en"].values()))

    print("BFCL weighted aggregate + expanded decontam probes (rigor #3):")
    from autoscientist_toolcaller.eval_bfcl import weighted_accuracy, BFCL_WEIGHTS
    ok &= check("BFCL weights sum to 1.0", abs(sum(BFCL_WEIGHTS.values()) - 1.0) < 1e-9)
    _wa = weighted_accuracy({"simple": {"accuracy": 1.0}, "multi_turn": {"accuracy": 0.0}})
    ok &= check("weighted_accuracy renormalizes over present cats", abs(_wa - (0.12 / 0.32)) < 1e-9)
    ok &= check("weighted_accuracy None when no data", weighted_accuracy({"simple": {"accuracy": None}}) is None)
    from autoscientist_toolcaller.decontaminate import DEFAULT_PROBES as _PROBES, decontaminate as _decon
    ok &= check("decontam probe fixture expanded (>60)", len(_PROBES) > 60)
    _dk, _dd = _decon([{"query": _PROBES[7], "meta": {"source": "x"}}], _PROBES)  # verbatim probe as a train row
    ok &= check("verbatim probe dropped at DEFAULT thresholds", len(_dd) == 1 and len(_dk) == 0)
    _dk2, _dd2 = _decon([{"query": "totally unrelated: refactor this legacy COBOL payroll module", "meta": {"source": "x"}}], _PROBES)
    ok &= check("clearly-unrelated query kept", len(_dk2) == 1 and len(_dd2) == 0)

    print("by-difficulty eval breakdown (curriculum wiring):")
    _recs_d = [
        {"tools": [_W], "query": "weather in Mumbai?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {}},
        {"tools": [_W, _S], "query": "write me a poem about the sea", "answer": {"type": "refuse", "content": "no tool applies"}, "meta": {}},
    ]
    _md = evaluate(_recs_d, lambda p: target_to_json_str(_recs_d[0]["answer"]) if "weather" in p.lower() else '{"action":"refuse","message":"none apply"}')
    ok &= check("evaluate emits by_difficulty", "by_difficulty" in _md)
    ok &= check("by_difficulty bands cover all records", sum(v["n"] for v in _md["by_difficulty"].values()) == 2)
    ok &= check("by_difficulty accuracies valid", all(0.0 <= v["accuracy"] <= 1.0 for v in _md["by_difficulty"].values()))

    print("agentic trajectories (multi-step, observation-in-the-loop):")
    from autoscientist_toolcaller import agentic as _ag
    from autoscientist_toolcaller.agentic import env_by_name as _env_by_name
    _trajs = _ag.generate_trajectories(30, seed=7)
    ok &= check("agentic trajectories generated", len(_trajs) > 0)
    ok &= check("trajectories have >=2 steps", all(len(t["steps"]) >= 2 for t in _trajs))
    # every gold CALL step is execution-valid on its pre_state (correct by construction)
    _valid = True
    for t in _trajs:
        env = _env_by_name(t["env"])
        for s in t["steps"]:
            if s["gold_action"] == "call":
                _, _ok, _ = env.apply(s["pre_state"], s["answer"]["calls"][0])
                _valid &= _ok
    ok &= check("every gold call step is execution-verified", _valid)
    ok &= check("recovery trajectories end in clarify", all(
        t["steps"][-1]["gold_action"] == "clarify" for t in _trajs if t["kind"] == "recovery") or
        not any(t["kind"] == "recovery" for t in _trajs))
    ok &= check("agentic determinism", [ (x["goal"], len(x["steps"])) for x in _ag.generate_trajectories(30, seed=7)] ==
                [ (x["goal"], len(x["steps"])) for x in _trajs])
    _ex = _ag.to_examples(_trajs[0])
    ok &= check("to_examples: source=agentic + pair_id + history", all(
        e["meta"]["source"] == "agentic" and e["meta"].get("pair_id") and
        (e.get("history") is None or isinstance(e["history"], list)) for e in _ex))

    print("agentic eval (rollout success + per-step):")
    from autoscientist_toolcaller.eval_agentic import evaluate_agentic, rollout
    _clean = next(t for t in _trajs if t["kind"] == "clean")
    def _oracle(traj):
        _it = iter([target_to_json_str(s["answer"]) for s in traj["steps"]])
        return lambda p: next(_it)
    _r = rollout(_clean, _oracle(_clean))
    ok &= check("oracle rollout succeeds on clean trajectory", _r["success"] and all(_r["per_step"]))
    _rr = rollout(_clean, lambda p: '{"action":"refuse","message":"no"}')
    ok &= check("always-refuse fails the trajectory", not _rr["success"])
    _rec = next((t for t in _trajs if t["kind"] == "recovery"), None)
    if _rec is not None:
        ok &= check("oracle rollout succeeds on recovery (abstains on impossible step)",
                    rollout(_rec, _oracle(_rec))["success"])
        # a model that blindly CALLS on the impossible final step must FAIL the recovery trajectory
        _blind = iter([target_to_json_str(s["answer"]) for s in _rec["steps"][:-1]] +
                      [target_to_json_str({"type": "tool_call", "calls": [_rec["steps"][0]["answer"]["calls"][0]]})])
        ok &= check("blindly calling the impossible step fails recovery", not rollout(_rec, lambda p: next(_blind))["success"])
    _m = evaluate_agentic([_clean], _oracle(_clean))
    ok &= check("evaluate_agentic: oracle success_rate == 1.0", _m["trajectory_success_rate"] == 1.0)

    print("scaled DPO axes (over-refusal / partial-parallel / agentic-step):")
    from autoscientist_toolcaller.build_preference import _axis_pairs, _confirmed_wrong
    import random as _rp2
    _rngp = _rp2.Random(0)
    _orx = {"tools": [_W], "query": "you could maybe check the weather in Mumbai",
            "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]},
            "meta": {"hn_kind": "over_refusal"}}
    _ap = _axis_pairs(_orx, _rngp)
    ok &= check("over-refusal axis pair: chosen=call, rejected=refuse", len(_ap) == 1 and
                json.loads(_ap[0]["chosen"])["action"] == "call" and json.loads(_ap[0]["rejected"])["action"] == "refuse")
    _pp2 = {"tools": [_W, _S], "query": "weather in Mumbai and price of AAPL",
            "answer": {"type": "tool_call", "calls": [
                {"name": "get_weather", "arguments": {"city": "Mumbai"}},
                {"name": "get_stock", "arguments": {"ticker": "AAPL"}}]},
            "meta": {"hn_kind": "partial_parallel"}}
    _pap = _axis_pairs(_pp2, _rngp)
    ok &= check("partial-parallel axis pair: rejected drops a call", len(_pap) == 1 and
                len(json.loads(_pap[0]["rejected"])["calls"]) == 1)
    ok &= check("poison guard: refuse rejected for a positive is confirmed-wrong",
                _confirmed_wrong(_orx, {"action": "refuse", "message": "x"}))
    ok &= check("poison guard: wrong call-count is confirmed-wrong",
                _confirmed_wrong(_pp2, {"action": "call", "calls": [_pp2["answer"]["calls"][0]]}))
    ok &= check("poison guard: gold-identical is NOT wrong",
                not _confirmed_wrong(_pp2, {"action": "call", "calls": _pp2["answer"]["calls"]}))
    _agd = _ag.generate_dpo(10, seed=1)
    ok &= check("agentic-step DPO pairs built (chosen != rejected)", len(_agd) > 0 and all(d["chosen"] != d["rejected"] for d in _agd))

    print("BFCL agentic category + calibration/abstention metrics:")
    ok &= check("categorize -> 'agentic' for agentic-meta example", categorize(
        {"tools": [_W], "history": [{"role": "user", "content": "x"}],
         "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "X"}}]},
         "meta": {"source": "agentic"}}) == "agentic")
    _recs_cal = [
        {"tools": [_W], "query": "weather in Mumbai", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {}},
        {"tools": [_W], "query": "could you maybe get the weather in Delhi", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Delhi"}}]}, "meta": {}},
        {"tools": [_W], "query": "write a poem about rain", "answer": {"type": "refuse", "content": "no tool"}, "meta": {}},
    ]
    _cal = evaluate(_recs_cal, lambda p: target_to_json_str(_recs_cal[0]["answer"]) if "Mumbai" in p else '{"action":"refuse","message":"x"}')["calibration"]
    ok &= check("confusion matrix cells sum to N", sum(sum(row.values()) for row in _cal["confusion"].values()) == 3)
    ok &= check("over-refusal rate correct (1 of 2 gold-calls refused)", abs(_cal["over_refusal_rate"] - 0.5) < 1e-9)
    ok &= check("abstention recall correct (refused when should)", _cal["abstention_recall"] == 1.0)

    print("LLM-judge synth loop (mockable, offline):")
    from autoscientist_toolcaller.synth_llm import synthesize as _synth
    def _is_crit(system):
        return "critique" in system.lower() or "judge" in system.lower()
    _valid_gen = '{"query":"What is the weather in Mumbai?","answer":{"type":"tool_call","calls":[{"name":"get_weather","arguments":{"city":"Mumbai"}}]}}'
    _fake_ok = lambda s, u: ('{"ok": true, "reason": "hard+correct"}' if _is_crit(s) else _valid_gen)
    _fake_badschema = lambda s, u: ('{"ok": true}' if _is_crit(s) else '{"query":"z","answer":{"type":"tool_call","calls":[{"name":"nonexistent_tool","arguments":{}}]}}')
    _fake_critno = lambda s, u: ('{"ok": false, "reason": "too easy"}' if _is_crit(s) else _valid_gen)
    _sv = _synth(1, [_W], complete_fn=_fake_ok, max_attempts=6)
    ok &= check("synth: a valid critiqued+verified case survives", len(_sv) == 1 and _sv[0]["meta"]["source"] == "llm_synth")
    ok &= check("synth: schema-invalid gold dropped by verify gate", _synth(1, [_W], complete_fn=_fake_badschema, max_attempts=6) == [])
    ok &= check("synth: critique rejection drops the case", _synth(1, [_W], complete_fn=_fake_critno, max_attempts=6) == [])
    ok &= check("synth: duplicate queries deduped", len(_synth(3, [_W], complete_fn=_fake_ok, max_attempts=20)) == 1)

    print("agent runtime (goal -> call -> observe -> repeat):")
    from autoscientist_toolcaller.agent import run_agent, safe_tools_registry, env_registry
    from autoscientist_toolcaller.envs import CartEnv as _CartEnv

    def _seq(*answers):
        _it = iter([target_to_json_str(a) for a in answers])
        return lambda p: next(_it)

    _r1 = run_agent("compute 2+3", safe_tools_registry(), _seq(
        {"type": "tool_call", "calls": [{"name": "add", "arguments": {"a": 2, "b": 3}}]},
        {"type": "tool_call", "calls": [{"name": "finish", "arguments": {"answer": "5"}}]}), max_steps=5)
    ok &= check("agent executes a real tool + finishes", _r1["status"] == "done" and _r1["result"] == "5"
                and "OK 5" in _r1["transcript"][0]["observation"])
    _r2 = run_agent("x", safe_tools_registry(), _seq(
        {"type": "tool_call", "calls": [{"name": "nope", "arguments": {}}]},
        {"type": "tool_call", "calls": [{"name": "finish", "arguments": {"answer": "x"}}]}), max_steps=5)
    ok &= check("agent surfaces unknown-tool error as an observation", "unknown tool" in _r2["transcript"][0]["observation"])
    _r3 = run_agent("impossible", safe_tools_registry(), lambda p: '{"action":"refuse","message":"no tool"}', max_steps=5)
    ok &= check("agent stops on refuse (abstention is terminal)", _r3["status"] == "refuse")
    _er = env_registry(_CartEnv())
    _r4 = run_agent("manage cart", _er, _seq(
        {"type": "tool_call", "calls": [{"name": "add_item", "arguments": {"item": "apples", "quantity": 3}}]},
        {"type": "tool_call", "calls": [{"name": "remove_item", "arguments": {"item": "apples", "quantity": 10}}]},
        {"type": "tool_call", "calls": [{"name": "finish", "arguments": {"answer": "done"}}]}), max_steps=6)
    ok &= check("agent executes stateful env tools; over-remove errors, state intact",
                _r4["status"] == "done" and "ERROR" in _r4["transcript"][1]["observation"]
                and _er._state["s"]["items"].get("apples") == 3)
    _r5 = run_agent("loop", safe_tools_registry(),
                    lambda p: target_to_json_str({"type": "tool_call", "calls": [{"name": "add", "arguments": {"a": 1, "b": 1}}]}),
                    max_steps=3)
    ok &= check("agent respects the step budget", _r5["status"] == "max_steps" and _r5["steps"] == 3)

    print("report + model-card wiring (agentic / calibration / multilingual):")
    from autoscientist_toolcaller.eval_report import render_agentic, render_calibration, render_multilingual
    from autoscientist_toolcaller.fill_model_card import build_metrics_block
    _agm = {"n": 5, "trajectory_success_rate": 0.8, "per_step_accuracy": 0.9, "avg_steps": 3.0, "by_env": {"cart": {"success_rate": 0.8, "n": 5}}}
    ok &= check("report renders agentic block", "trajectory success" in render_agentic(_agm) and "cart" in render_agentic(_agm))
    _ftc = {"calibration": {"over_refusal_rate": 0.1, "abstention_precision": 0.9, "abstention_recall": 0.8}}
    ok &= check("report renders calibration block", "over-refusal" in render_calibration(_ftc))
    _mlm = {"languages": ["en", "hi"], "by_lang": {"en": {"accuracy": 0.9}, "hi": {"accuracy": 0.8}},
            "matched_pair_delta_vs_en": {"hi": {"delta_vs_en": -0.1}}}
    ok &= check("report renders multilingual Δ block", "hi" in render_multilingual(_mlm) and "vs en" in render_multilingual(_mlm))
    _blk = build_metrics_block({}, _ftc, {}, agentic=_agm, multilingual=_mlm)
    ok &= check("model card includes agentic + Δ + over-refusal rows",
                "Agentic trajectory success" in _blk and "Δacc(hi−en)" in _blk and "Over-refusal rate" in _blk)

    print("DPO training config (config-driven, CLI-overridable):")
    from autoscientist_toolcaller.train_dpo import _dpo_params
    import types as _types
    _dp = _dpo_params({"beta": 0.2, "lora_r": 8}, _types.SimpleNamespace(beta=None, lr=None, epochs=None))
    ok &= check("dpo params: config value used, defaults filled", _dp["beta"] == 0.2 and _dp["lora_r"] == 8 and _dp["lora_alpha"] == 32 and _dp["learning_rate"] == 5e-6)
    _dp2 = _dpo_params({"beta": 0.2}, _types.SimpleNamespace(beta=0.05, lr=None, epochs=3.0))
    ok &= check("dpo params: CLI overrides config", _dp2["beta"] == 0.05 and _dp2["epochs"] == 3.0)
    import yaml as _yaml
    _cfgd = _yaml.safe_load(open("config.yaml"))
    ok &= check("config.yaml has a dpo section", isinstance(_cfgd.get("dpo"), dict) and _cfgd["dpo"]["beta"] == 0.1)

    print("real agent tools (sandboxed filesystem):")
    from autoscientist_toolcaller.agent_tools import sandbox_fs_registry
    import tempfile as _tf3, os as _os3
    _sbx = _tf3.mkdtemp()
    open(_os3.path.join(_sbx, "notes.txt"), "w", encoding="utf-8").write("hello agent world")
    _fsreg = sandbox_fs_registry(_sbx, allow_write=True)
    ok &= check("sandbox rejects path traversal", not _fsreg.call("read_file", {"path": "../../etc/hosts"})[0])
    _okr, _res = _fsreg.call("read_file", {"path": "notes.txt"})
    ok &= check("sandbox read_file returns content", _okr and "hello agent world" in _res)
    _rfs = run_agent("read notes.txt", _fsreg, _seq(
        {"type": "tool_call", "calls": [{"name": "read_file", "arguments": {"path": "notes.txt"}}]},
        {"type": "tool_call", "calls": [{"name": "finish", "arguments": {"answer": "done"}}]}), max_steps=4)
    ok &= check("agent reads a real file via the sandbox tool", "hello agent world" in _rfs["transcript"][0]["observation"])
    _okw, _ = _fsreg.call("write_file", {"path": "out.txt", "content": "written by agent"})
    _okr2, _res2 = _fsreg.call("read_file", {"path": "out.txt"})
    ok &= check("sandbox write+read round-trip", _okw and _okr2 and _res2 == "written by agent")
    ok &= check("sandbox list_dir sees files", "notes.txt" in _fsreg.call("list_dir", {})[1])

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
