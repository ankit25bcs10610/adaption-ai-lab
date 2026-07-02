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
    from src.eval_bfcl import args_match_lenient, categorize, judge_bfcl, normalize_value
    ok &= check("categorize simple", categorize({"tools": [TOOLS[0]], "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "simple")
    ok &= check("categorize multiple", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {}}]}}) == "multiple")
    ok &= check("categorize parallel", categorize({"tools": TOOLS, "answer": {"type": "tool_call", "calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]}}) == "parallel")
    ok &= check("categorize irrelevance", categorize({"tools": TOOLS, "answer": {"type": "refuse", "content": "x"}}) == "irrelevance")
    ok &= check("normalize num string", normalize_value("42") == 42.0)
    ok &= check("lenient case/space match", args_match_lenient({"city": " New York "}, {"city": "new york"}))
    ok &= check("lenient acceptable list", args_match_lenient({"city": "NYC"}, {"city": {"_acceptable": ["nyc", "new york city"]}}))
    ok &= check("lenient judge correct", judge_bfcl(pos, target_to_json_str(pos["answer"]))["correct"])

    print("bfcl_handler (envelope translation):")
    from src.bfcl_handler import translate, is_no_call
    ok &= check("call -> decoded ast", translate('{"action":"call","calls":[{"name":"f","arguments":{"a":1}}]}') == [{"f": {"a": 1}}])
    ok &= check("refuse -> empty (no call)", translate('{"action":"refuse","message":"x"}') == [])
    ok &= check("is_no_call on clarify", is_no_call('{"action":"clarify","message":"x"}'))
    ok &= check("think block firewalled in handler", translate('<think>{z}</think>{"action":"call","calls":[{"name":"g","arguments":{}}]}') == [{"g": {}}])

    print("export_bfcl:")
    from src.export_bfcl import to_bfcl_prompt, to_bfcl_answer
    _ex = {"tools": TOOLS, "query": "weather?", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}}
    _p = to_bfcl_prompt(_ex, "BFCL_v4_simple_0")
    _a = to_bfcl_answer(_ex, "BFCL_v4_simple_0")
    ok &= check("prompt has question turns", _p["question"] == [[{"role": "user", "content": "weather?"}]])
    ok &= check("answer wraps values in lists", _a["ground_truth"] == [{"get_weather": {"city": ["Mumbai"]}}])
    ok &= check("irrelevance answer empty", to_bfcl_answer({"tools": TOOLS, "query": "poem?", "answer": {"type": "refuse", "content": "x"}}, "i")["ground_truth"] == [])

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
    # Poison guard: no pair may have rejected == chosen, and every positive's rejected is a real error.
    ok &= check("no poison pair (chosen==rejected)", all(p["chosen"] != p["rejected"] for p in pairs))
    from src.build_preference import _confirmed_wrong
    ok &= check("positive rejected is confirmed wrong", _confirmed_wrong(pos, json.loads(pairs[0]["rejected"])))

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

    print("robustness_table:")
    from src.robustness_table import robustness
    base_p = ([{"category": "simple", "correct": True}] * 9 + [{"category": "simple", "correct": False}]
              + [{"category": "multi_turn", "correct": True}] * 4 + [{"category": "multi_turn", "correct": False}] * 6)
    ft_p = ([{"category": "simple", "correct": True}] * 9 + [{"category": "simple", "correct": False}]
            + [{"category": "multi_turn", "correct": True}] * 8 + [{"category": "multi_turn", "correct": False}] * 2)
    r = robustness(base_p, ft_p)
    ok &= check("robustness computes drops", r["mean_base_drop"] is not None and r["mean_ft_drop"] is not None)
    ok &= check("ft more robust (smaller drop)", r["mean_ft_drop"] < r["mean_base_drop"])

    print("eval_decompose (gap decomposition):")
    from src.eval_decompose import decompose, aggregate_seeds, to_html as dec_html
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

    print("reliability_probe:")
    from src.reliability_probe import probe_cases, score, to_markdown as probe_md
    from src.format_utils import build_eval_prompt as _bep, target_to_json_str as _tjs
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
    from src.envs import CalendarEnv, CartEnv
    from src.envs import generate as env_gen, generate_dpo as env_dpo
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
    from src.format_utils import sample_value
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
    from src.envs import generate_multicall
    _mc = generate_multicall(8, seed=1)
    ok &= check("multicall 2-3 verified calls", _mc and all(2 <= len(e["answer"]["calls"]) <= 3 and validate_answer(e["answer"], e["tools"])[0] for e in _mc))
    ok &= check("multicall drop-one is WRONG", not judge(_mc[0], target_to_json_str({"type": "tool_call", "calls": _mc[0]["answer"]["calls"][:-1]}))["correct"])
    ok &= check("multicall determinism", generate_multicall(8, seed=1) == _mc)

    print("pref hardness + env merge (#5):")
    from src.build_preference import _hardness
    _chosen = {"action": "call", "calls": [{"name": "f", "arguments": {"a": 1, "b": 2}}]}
    _nearmiss = {"action": "call", "calls": [{"name": "f", "arguments": {"a": 1, "b": 9}}]}
    _swap = {"action": "call", "calls": [{"name": "g", "arguments": {"a": 1, "b": 2}}]}
    ok &= check("near-miss harder (smaller dist) than tool swap", _hardness(_chosen, _nearmiss) < _hardness(_chosen, _swap))

    print("decontamination (#6):")
    from src.decontaminate import decontaminate, DEFAULT_PROBES
    _rows = [{"query": DEFAULT_PROBES[0], "meta": {"source": "toolace"}},
             {"query": "Completely unrelated: parse this XML config file for me.", "meta": {"source": "toolace"}}]
    _kept, _dropped = decontaminate(_rows, DEFAULT_PROBES, ngram_threshold=0.5, cos_threshold=0.95)
    ok &= check("decontam drops the contaminated row", len(_dropped) == 1 and len(_kept) == 1)
    ok &= check("decontam keeps the unrelated row", _kept[0]["query"].startswith("Completely unrelated"))
    from src.dedup import _embed_model
    ok &= check("embed model cached (same object)", _embed_model() is _embed_model())

    print("schema-drift eval + worst-seed (#7):")
    _rec = [{"tools": [_W], "query": "x", "answer": {"type": "tool_call", "calls": [{"name": "get_weather", "arguments": {"city": "Mumbai"}}]}, "meta": {"sd_kind": "rename"}}]
    _m = evaluate(_rec, lambda p: target_to_json_str(_rec[0]["answer"]))
    ok &= check("evaluate reports by_sd_kind", "by_sd_kind" in _m and "rename" in _m["by_sd_kind"])
    ok &= check("schema_drift_accuracy present", _m.get("schema_drift_accuracy") == 1.0)
    _agg = {"seeds": 2, "base_acc": {"mean": 0.5, "std": 0.0}, "ft_acc": {"mean": 0.7, "std": 0.0}, "gap": {"mean": 0.2, "std": 0.05}, "worst_seed_overall_gap": 0.15, "worst_seed_index": 1}
    from src.eval_decompose import seeds_to_markdown as _sm
    ok &= check("worst-seed in banner", "worst seed" in _sm(_agg))

    print("release preflight (#8):")
    from src.release import preflight
    import tempfile as _tf, os as _os2
    _td = _tf.mkdtemp()
    for _n in ("train.jsonl", "val.jsonl", "test.jsonl", "stats.json"):
        open(_os2.path.join(_td, _n), "w").write("{}\n")
    ok &= check("preflight flags placeholder card", any("placeholder" in p for p in preflight(_td, card_paths=["clean text with YOUR_USERNAME here\n"], check_manifest=False)))
    ok &= check("preflight passes clean card + artifacts", preflight(_td, card_paths=["all real numbers, no markers\n"], check_manifest=False) == [])
    _td2 = _tf.mkdtemp()  # missing artifacts
    ok &= check("preflight flags missing artifact", any("missing artifact" in p for p in preflight(_td2, card_paths=["clean\n"], check_manifest=False)))

    print("reproducibility manifest:")
    from src.manifest import write as _mwrite, verify as _mverify
    _mpath = _os2.path.join(_td, "manifest.json")
    _mwrite(out_dir=_td, config_path="config.yaml", manifest_path=_mpath)
    ok &= check("manifest verifies clean", _mverify(_td, _mpath) == [])
    open(_os2.path.join(_td, "train.jsonl"), "w").write('{"tampered":1}\n')
    ok &= check("manifest detects tampered artifact", any("changed" in p for p in _mverify(_td, _mpath)))
    ok &= check("manifest missing -> flagged", _mverify(_td, _mpath + ".nope") != [])
    ok &= check("preflight blocks on manifest mismatch", any("changed" in p or "manifest" in p for p in preflight(_td, card_paths=["clean\n"], manifest_path=_mpath)))

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
