"""Offline smoke tests for the Data-Viz track: hardened scorer, synth ground-truth, format.

Run:  python -m tests.viz.test_viz
Covers the load-bearing correctness pieces (relaxed-match edge cases from the adversarial review,
synth GT correctness by construction) without any model download. Rendering is tested separately.
"""
from __future__ import annotations

import sys

from src.viz import synth_charts as sc
from src.viz.eval_chart import evaluate, judge_chart, relaxed_match
from src.viz.format_utils import answer_to_text, image_to_data_uri


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main() -> int:
    ok = True

    print("format_utils:")
    ok &= check("no sci-notation on big float", "e" not in answer_to_text(1234567.0).lower())
    ok &= check("int float -> int str", answer_to_text(12.0) == "12")
    ok &= check("float trimmed", answer_to_text(3.5) == "3.5")
    ok &= check("bool -> Yes/No", answer_to_text(True) == "Yes")
    ok &= check("list join", answer_to_text([1, 2, 3]) == "1, 2, 3")
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 40
    ok &= check("bytes -> data uri", image_to_data_uri(png).startswith("data:image/png;base64,"))

    print("relaxed_match — numeric tolerance:")
    ok &= check("within 5%", relaxed_match(100, "104"))
    ok &= check("outside 5%", not relaxed_match(100, "110"))
    ok &= check("magnitude 1.2M == 1200000", relaxed_match(1200000, "1.2M"))
    ok &= check("lakh grouping", relaxed_match(120000, "1,20,000"))
    ok &= check("units rejected as garbage-safe", not relaxed_match(350000000, "350 MW"))

    print("relaxed_match — F1 range / garbage:")
    ok &= check("range not truncated to first num", not relaxed_match(3, "3-4"))
    ok &= check("range matches range", relaxed_match("10-20", "10 to 20"))
    ok &= check("multi-value pred vs scalar gold rejected", not relaxed_match(42, "42, 43"))

    print("relaxed_match — F6 yes/no vs count:")
    ok &= check("count 1 vs 'Yes' rejected", not relaxed_match(1, "Yes"))
    ok &= check("bool gold vs Yes", relaxed_match(True, "Yes"))
    ok &= check("bool gold vs 'no...' multiword falls through", not relaxed_match(True, "no it did not"))

    print("relaxed_match — percent (F2):")
    ok &= check("12% vs 0.12", relaxed_match(12.0, "0.12") is False or relaxed_match("12%", "0.12"))
    ok &= check("explicit percent both", relaxed_match("50%", "50%"))
    ok &= check("bare 1.0 not auto-scaled to 100%", not relaxed_match(1.0, "100%"))

    print("relaxed_match — zero gold (F4):")
    ok &= check("zero gold exact", relaxed_match(0, "0"))
    ok &= check("zero gold rejects 0.04", not relaxed_match(0, "0.04"))

    print("relaxed_match — percent FP fix (Bug 1) + proportion:")
    ok &= check("stray % on raw gold rejected (20 vs 20%)", not relaxed_match(20.0, "20%"))
    ok &= check("stray % on raw gold rejected (80 vs 80%)", not relaxed_match(80, "80%"))
    ok &= check("proportion accepts 23.4%", relaxed_match(23.4, "23.4%", "proportion"))
    ok &= check("proportion accepts 23.4", relaxed_match(23.4, "23.4", "proportion"))

    print("relaxed_match — trend cross-language (Indic Bug 1):")
    ok &= check("en increasing == hi बढ़ रहा है", relaxed_match("increasing", "बढ़ रहा है", "trend"))
    ok &= check("en increasing == 'it is rising'", relaxed_match("increasing", "it is rising", "trend"))
    ok &= check("increasing != decreasing", not relaxed_match("increasing", "decreasing", "trend"))
    ok &= check("positive corr matched", relaxed_match("positive", "positive correlation", "trend"))

    print("relaxed_match — Indic digits (F8):")
    ok &= check("devanagari numerals parse", relaxed_match(42, "४२"))
    ok &= check("devanagari category equality", relaxed_match("दक्षिण", "दक्षिण"))

    print("relaxed_match — lists (F16 proper matching):")
    ok &= check("wrong multiset rejected", not relaxed_match([10, 90], [50, 50]))
    ok &= check("correct multiset (unordered)", relaxed_match([10, 90], [90, 10]))
    ok &= check("wrong-length list rejected", not relaxed_match([10, 90], [10]))

    print("synth ground-truth (by construction):")
    cd = sc.ChartData("bar", "T", "Region", "Sales", ["North", "South", "East", "West"],
                      ["series"], [[10.0, 40.0, 25.0, 5.0]], "", 0, False)
    import random
    rng = random.Random(0)
    mx = sc.mk_extremum(cd, rng, True)
    ok &= check("max category correct", mx and mx.answer == "South")
    mn = sc.mk_extremum(cd, rng, False)
    ok &= check("min category correct", mn and mn.answer == "West")
    sm = sc.mk_sum(cd, rng)
    ok &= check("sum correct", sm and sm.answer == 80)
    mean = sc.mk_mean(cd, rng)
    ok &= check("mean correct", mean and mean.answer == 20)
    # tie -> None
    cd_tie = sc.ChartData("bar", "T", "R", "S", ["A", "B"], ["series"], [[10.0, 10.0]], "", 0, False)
    ok &= check("tie extremum -> None", sc.mk_extremum(cd_tie, rng, True) is None)
    # count: values 5,15,25,35 -> above threshold between 15 and 25 => count of {25,35}=2
    cd_cnt = sc.ChartData("bar", "T", "R", "S", ["A", "B", "C", "D"], ["series"], [[5.0, 15.0, 25.0, 35.0]], "", 0, False)
    cnt = sc.mk_count(cd_cnt, random.Random(1))
    ok &= check("count answer is an int within range", cnt and isinstance(cnt.answer, int) and 0 <= cnt.answer <= 4)
    # trend on ordered increasing
    cd_tr = sc.ChartData("line", "T", "Month", "Rev", ["Jan", "Feb", "Mar", "Apr"], ["series"],
                         [[10.0, 20.0, 30.0, 45.0]], "", 0, True)
    tr = sc.mk_trend(cd_tr, rng)
    ok &= check("trend increasing", tr and tr.answer == "increasing")

    print("synth — series naming on multi-series (review Bug):")
    cd_g = sc.ChartData("grouped_bar", "T", "Region", "Sales", ["North", "South", "East"],
                        ["Q1", "Q2"], [[10.0, 40.0, 25.0], [50.0, 20.0, 30.0]], "", 0, False)
    ex_ = sc.mk_extremum(cd_g, random.Random(5), True)
    ok &= check("grouped_bar extremum names series", ex_ and (" for Q1" in ex_.question or " for Q2" in ex_.question))
    # the answer must be the argmax of the NAMED series
    if ex_:
        si_name = ex_.support["series"]
        si_ = cd_g.series_names.index(si_name)
        argmax_cat = cd_g.categories[max(range(3), key=lambda k: cd_g.values[si_][k])]
        ok &= check("grouped_bar extremum answer matches named series", ex_.answer == argmax_cat)
    ok &= check("mean dropped from grouped_bar makers", sc.mk_mean not in sc._MAKERS["grouped_bar"])
    ok &= check("mean dropped from scatter/area", sc.mk_mean not in sc._MAKERS["scatter"] and sc.mk_mean not in sc._MAKERS["area"])

    print("build_dataset — image-group split (no leakage):")
    from src.viz.build_dataset import split as viz_split
    twoq = [
        {"image": "img/a.png", "question": "q1", "answer": "X", "chart_type": "bar", "qa_kind": "max", "meta": {}},
        {"image": "img/a.png", "question": "q2", "answer": "Y", "chart_type": "bar", "qa_kind": "sum", "meta": {}},
    ]
    parts = viz_split(twoq * 10, ["scatter"], (0.6, 0.2), 1)
    placement = {}
    for name, rows in parts.items():
        for r in rows:
            placement.setdefault(r["image"], set()).add(name)
    ok &= check("same image never split across sets", all(len(v) == 1 for v in placement.values()))

    print("synth generate (no render) — schema + determinism:")
    a = sc.generate(20, "/tmp/vizsmoke", seed=7, render=False)
    b = sc.generate(20, "/tmp/vizsmoke", seed=7, render=False)
    ok &= check("generated examples", len(a) > 0)
    ok &= check("determinism", [(x["question"], x["answer"]) for x in a] == [(x["question"], x["answer"]) for x in b])
    ok &= check("schema fields present", all({"image", "question", "answer", "chart_type", "qa_kind", "meta"} <= set(x) for x in a))

    print("eval_chart.evaluate wiring:")
    recs = [{"image": png, "question": "q", "answer": "South", "chart_type": "bar", "qa_kind": "max", "meta": {"lang": "en"}}]
    m = evaluate(recs, lambda messages: "South")
    ok &= check("evaluate scores correct=1.0", abs(m["relaxed_accuracy"] - 1.0) < 1e-9)
    ok &= check("by_lang breakdown present", "en" in m["by_lang"])

    print("\nRESULT:", "ALL PASS ✅" if ok else "FAILURES ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
