"""Vega-Lite spec-reading — a text-only, correct-by-construction chart-QA modality.

Complements the image-based chart-QA: here the "chart" is a **Vega-Lite JSON spec** and the model must
answer from the spec's `data.values` (no pixels, no VLM). This is offline-evaluable with the existing
relaxed scorer, adds a second modality (spec comprehension), and every answer is computed from the same
data embedded in the spec — so labels are correct by construction.

`generate(n, seed)` returns rows: {spec (JSON str), question, answer, qa_kind, meta}.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List

_CATSETS = [
    ["Alpha", "Beta", "Gamma", "Delta"],
    ["North", "South", "East", "West"],
    ["Q1", "Q2", "Q3", "Q4"],
    ["Apples", "Bananas", "Cherries", "Dates", "Figs"],
    ["Jan", "Feb", "Mar", "Apr", "May"],
]
_YLABELS = ["revenue", "users", "sales", "score", "count"]
_MARKS = ["bar", "line", "point"]


def _distinct_ints(rng: random.Random, n: int) -> List[int]:
    """n distinct integers in [10,99] so max/min/compare are unambiguous."""
    return rng.sample(range(10, 100), n)


def _spec(cats: List[str], vals: List[int], ylabel: str, mark: str) -> Dict[str, Any]:
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "data": {"values": [{"category": c, ylabel: v} for c, v in zip(cats, vals)]},
        "mark": mark,
        "encoding": {
            "x": {"field": "category", "type": "nominal"},
            "y": {"field": ylabel, "type": "quantitative"},
        },
    }


def generate(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed + 1313)
    out: List[Dict[str, Any]] = []
    kinds = ["value_lookup", "max", "min", "compare", "compare_then_compute"]
    while len(out) < n:
        cats = list(rng.choice(_CATSETS))
        vals = _distinct_ints(rng, len(cats))
        ylabel = rng.choice(_YLABELS)
        mark = rng.choice(_MARKS)
        spec = _spec(cats, vals, ylabel, mark)
        kind = rng.choice(kinds)
        if kind == "value_lookup":
            i = rng.randrange(len(cats))
            q, ans, at = f"What is the {ylabel} of {cats[i]}?", vals[i], "numeric"
        elif kind == "max":
            i = vals.index(max(vals))
            q, ans, at = f"Which category has the highest {ylabel}?", cats[i], "category"
        elif kind == "min":
            i = vals.index(min(vals))
            q, ans, at = f"Which category has the lowest {ylabel}?", cats[i], "category"
        elif kind == "compare":
            i, j = rng.sample(range(len(cats)), 2)
            q = f"Is the {ylabel} of {cats[i]} greater than {cats[j]}?"
            ans, at = bool(vals[i] > vals[j]), "boolean"
        else:  # compare_then_compute: max - min
            q = f"By how much does the highest {ylabel} exceed the lowest?"
            ans, at = max(vals) - min(vals), "numeric"
        out.append({
            "spec": json.dumps(spec, ensure_ascii=False),
            "question": q,
            "answer": ans,
            "qa_kind": kind,
            "answer_type": at,
            "meta": {"source": "vega_spec", "mark": mark, "qa_kind": kind},
        })
    return out
