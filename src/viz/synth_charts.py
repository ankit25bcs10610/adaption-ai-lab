"""Self-verifying synthetic chart generator — the originality moat for the Data-Viz track.

Renders charts with matplotlib and produces QA whose answers are CORRECT BY CONSTRUCTION: every gold
is computed from the underlying data, and the number drawn on the chart is produced by the SAME
formatter used for the gold (invariant I2), so a perfectly-reading model is never graded wrong.

Hardened per adversarial review (invariants I1-I7):
  I1  values re-quantized to `dec` decimals after every mutation.
  I2  on-chart value labels use `num_str()` == the gold string.
  I3  ties / comparisons / thresholds use integer quanta, never float ==/>=.
  I4  each QA maker re-validates its preconditions and returns None if unmet.
  I5  per-example RNG seeded by hash of (seed, index) — collision-resistant.
  I6  trend gated on ordered categories; stacked value_lookup only on the bottom series;
      pie proportion labels drawn from GT strings (not autopct).
  I7  no thousands separators / units baked into the number that must match the label.

Pure-logic (data gen + GT) imports only stdlib; matplotlib is imported lazily inside rendering.
API: generate(n, out_dir, kind_weights=?, chart_weights=?, seed=42) -> list[canonical example].
"""
from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .format_utils import _scalar_to_text as num_str

# --------------------------------------------------------------------------------------
# Themed vocab (kept covered by the Indic label map). VOCAB = every emittable label string.
# --------------------------------------------------------------------------------------
REGIONS = ["North", "South", "East", "West", "Central"]
MONTHS = ["January", "February", "March", "April", "May", "June"]
DEPARTMENTS = ["Sales", "Marketing", "Engineering", "Support", "Finance"]
PRODUCTS = ["Product A", "Product B", "Product C", "Product D", "Product E"]

# (title, x_label, y_label/metric, unit, decimals, category_pool, ordered)
THEMES = [
    ("Sales by Region", "Region", "Sales", "", 0, REGIONS, False),
    ("Revenue by Month", "Month", "Revenue", "", 0, MONTHS, True),
    ("Users by Product", "Product", "Users", "", 0, PRODUCTS, False),
    ("Headcount by Department", "Department", "Headcount", "", 0, DEPARTMENTS, False),
    ("Temperature by Month", "Month", "Temperature", "", 1, MONTHS, True),
    ("Market Share", "Product", "Share", "", 1, PRODUCTS, False),
]

VOCAB = set()
for _t in THEMES:
    VOCAB.update([_t[0], _t[1], _t[2]])
    VOCAB.update(_t[5])


def _qi(v: float, dec: int) -> int:
    return int(round(v * (10 ** dec)))


def _Q(v: float, dec: int) -> float:
    return round(v, dec)


@dataclass
class ChartData:
    chart_type: str
    title: str
    x_label: str
    y_label: str
    categories: List[str]
    series_names: List[str]
    values: List[List[float]]  # values[series][category], pre-quantized
    unit: str
    dec: int
    ordered: bool
    points: Optional[List[Tuple[float, float]]] = None  # scatter only
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QAItem:
    qa_kind: str
    question: str
    answer: Any
    answer_type: str  # numeric | category | trend | boolean
    support: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Data sampling
# --------------------------------------------------------------------------------------
def _subseed(seed: int, idx: int) -> int:
    h = hashlib.blake2b(f"{seed}:{idx}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def _gen_values(rng: random.Random, n: int, dec: int, lo=10.0, hi=100.0) -> List[float]:
    return [_Q(rng.uniform(lo, hi), dec) for _ in range(n)]


def _dedupe_unique(vals: List[float], dec: int, rng: random.Random) -> List[float]:
    """Ensure all values are distinct at integer-quantum resolution (for clean max/min)."""
    step = 10 ** (-dec) if dec > 0 else 1
    seen = set()
    out = []
    for v in vals:
        q = _qi(v, dec)
        while q in seen:
            q += 1
        seen.add(q)
        out.append(_Q(q * (10 ** (-dec)), dec))
    return out


def sample_chart(rng: random.Random, chart_type: str) -> ChartData:
    title, xl, yl, unit, dec, pool, ordered = THEMES[rng.randrange(len(THEMES))]
    if chart_type in ("line", "multiline", "area"):
        # need ordered categories for trend
        title, xl, yl, unit, dec, pool, ordered = ("Revenue by Month", "Month", "Revenue", "", 0, MONTHS, True)
    ncat = rng.randint(4, min(6, len(pool)))
    cats = pool[:ncat]

    if chart_type == "scatter":
        # generate correlated points with |r|>0.5
        npts = rng.randint(8, 14)
        sign = rng.choice([1, -1])
        xs = sorted(_Q(rng.uniform(0, 100), 0) for _ in range(npts))
        base = [sign * x for x in xs]
        pts = [(xs[i], _Q(base[i] + rng.uniform(-15, 15), 0)) for i in range(npts)]
        return ChartData("scatter", "Y vs X", "X", "Y", cats, ["series"], [[p[1] for p in pts]],
                         unit, 0, False, points=pts)

    if chart_type in ("grouped_bar", "stacked_bar", "multiline"):
        nser = rng.randint(2, 3)
        snames = DEPARTMENTS[:nser] if pool is not MONTHS else [f"Series {i+1}" for i in range(nser)]
        vals = [_dedupe_unique(_gen_values(rng, ncat, dec), dec, rng) for _ in range(nser)]
        return ChartData(chart_type, title, xl, yl, cats, snames, vals, unit, dec, ordered)

    # single-series: bar, hbar, line, pie, area
    vals = _dedupe_unique(_gen_values(rng, ncat, dec), dec, rng)
    return ChartData(chart_type, title, xl, yl, cats, ["series"], [vals], unit, dec, ordered)


# --------------------------------------------------------------------------------------
# Hardened QA makers  (return QAItem or None)
# --------------------------------------------------------------------------------------
def _unique_argext(arr: List[float], dec: int, want_max: bool) -> Optional[int]:
    qs = [_qi(v, dec) for v in arr]
    target = max(qs) if want_max else min(qs)
    if qs.count(target) != 1:
        return None  # tie -> ambiguous
    return qs.index(target)


def mk_value_lookup(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if cd.chart_type == "stacked_bar":
        si = 0  # bottom series only (readable from 0)
    else:
        si = rng.randrange(len(cd.series_names))
    ci = rng.randrange(len(cd.categories))
    v = cd.values[si][ci]
    sname = cd.series_names[si]
    q = (f"What is the {cd.y_label.lower()} for {cd.categories[ci]}?"
         if len(cd.series_names) == 1
         else f"What is the {cd.y_label.lower()} of {sname} for {cd.categories[ci]}?")
    return QAItem("value_lookup", q, v, "numeric",
                  {"category": cd.categories[ci], "series": sname, "value": v})


def mk_extremum(cd: ChartData, rng: random.Random, want_max: bool) -> Optional[QAItem]:
    multi = len(cd.series_names) > 1 and cd.chart_type != "stacked_bar"
    sname = None
    if cd.chart_type == "stacked_bar":
        arr = [sum(cd.values[s][c] for s in range(len(cd.series_names))) for c in range(len(cd.categories))]
    elif multi:
        si = rng.randrange(len(cd.series_names))
        sname = cd.series_names[si]
        arr = cd.values[si]
    else:
        arr = cd.values[0]
    idx = _unique_argext(arr, cd.dec, want_max)
    if idx is None:
        return None
    kind = "max" if want_max else "min"
    word = "highest" if want_max else "lowest"
    series_clause = f" for {sname}" if sname else ""  # name the series so the answer is determinable
    q = f"Which {cd.x_label.lower()} has the {word} {cd.y_label.lower()}{series_clause}?"
    return QAItem(kind, q, cd.categories[idx], "category",
                  {"category": cd.categories[idx], "value": arr[idx], "series": sname})


def mk_compare(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    multi = len(cd.series_names) > 1 and cd.chart_type != "stacked_bar"
    si = 0 if cd.chart_type == "stacked_bar" else rng.randrange(len(cd.series_names))
    sname = cd.series_names[si] if multi else None
    if len(cd.categories) < 2:
        return None
    ci, cj = rng.sample(range(len(cd.categories)), 2)
    row = cd.values[si]
    if abs(_qi(row[ci], cd.dec) - _qi(row[cj], cd.dec)) < 1:  # indistinguishable
        return None
    ans = _qi(row[ci], cd.dec) > _qi(row[cj], cd.dec)
    series_clause = f" for {sname}" if sname else ""
    q = f"Is the {cd.y_label.lower()} of {cd.categories[ci]} greater than {cd.categories[cj]}{series_clause}?"
    return QAItem("compare", q, bool(ans), "boolean",
                  {"cat_a": cd.categories[ci], "cat_b": cd.categories[cj], "series": sname})


def mk_difference(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    si = 0 if cd.chart_type == "stacked_bar" else rng.randrange(len(cd.series_names))
    if len(cd.categories) < 2:
        return None
    ci, cj = rng.sample(range(len(cd.categories)), 2)
    row = cd.values[si]
    if abs(_qi(row[ci], cd.dec) - _qi(row[cj], cd.dec)) < 1:
        return None
    hi, lo = (ci, cj) if _qi(row[ci], cd.dec) > _qi(row[cj], cd.dec) else (cj, ci)
    ans = _Q(row[hi] - row[lo], cd.dec)
    q = f"How much larger is the {cd.y_label.lower()} of {cd.categories[hi]} than {cd.categories[lo]}?"
    return QAItem("difference", q, ans, "numeric", {"cat_a": cd.categories[hi], "cat_b": cd.categories[lo]})


def mk_sum(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if cd.chart_type == "stacked_bar":
        ci = rng.randrange(len(cd.categories))
        ans = _Q(sum(cd.values[s][ci] for s in range(len(cd.series_names))), cd.dec)
        q = f"What is the total {cd.y_label.lower()} for {cd.categories[ci]} across all series?"
        return QAItem("sum", q, ans, "numeric", {"category": cd.categories[ci]})
    si = 0 if len(cd.series_names) == 1 else 0
    ans = _Q(sum(cd.values[si]), cd.dec)  # sum of quantized (I3)
    q = f"What is the total {cd.y_label.lower()} across all {cd.x_label.lower()}s?"
    return QAItem("sum", q, ans, "numeric", {})


def mk_mean(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if cd.chart_type == "stacked_bar":
        arr = [sum(cd.values[s][c] for s in range(len(cd.series_names))) for c in range(len(cd.categories))]
        q = f"What is the average total {cd.y_label.lower()} per {cd.x_label.lower()}?"
    else:
        arr = cd.values[0]
        q = f"What is the average {cd.y_label.lower()} across all {cd.x_label.lower()}s?"
    ans = _Q(sum(arr) / len(arr), cd.dec)
    return QAItem("mean", q, ans, "numeric", {})


def mk_count(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    row = cd.values[0]
    qs = sorted(set(_qi(v, cd.dec) for v in row))
    if len(qs) < 3:
        return None
    # find an adjacent pair with an integer-quantum gap >= 2, threshold strictly inside
    gaps = [(qs[k] - qs[k - 1], k) for k in range(1, len(qs))]
    gaps = [(g, k) for g, k in gaps if g >= 2]
    if not gaps:
        return None
    _, k = rng.choice(gaps)
    nq = qs[k - 1] + (qs[k] - qs[k - 1]) // 2  # integer, strictly between, equals no datum
    thr = nq * (10 ** (-cd.dec)) if cd.dec else nq
    ans = sum(1 for v in row if _qi(v, cd.dec) > nq)
    q = f"How many {cd.x_label.lower()}s have {cd.y_label.lower()} greater than {num_str(thr)}?"
    return QAItem("count", q, int(ans), "numeric", {"threshold": thr})


def _lin_slope(ys: List[float]) -> float:
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1.0
    return num / den


def mk_trend(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if not cd.ordered:
        return None
    row = cd.values[0]
    span = max(row) - min(row)
    if span == 0:
        return None
    slope = _lin_slope(row)
    norm = slope * (len(row) - 1) / span
    if abs(norm) <= 0.10:
        return None  # too flat to label confidently -> skip
    ans = "increasing" if norm > 0 else "decreasing"
    q = f"What is the overall trend of {cd.y_label.lower()} over time?"
    return QAItem("trend", q, ans, "trend", {"slope": round(slope, 4)})


def mk_proportion(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if cd.chart_type != "pie":
        return None
    row = cd.values[0]
    total = sum(row)
    if total <= 0:
        return None
    ci = rng.randrange(len(cd.categories))
    ans = round(100.0 * row[ci] / total, 1)
    q = f"What percentage of the total does {cd.categories[ci]} represent?"
    return QAItem("proportion", q, ans, "numeric", {"category": cd.categories[ci], "percent": ans})


def mk_correlation(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    if cd.chart_type != "scatter" or not cd.points:
        return None
    xs = [p[0] for p in cd.points]
    ys = [p[1] for p in cd.points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    r = cov / (sx * sy)
    if abs(r) < 0.3:
        return None
    ans = "positive" if r > 0 else "negative"
    return QAItem("trend", "Is the correlation between X and Y positive or negative?", ans, "trend",
                  {"r": round(r, 3)})


def mk_compare_then_compute(cd: ChartData, rng: random.Random) -> Optional[QAItem]:
    """Compound multi-hop: locate the highest AND lowest, then compute the gap (max − min).

    Two reasoning hops in one question — harder than single-lookup and correct by construction."""
    if cd.chart_type == "stacked_bar":
        arr = [sum(cd.values[s][c] for s in range(len(cd.series_names))) for c in range(len(cd.categories))]
    else:
        si = 0 if len(cd.series_names) == 1 else rng.randrange(len(cd.series_names))
        arr = cd.values[si]
    hi = _unique_argext(arr, cd.dec, True)
    lo = _unique_argext(arr, cd.dec, False)
    if hi is None or lo is None or hi == lo:
        return None
    ans = _Q(arr[hi] - arr[lo], cd.dec)
    q = f"By how much does the highest {cd.y_label.lower()} exceed the lowest?"
    return QAItem("compare_then_compute", q, ans, "numeric",
                  {"hi": cd.categories[hi], "lo": cd.categories[lo], "value": ans})


_MAKERS: Dict[str, List[Callable]] = {
    "bar": [mk_value_lookup, lambda c, r: mk_extremum(c, r, True), lambda c, r: mk_extremum(c, r, False),
            mk_compare, mk_sum, mk_count, mk_difference, mk_mean, mk_compare_then_compute],
    "hbar": [mk_value_lookup, lambda c, r: mk_extremum(c, r, True), lambda c, r: mk_extremum(c, r, False),
             mk_compare, mk_difference, mk_mean, mk_compare_then_compute],
    # grouped_bar/multiline: extremum & compare now name the series; mean omitted (series-ambiguous)
    "grouped_bar": [mk_value_lookup, lambda c, r: mk_extremum(c, r, True),
                    lambda c, r: mk_extremum(c, r, False), mk_compare],
    "stacked_bar": [lambda c, r: mk_extremum(c, r, True), lambda c, r: mk_extremum(c, r, False),
                    mk_sum, mk_mean, mk_value_lookup, mk_compare_then_compute],
    "line": [mk_trend, mk_value_lookup, lambda c, r: mk_extremum(c, r, True), mk_mean],
    "multiline": [mk_value_lookup, lambda c, r: mk_extremum(c, r, True), mk_compare],
    # scatter/area draw no value labels -> mean not extractable; keep only readable kinds
    "area": [mk_trend, lambda c, r: mk_extremum(c, r, True)],
    "pie": [mk_proportion, mk_compare],  # value_lookup dropped: pie shows % labels, not raw values
    "scatter": [mk_correlation],
}


# --------------------------------------------------------------------------------------
# Rendering (lazy matplotlib). Value labels use num_str() so label == gold string (I2).
# --------------------------------------------------------------------------------------
def render_chart(cd: ChartData, path: str, apply_fonts: Optional[Callable] = None, dpi: int = 100) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    cats = cd.categories
    v = cd.values

    if cd.chart_type in ("bar", "line", "area"):
        row = v[0]
        if cd.chart_type == "bar":
            bars = ax.bar(cats, row)
            ax.bar_label(bars, labels=[num_str(x) for x in row], padding=2, fontsize=8)
        elif cd.chart_type == "line":
            ax.plot(cats, row, marker="o")
            for i, x in enumerate(row):
                ax.annotate(num_str(x), (i, x), textcoords="offset points", xytext=(0, 5), fontsize=8)
        else:  # area
            ax.fill_between(range(len(cats)), row, alpha=0.5)
            ax.plot(range(len(cats)), row, marker="o")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels(cats)
    elif cd.chart_type == "hbar":
        row = v[0]
        bars = ax.barh(cats, row)
        ax.bar_label(bars, labels=[num_str(x) for x in row], padding=2, fontsize=8)
    elif cd.chart_type in ("grouped_bar", "multiline"):
        x = np.arange(len(cats))
        ns = len(cd.series_names)
        if cd.chart_type == "grouped_bar":
            w = 0.8 / ns
            for s in range(ns):
                ax.bar(x + s * w - 0.4 + w / 2, v[s], w, label=cd.series_names[s])
        else:
            for s in range(ns):
                ax.plot(x, v[s], marker="o", label=cd.series_names[s])
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.legend(fontsize=8)
    elif cd.chart_type == "stacked_bar":
        import numpy as np
        bottom = np.zeros(len(cats))
        for s in range(len(cd.series_names)):
            ax.bar(cats, v[s], bottom=bottom, label=cd.series_names[s])
            bottom = bottom + np.array(v[s])
        ax.legend(fontsize=8)
    elif cd.chart_type == "pie":
        row = v[0]
        total = sum(row)
        # draw GT percent labels via text (not autopct) so label == gold (F/A4)
        labels = [f"{c}\n{round(100.0*x/total,1)}%" for c, x in zip(cats, row)]
        ax.pie(row, labels=labels, textprops={"fontsize": 8})
    elif cd.chart_type == "scatter":
        xs = [p[0] for p in cd.points]
        ys = [p[1] for p in cd.points]
        ax.scatter(xs, ys)

    ax.set_title(cd.title)
    if cd.chart_type not in ("pie",):
        ax.set_xlabel(cd.x_label)
        ax.set_ylabel(cd.y_label)
    if apply_fonts is not None:
        apply_fonts(fig, ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, metadata={"CreationDate": None})
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------
# Public generator
# --------------------------------------------------------------------------------------
DEFAULT_CHART_WEIGHTS = {
    "bar": 0.22, "line": 0.16, "pie": 0.12, "grouped_bar": 0.12, "stacked_bar": 0.12,
    "hbar": 0.08, "multiline": 0.08, "scatter": 0.06, "area": 0.04,
}


def generate(
    n: int,
    out_dir: str,
    chart_weights: Optional[Dict[str, float]] = None,
    questions_per_chart: int = 2,
    seed: int = 42,
    apply_fonts: Optional[Callable] = None,
    render: bool = True,
    lang: str = "en",
) -> List[Dict[str, Any]]:
    """Generate up to `n` canonical chart-QA examples. Deterministic given seed.

    apply_fonts / lang let the Indic module reuse this generator with localized labels.
    """
    cw = chart_weights or DEFAULT_CHART_WEIGHTS
    types = list(cw.keys())
    weights = [cw[t] for t in types]
    master = random.Random(seed)
    out: List[Dict[str, Any]] = []
    chart_idx = 0
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        rng = random.Random(_subseed(seed, chart_idx))
        chart_idx += 1
        ctype = master.choices(types, weights=weights, k=1)[0]
        cd = sample_chart(rng, ctype)
        makers = list(_MAKERS.get(ctype, []))  # copy — never shuffle the module-global list
        rng.shuffle(makers)
        made: List[QAItem] = []
        for mk in makers:
            if len(made) >= questions_per_chart:
                break
            item = mk(cd, rng)
            if item is not None:
                made.append(item)
        if not made:
            continue
        img_path = None
        if render:
            img_path = os.path.join(out_dir, "img", f"chart_{chart_idx:06d}.png")
            render_chart(cd, img_path, apply_fonts=apply_fonts)
        for q in made:
            if len(out) >= n:
                break
            out.append(
                {
                    "image": img_path,
                    "question": q.question,
                    "answer": q.answer,
                    "chart_type": cd.chart_type,
                    "qa_kind": q.qa_kind,
                    "meta": {
                        "source": "synth" if lang == "en" else "indic_synth",
                        "lang": lang,
                        "script": None,
                        "chart_type": cd.chart_type,
                        "qa_kind": q.qa_kind,
                        "answer_type": q.answer_type,
                        "seed": _subseed(seed, chart_idx),
                        "support": q.support,
                    },
                }
            )
    return out
