"""Self-contained HTML evaluation report — dependency-free (no matplotlib/plotly).

Judges skim submissions in minutes, so make the results instantly graspable. This renders one offline
HTML file with:
  * a base-vs-fine-tuned metric table with colored deltas,
  * per-category accuracy bars (from results/eval_bfcl.json, if present),
  * a call/refuse/clarify CONFUSION MATRIX (from results/predictions.jsonl, if present) — the sharpest,
    most tool-calling-specific visual; off-diagonal cells expose the interesting failures.

Everything is inline HTML+CSS so the file opens anywhere with no assets.

Usage:
  python -m src.eval_report --baseline results/baseline.json --finetuned results/eval.json \
    --bfcl results/eval_bfcl.json --predictions results/predictions.jsonl --out results/report.html
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

ACTIONS = ["call", "refuse", "clarify", "parse_fail"]
GOLD = ["call", "refuse", "clarify"]


def _load(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def _load_jsonl(path: Optional[str]) -> List[Dict[str, Any]]:
    if path and os.path.exists(path):
        return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return []


def _pct(x: Any) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "n/a"


def _delta_cell(base: Any, ft: Any, lower_is_better: bool = False) -> str:
    if not isinstance(base, (int, float)) or not isinstance(ft, (int, float)):
        return "<td>n/a</td>"
    d = ft - base
    good = (d < 0) if lower_is_better else (d > 0)
    color = "#16a34a" if good else ("#dc2626" if d != 0 else "#64748b")
    sign = "+" if d > 0 else ""
    return f'<td style="color:{color};font-weight:600">{sign}{d * 100:.1f} pp</td>'


METRIC_ROWS = [
    ("Overall accuracy", "overall_accuracy", False),
    ("Positive (tool-call) accuracy", "positive_accuracy", False),
    ("Refusal accuracy", "refusal_accuracy", False),
    ("Clarify accuracy", "clarify_accuracy", False),
    ("Hallucination rate", "hallucination_rate", True),
]


def render_metric_table(base: Dict[str, Any], ft: Dict[str, Any]) -> str:
    rows = []
    for label, key, lower in METRIC_ROWS:
        arrow = " &darr;" if lower else ""
        rows.append(
            f"<tr><td>{label}{arrow}</td><td>{_pct(base.get(key))}</td>"
            f"<td>{_pct(ft.get(key))}</td>{_delta_cell(base.get(key), ft.get(key), lower)}</tr>"
        )
    return (
        '<table><thead><tr><th>Metric</th><th>Base</th><th>Fine-tuned</th><th>&Delta;</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def render_category_bars(bfcl: Dict[str, Any]) -> str:
    by_cat = (bfcl or {}).get("by_category") or {}
    if not by_cat:
        return ""
    bars = []
    for cat, d in by_cat.items():
        acc = d.get("accuracy")
        if acc is None:
            continue
        w = acc * 100
        bars.append(
            f'<div class="barrow"><span class="barlabel">{cat} '
            f'<em>(n={d.get("n", 0)})</em></span>'
            f'<span class="bartrack"><span class="barfill" style="width:{w:.1f}%"></span></span>'
            f'<span class="barval">{w:.1f}%</span></div>'
        )
    if not bars:
        return ""
    return f'<h2>Per-category accuracy (fine-tuned)</h2><div class="bars">{"".join(bars)}</div>'


def confusion_from_predictions(preds: List[Dict[str, Any]]) -> Dict[Tuple[str, str], int]:
    m: Dict[Tuple[str, str], int] = {}
    for p in preds:
        g, pr = p.get("gold"), p.get("pred")
        if g in GOLD and pr in ACTIONS:
            m[(g, pr)] = m.get((g, pr), 0) + 1
    return m


def render_confusion(preds: List[Dict[str, Any]]) -> str:
    if not preds:
        return ""
    m = confusion_from_predictions(preds)
    row_totals = {g: sum(m.get((g, a), 0) for a in ACTIONS) for g in GOLD}
    cells = []
    header = "".join(f"<th>{a}</th>" for a in ACTIONS)
    for g in GOLD:
        tds = []
        for a in ACTIONS:
            n = m.get((g, a), 0)
            frac = n / row_totals[g] if row_totals[g] else 0.0
            on_diag = g == a
            # green intensity on the diagonal, red intensity off it
            base = "22,163,74" if on_diag else "220,38,38"
            bg = f"rgba({base},{0.12 + 0.6 * frac:.2f})" if n else "transparent"
            tds.append(f'<td style="background:{bg}">{n}<br><small>{frac*100:.0f}%</small></td>')
        cells.append(f"<tr><th>{g}</th>{''.join(tds)}</tr>")
    return (
        "<h2>Decision confusion matrix</h2>"
        '<p class="muted">Rows = correct action, columns = model&rsquo;s action. '
        "Off-diagonal cells are the interesting failures (e.g. should-have-clarified but called).</p>"
        f'<table class="confusion"><thead><tr><th>gold \\ pred</th>{header}</tr></thead>'
        f'<tbody>{"".join(cells)}</tbody></table>'
    )


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:900px;margin:40px auto;padding:0 20px;color:#0f172a}
h1{font-size:28px} h2{font-size:20px;margin-top:36px}
table{border-collapse:collapse;width:100%;margin-top:12px}
th,td{border:1px solid #e2e8f0;padding:8px 12px;text-align:left;font-size:14px}
thead th{background:#f1f5f9}
.confusion td{text-align:center}
.muted{color:#64748b;font-size:13px}
.bars{margin-top:12px}
.barrow{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px}
.barlabel{width:230px} .barlabel em{color:#94a3b8;font-style:normal}
.bartrack{flex:1;height:14px;background:#f1f5f9;border-radius:7px;overflow:hidden}
.barfill{display:block;height:100%;background:linear-gradient(90deg,#22c55e,#22d3ee)}
.barval{width:56px;text-align:right;font-variant-numeric:tabular-nums}
.badge{display:inline-block;background:#0f172a;color:#fff;border-radius:6px;padding:2px 8px;font-size:12px}
.sig{display:flex;align-items:center;gap:12px;border:1px solid;border-radius:12px;padding:12px 16px;margin-top:14px;font-size:14px}
.sigbadge{color:#fff;border-radius:8px;padding:4px 10px;font-weight:700;white-space:nowrap}
svg{max-width:100%}
footer{margin-top:40px;color:#94a3b8;font-size:12px}
"""


def render_significance(stats: Dict[str, Any]) -> str:
    """Banner from src.eval_stats output: paired gap + 95% CI + bootstrap/McNemar p-values."""
    g = (stats or {}).get("paired_gap")
    if not g:
        return ""
    p = g.get("p_value", 1.0)
    sig = p < 0.05
    color = "#16a34a" if sig else "#d97706"
    label = "statistically significant" if sig else "not yet significant"
    mc = (stats or {}).get("mcnemar", {}).get("p_value")
    mc_txt = f" · McNemar p={mc:.3f}" if isinstance(mc, (int, float)) else ""
    return (
        f'<div class="sig" style="border-color:{color}55;background:{color}12">'
        f'<span class="sigbadge" style="background:{color}">+{g["gap"]*100:.1f} pp</span>'
        f'<span>fine-tuned − baseline, 95% CI [{g["lo"]*100:.1f}, {g["hi"]*100:.1f}] pp · '
        f'bootstrap p={p:.3f}{mc_txt} — <b style="color:{color}">{label}</b> (n={g.get("n","?")})</span>'
        f"</div>"
    )


def render_radar(bfcl: Dict[str, Any], size: int = 300) -> str:
    """Inline-SVG radar of per-category accuracy (dependency-free)."""
    import math

    by = (bfcl or {}).get("by_category") or {}
    cats = [(c, d["accuracy"]) for c, d in by.items() if d.get("accuracy") is not None]
    if len(cats) < 3:
        return ""
    cx = cy = size / 2
    R = size / 2 - 46
    n = len(cats)
    pts, axes, labels = [], [], []
    for i, (c, a) in enumerate(cats):
        ang = -math.pi / 2 + i * 2 * math.pi / n
        ex, ey = cx + R * math.cos(ang), cy + R * math.sin(ang)
        px, py = cx + a * R * math.cos(ang), cy + a * R * math.sin(ang)
        pts.append(f"{px:.1f},{py:.1f}")
        axes.append(f'<line x1="{cx}" y1="{cy}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#e2e8f0"/>')
        anchor = "middle" if abs(math.cos(ang)) < 0.4 else ("start" if math.cos(ang) > 0 else "end")
        lx, ly = cx + (R + 14) * math.cos(ang), cy + (R + 14) * math.sin(ang)
        labels.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" font-size="11" fill="#475569">{c} {a*100:.0f}%</text>')
    rings = "".join(
        f'<circle cx="{cx}" cy="{cy}" r="{R*k:.1f}" fill="none" stroke="#f1f5f9"/>' for k in (0.33, 0.66, 1.0)
    )
    return (
        "<h2>Per-category accuracy (radar)</h2>"
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img" aria-label="per-category accuracy radar">'
        f"{rings}{''.join(axes)}"
        f'<polygon points="{" ".join(pts)}" fill="rgba(34,197,94,0.25)" stroke="#22c55e" stroke-width="2"/>'
        f"{''.join(labels)}</svg>"
    )


def render_html(
    base: Dict[str, Any],
    ft: Dict[str, Any],
    bfcl: Dict[str, Any],
    preds: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
) -> str:
    improvement = ""
    b, f = base.get("overall_accuracy"), ft.get("overall_accuracy")
    if isinstance(b, (int, float)) and isinstance(f, (int, float)):
        improvement = f'<p><span class="badge">+{(f - b) * 100:.1f} pp overall</span></p>'
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>AutoScientist Tool-Caller — Eval Report</title><style>{_CSS}</style></head><body>
<h1>AutoScientist Tool-Caller — Evaluation Report</h1>
{improvement}
{render_significance(stats or {})}
<h2>Base vs. fine-tuned</h2>
<p class="muted">Identical greedy decoding for both. Lower is better for hallucination.</p>
{render_metric_table(base, ft)}
{render_category_bars(bfcl)}
{render_radar(bfcl)}
{render_confusion(preds)}
<footer>Generated by src/eval_report.py — reproduce with the repo pipeline.</footer>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline.json")
    ap.add_argument("--finetuned", default="results/eval.json")
    ap.add_argument("--bfcl", default="results/eval_bfcl.json")
    ap.add_argument("--predictions", default="results/predictions.jsonl")
    ap.add_argument("--stats", default="results/eval_stats.json")
    ap.add_argument("--out", default="results/report.html")
    args = ap.parse_args()

    html = render_html(
        _load(args.baseline), _load(args.finetuned), _load(args.bfcl),
        _load_jsonl(args.predictions), _load(args.stats),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(html)
    print(f"[report] wrote {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
