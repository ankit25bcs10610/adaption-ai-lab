"""Self-contained HTML evaluation report — dependency-free (no matplotlib/plotly).

Judges skim submissions in minutes, so make the results instantly graspable. This renders one offline
HTML file with:
  * a base-vs-fine-tuned metric table with colored deltas,
  * per-category accuracy bars (from results/eval_bfcl.json, if present),
  * a call/refuse/clarify CONFUSION MATRIX (from results/predictions.jsonl, if present) — the sharpest,
    most tool-calling-specific visual; off-diagonal cells expose the interesting failures.

Everything is inline HTML+CSS so the file opens anywhere with no assets.

Usage:
  python -m autoscientist_toolcaller.eval_report --baseline results/baseline.json --finetuned results/eval.json \
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
    """Banner from autoscientist_toolcaller.eval_stats output: paired gap + 95% CI + bootstrap/McNemar p-values."""
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


def render_schema_drift(ft: Dict[str, Any]) -> str:
    """Per-sd_kind accuracy table from eval metrics (schema-awareness is a distinctive claim)."""
    by_sd = (ft or {}).get("by_sd_kind")
    if not by_sd:
        return ""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v['accuracy']*100:.1f}%</td><td>±{v['stderr']*100:.1f}</td><td>{v['n']}</td></tr>"
        for k, v in sorted(by_sd.items())
    )
    overall = ft.get("schema_drift_accuracy")
    cap = f" — overall {overall*100:.1f}%" if isinstance(overall, (int, float)) else ""
    return ("<h2>Schema-drift robustness (per drift kind)" + cap + "</h2>"
            "<table><thead><tr><th>Drift kind</th><th>Accuracy</th><th>SE</th><th>n</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>")


def render_error_explorer(errors: List[Dict[str, Any]], limit: int = 20) -> str:
    """Worst/representative failures with tool context (from autoscientist_toolcaller.error_analysis's errors.jsonl).

    Each row: category · failure reason · the user query · the gold envelope · the model's output. Lets a
    judge (or you) see *what* the model gets wrong and why — the data-centric iteration loop, legible."""
    if not errors:
        return ""
    def esc(s: str) -> str:
        return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    # a few per category so the table is representative, not dominated by one bucket
    per_cat: Dict[str, int] = {}
    picked = []
    for e in errors:
        c = e.get("category", "other")
        if per_cat.get(c, 0) < 4:
            per_cat[c] = per_cat.get(c, 0) + 1
            picked.append(e)
        if len(picked) >= limit:
            break
    head = "".join(f"<th>{h}</th>" for h in ["Category", "Reason", "Query", "Gold", "Model output"])
    body = ""
    for e in picked:
        gold = e.get("gold")
        gold_s = json.dumps(gold, ensure_ascii=False) if not isinstance(gold, str) else gold
        body += (
            f"<tr><td>{esc(e.get('category',''))}</td>"
            f"<td><code>{esc(e.get('reason',''))}</code></td>"
            f"<td>{esc(str(e.get('query',''))[:120])}</td>"
            f"<td><code>{esc(gold_s[:120])}</code></td>"
            f"<td><code>{esc(str(e.get('output',''))[:120])}</code></td></tr>"
        )
    return ("<h2>Error explorer (representative failures)</h2>"
            f'<p class="muted">{len(errors)} failures; showing up to {limit} across categories.</p>'
            f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")


def render_decomposition(decomp: Dict[str, Any]) -> str:
    """Gap decomposition + multi-seed banner from autoscientist_toolcaller.eval_decompose output (results/eval_decompose.json).

    Shows WHERE the headline gain comes from (contributions that sum to the total) and, if present, the
    across-seed mean ± std — the two artifacts that turn "+X pp" from lucky-looking into believable.
    """
    if not decomp:
        return ""
    from .eval_decompose import to_html as _dec_html, seeds_to_markdown as _seeds_md

    out = []
    ms = decomp.get("multiseed")
    if ms:
        out.append(f'<div class="sig" style="border-color:#0ea5e955;background:#0ea5e912">'
                   f'<span>{_seeds_md(ms).replace("**", "")}</span></div>')
    dec = decomp.get("decomposition")
    if dec and dec.get("by_condition"):
        ok = "✓ contributions sum to the total" if dec.get("identity_ok") else "✗ identity broken"
        out.append("<h2>Where the gain comes from (decomposition)</h2>"
                   f'<p class="muted">The overall gap split by condition — {ok}.</p>'
                   + _dec_html(dec))
    return "\n".join(out)


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


def render_agentic(ag: Dict[str, Any]) -> str:
    """Agentic trajectory eval (results/eval_agentic.json): free-rollout success + per-step + by-env."""
    if not ag or not ag.get("n"):
        return ""
    by = ag.get("by_env") or {}
    rows = "".join(
        f"<tr><td>{e}</td><td>{v['success_rate']*100:.1f}%</td><td>{v['n']}</td></tr>"
        for e, v in sorted(by.items())
    )
    tbl = (f'<table><thead><tr><th>Env</th><th>Success</th><th>n</th></tr></thead>'
           f'<tbody>{rows}</tbody></table>') if rows else ""
    return ("<h2>Agentic trajectory eval</h2>"
            '<p class="muted">Free rollout — the environment is advanced by the model&rsquo;s own actions '
            "(observation-in-the-loop). Recovery trajectories must abstain on the impossible step.</p>"
            f'<p><span class="badge">{ag["trajectory_success_rate"]*100:.1f}% trajectory success</span> '
            f'· per-step {ag["per_step_accuracy"]*100:.1f}% · avg {ag.get("avg_steps",0):.1f} steps · n={ag["n"]}</p>'
            + tbl)


def render_calibration(ft: Dict[str, Any]) -> str:
    """Abstention calibration (from the fine-tuned eval's calibration block)."""
    cal = (ft or {}).get("calibration")
    if not cal:
        return ""
    return ("<h2>Calibration &amp; abstention</h2>"
            '<p class="muted">Does the model abstain when it should (refuse/clarify) WITHOUT over-refusing '
            "satisfiable requests?</p>"
            f'<p><span class="badge">over-refusal {cal["over_refusal_rate"]*100:.1f}%</span> '
            f'· abstention precision {cal["abstention_precision"]*100:.1f}% '
            f'· recall {cal["abstention_recall"]*100:.1f}%</p>')


def render_multilingual(ml: Dict[str, Any]) -> str:
    """Matched-pair Δaccuracy(lang−en) (results/eval_multilingual.json)."""
    if not ml or not ml.get("languages"):
        return ""
    by, d = ml["by_lang"], ml.get("matched_pair_delta_vs_en", {})
    rows = []
    for lang in ml["languages"]:
        acc = by[lang]["accuracy"]
        acc_s = f"{acc*100:.1f}%" if isinstance(acc, (int, float)) else "n/a"
        dv = "—" if lang == "en" else (f'{d[lang]["delta_vs_en"]*100:+.1f} pp' if lang in d else "n/a")
        rows.append(f"<tr><td>{lang}</td><td>{acc_s}</td><td>{dv}</td></tr>")
    return ("<h2>Multilingual robustness (matched-pair &Delta;)</h2>"
            "<table><thead><tr><th>Lang</th><th>Accuracy</th><th>&Delta; vs en</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


def render_html(
    base: Dict[str, Any],
    ft: Dict[str, Any],
    bfcl: Dict[str, Any],
    preds: List[Dict[str, Any]],
    stats: Optional[Dict[str, Any]] = None,
    decomp: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    agentic: Optional[Dict[str, Any]] = None,
    multilingual: Optional[Dict[str, Any]] = None,
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
{render_decomposition(decomp or {})}
<h2>Base vs. fine-tuned</h2>
<p class="muted">Identical greedy decoding for both. Lower is better for hallucination.</p>
{render_metric_table(base, ft)}
{('<p><span class="badge">BFCL-weighted ' + f"{bfcl['weighted_accuracy']*100:.1f}%" + '</span> (proxy, category-weighted)</p>') if isinstance((bfcl or {}).get('weighted_accuracy'), (int, float)) else ''}
{render_category_bars(bfcl)}
{render_radar(bfcl)}
{render_agentic(agentic or {})}
{render_calibration(ft)}
{render_multilingual(multilingual or {})}
{render_schema_drift(ft)}
{render_confusion(preds)}
{render_error_explorer(errors or [])}
<footer>Generated by autoscientist_toolcaller/eval_report.py — reproduce with the repo pipeline.</footer>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="results/baseline.json")
    ap.add_argument("--finetuned", default="results/eval.json")
    ap.add_argument("--bfcl", default="results/eval_bfcl.json")
    ap.add_argument("--predictions", default="results/predictions.jsonl")
    ap.add_argument("--stats", default="results/eval_stats.json")
    ap.add_argument("--decomp", default="results/eval_decompose.json")
    ap.add_argument("--errors", default="results/errors.jsonl")
    ap.add_argument("--agentic", default="results/eval_agentic.json")
    ap.add_argument("--multilingual", default="results/eval_multilingual.json")
    ap.add_argument("--out", default="results/report.html")
    args = ap.parse_args()

    html = render_html(
        _load(args.baseline), _load(args.finetuned), _load(args.bfcl),
        _load_jsonl(args.predictions), _load(args.stats), _load(args.decomp),
        _load_jsonl(args.errors), _load(args.agentic), _load(args.multilingual),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(html)
    print(f"[report] wrote {args.out} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
