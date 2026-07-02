"""Headline-gap DECOMPOSITION + multi-seed aggregation — make "+X pp" believable, not lucky.

A single overall improvement number invites the reviewer's real question: *where does it come from, and
is it just noise/leakage?* This module answers both, offline:

1. `decompose(rows)` attributes the overall base→fine-tuned accuracy gap to its mechanisms via the
   exact identity

       gap = ft_acc − base_acc = Σ_c (n_c / N) · (ft_acc_c − base_acc_c)

   so the headline splits cleanly into per-condition contributions (clean / multi-turn / irrelevance /
   clarify / …) that sum back to the total. Each condition also gets a paired bootstrap CI, so you can
   say "the gain is concentrated in the refuse/clarify moat, and it's significant there."

2. `aggregate_seeds([...])` averages the overall accuracy and gap across several eval runs (seeds
   41/42/43 …), reporting mean ± std — the single strongest tell against an overfit/lucky claim.

Both consume the per-example `predictions.jsonl` files written by `src.error_analysis`
(rows: {category, correct, ...}), aligned by position (same test set, same order) between baseline and
fine-tuned. Everything is pure stdlib and deterministic, so it unit-tests offline with mock predictions
and drops in real model outputs later unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from .eval_stats import paired_gap

# Canonical display order; unknown conditions are appended after these.
ORDER = ["simple", "multiple", "parallel", "parallel_multiple", "multi_turn", "clarify", "irrelevance"]


def _load(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_aligned(base_path: str, ft_path: str) -> List[Dict[str, Any]]:
    """Zip baseline + fine-tuned predictions into aligned rows {category, base, ft}."""
    base, ft = _load(base_path), _load(ft_path)
    n = min(len(base), len(ft))
    rows = []
    for i in range(n):
        rows.append({
            "category": base[i].get("category", ft[i].get("category", "other")),
            "base": int(bool(base[i].get("correct"))),
            "ft": int(bool(ft[i].get("correct"))),
        })
    return rows


def _order_key(cond: str) -> Tuple[int, str]:
    return (ORDER.index(cond), "") if cond in ORDER else (len(ORDER), cond)


def decompose(rows: List[Dict[str, Any]], n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Attribute the overall gap to per-condition contributions (summing to the total)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "overall": {"base_acc": 0.0, "ft_acc": 0.0, "gap": 0.0},
                "by_condition": [], "identity_ok": True}
    base_acc = sum(r["base"] for r in rows) / n
    ft_acc = sum(r["ft"] for r in rows) / n
    gap = ft_acc - base_acc

    conds = sorted({r["category"] for r in rows}, key=_order_key)
    by_condition: List[Dict[str, Any]] = []
    for c in conds:
        sub = [r for r in rows if r["category"] == c]
        n_c = len(sub)
        weight = n_c / n
        b_bits = [r["base"] for r in sub]
        f_bits = [r["ft"] for r in sub]
        base_acc_c = sum(b_bits) / n_c
        ft_acc_c = sum(f_bits) / n_c
        gap_c = ft_acc_c - base_acc_c
        contribution = weight * gap_c
        pg = paired_gap(b_bits, f_bits, n_boot=n_boot, seed=seed)
        by_condition.append({
            "condition": c,
            "n": n_c,
            "weight": round(weight, 4),
            "base_acc": base_acc_c,
            "ft_acc": ft_acc_c,
            "gap": gap_c,
            "contribution": contribution,                       # pp toward the overall gap
            "pct_of_total_gain": (contribution / gap) if abs(gap) > 1e-12 else None,
            "gap_ci": [pg["lo"], pg["hi"]],
            "gap_p_value": pg["p_value"],
        })
    total_contrib = sum(r["contribution"] for r in by_condition)
    return {
        "n": n,
        "overall": {"base_acc": base_acc, "ft_acc": ft_acc, "gap": gap},
        "by_condition": by_condition,
        "sum_contributions": total_contrib,
        "identity_ok": abs(total_contrib - gap) < 1e-9,   # the decomposition must sum to the whole
    }


def aggregate_seeds(pairs: List[Tuple[str, str]], n_boot: int = 1000) -> Dict[str, Any]:
    """Aggregate overall accuracy + gap across several (base, ft) prediction-file pairs (one per seed).

    Reports mean ± population std of base_acc / ft_acc / gap. A tight std across seeds is the headline
    stability signal. Also returns each seed's decomposition for the per-seed table.
    """
    per_seed = []
    for base_path, ft_path in pairs:
        rows = load_aligned(base_path, ft_path)
        d = decompose(rows, n_boot=n_boot)
        per_seed.append(d)

    def _stats(vals: List[float]) -> Dict[str, float]:
        k = len(vals)
        if k == 0:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        m = sum(vals) / k
        var = sum((v - m) ** 2 for v in vals) / k
        return {"mean": m, "std": var ** 0.5, "n": k}

    return {
        "seeds": len(per_seed),
        "base_acc": _stats([d["overall"]["base_acc"] for d in per_seed]),
        "ft_acc": _stats([d["overall"]["ft_acc"] for d in per_seed]),
        "gap": _stats([d["overall"]["gap"] for d in per_seed]),
        "per_seed": per_seed,
    }


def _pp(x: float) -> str:
    sign = "+" if x >= 0 else "−"
    return f"{sign}{abs(x) * 100:.1f}pp"


def to_markdown(dec: Dict[str, Any]) -> str:
    o = dec["overall"]
    lines = [
        f"**Overall:** baseline {o['base_acc']*100:.1f}% → fine-tuned {o['ft_acc']*100:.1f}% "
        f"= **{_pp(o['gap'])}**, decomposed by condition (contributions sum to the total):",
        "",
        "| Condition | n | weight | Base | Fine-tuned | Gap | Contribution | % of gain | Sig (p) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in dec["by_condition"]:
        pct = f"{r['pct_of_total_gain']*100:.0f}%" if r["pct_of_total_gain"] is not None else "—"
        lines.append(
            f"| {r['condition']} | {r['n']} | {r['weight']:.2f} | {r['base_acc']*100:.1f}% | "
            f"{r['ft_acc']*100:.1f}% | {_pp(r['gap'])} | {_pp(r['contribution'])} | {pct} | "
            f"{r['gap_p_value']:.3f} |"
        )
    lines.append("")
    lines.append(
        f"Σ contributions = {_pp(dec['sum_contributions'])} "
        f"({'✓ matches' if dec['identity_ok'] else '✗ DOES NOT match'} the overall gap)."
    )
    return "\n".join(lines)


def to_html(dec: Dict[str, Any]) -> str:
    o = dec["overall"]
    head = "".join(f"<th>{h}</th>" for h in
                   ["Condition", "n", "wt", "Base", "FT", "Gap", "Contribution", "% gain", "p"])
    body = ""
    for r in dec["by_condition"]:
        pct = f"{r['pct_of_total_gain']*100:.0f}%" if r["pct_of_total_gain"] is not None else "—"
        pos = r["contribution"] >= 0
        color = "#16a34a" if pos else "#dc2626"
        sig = "font-weight:700" if r["gap_p_value"] < 0.05 else "color:#94a3b8"
        body += (
            f"<tr><td>{r['condition']}</td><td>{r['n']}</td><td>{r['weight']:.2f}</td>"
            f"<td>{r['base_acc']*100:.1f}%</td><td>{r['ft_acc']*100:.1f}%</td>"
            f"<td>{_pp(r['gap'])}</td><td style='color:{color};font-weight:600'>{_pp(r['contribution'])}</td>"
            f"<td>{pct}</td><td style='{sig}'>{r['gap_p_value']:.3f}</td></tr>"
        )
    banner = (f"<p><strong>Overall {_pp(o['gap'])}</strong> "
              f"(baseline {o['base_acc']*100:.1f}% → fine-tuned {o['ft_acc']*100:.1f}%), "
              f"decomposed into contributions that sum to the total.</p>")
    return f"{banner}<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def seeds_to_markdown(agg: Dict[str, Any]) -> str:
    g, b, f = agg["gap"], agg["base_acc"], agg["ft_acc"]
    return (
        f"**Multi-seed ({agg['seeds']} seeds):** baseline {b['mean']*100:.1f}% ± {b['std']*100:.1f} · "
        f"fine-tuned {f['mean']*100:.1f}% ± {f['std']*100:.1f} · "
        f"gap **{_pp(g['mean'])} ± {g['std']*100:.1f}pp** (mean ± std across seeds)."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="baseline predictions.jsonl")
    ap.add_argument("--finetuned", help="fine-tuned predictions.jsonl")
    ap.add_argument("--pair", action="append", default=[],
                    help="base,ft prediction-file pair for a seed (repeatable for multi-seed aggregation)")
    ap.add_argument("--out", default="results/eval_decompose.json")
    args = ap.parse_args()

    out: Dict[str, Any] = {}
    if args.base and args.finetuned:
        dec = decompose(load_aligned(args.base, args.finetuned))
        out["decomposition"] = dec
        print(to_markdown(dec))
    if args.pair:
        pairs = []
        for spec in args.pair:
            b, f = spec.split(",", 1)
            pairs.append((b.strip(), f.strip()))
        agg = aggregate_seeds(pairs)
        out["multiseed"] = agg
        print("\n" + seeds_to_markdown(agg))
    if not out:
        raise SystemExit("provide --base/--finetuned and/or one or more --pair base,ft")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[decompose] wrote {args.out}")


if __name__ == "__main__":
    main()
