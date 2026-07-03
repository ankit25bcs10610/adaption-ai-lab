"""Robustness-delta table — the credibility centerpiece (research upgrade #1).

A large win on a hidden baseline reads as suspicious *unless* you explain it. The explanation: baselines
break under distribution shift (BFCL's own authors report 11–19% accuracy drops from mere paraphrasing).
This artifact shows, per condition (clean → multi-turn / irrelevance / clarify / parallel / …), that the
fine-tuned model's accuracy DROP is smaller than the baseline's — i.e. it degrades gracefully.

Consumes two per-example predictions files (baseline + fine-tuned) produced by `autoscientist_toolcaller.error_analysis`
(each row carries `category` + `correct`). "simple" is the clean reference; drop = simple_acc − cond_acc.
Renders a markdown table + a self-contained HTML block, and prints the headline robustness comparison.

Pure stdlib — offline, testable.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

CLEAN = "simple"
ORDER = ["simple", "multiple", "parallel", "multi_turn", "clarify", "irrelevance"]


def _load(path: str) -> List[Dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def per_condition(preds: List[Dict]) -> Dict[str, Dict[str, float]]:
    acc: Dict[str, Dict[str, int]] = {}
    for p in preds:
        c = p.get("category", "other")
        b = acc.setdefault(c, {"n": 0, "ok": 0})
        b["n"] += 1
        b["ok"] += int(bool(p.get("correct")))
    return {c: {"n": v["n"], "acc": v["ok"] / v["n"] if v["n"] else 0.0} for c, v in acc.items()}


def robustness(base_preds: List[Dict], ft_preds: List[Dict]) -> Dict:
    b, f = per_condition(base_preds), per_condition(ft_preds)
    clean_b = b.get(CLEAN, {}).get("acc")
    clean_f = f.get(CLEAN, {}).get("acc")
    conds = [c for c in ORDER if c in b or c in f] + [c for c in (set(b) | set(f)) if c not in ORDER]
    rows = []
    for c in conds:
        ba, fa = b.get(c, {}).get("acc"), f.get(c, {}).get("acc")
        rows.append({
            "condition": c,
            "base_acc": ba,
            "ft_acc": fa,
            "base_drop": (clean_b - ba) if (clean_b is not None and ba is not None and c != CLEAN) else None,
            "ft_drop": (clean_f - fa) if (clean_f is not None and fa is not None and c != CLEAN) else None,
            "n": (f.get(c) or b.get(c) or {}).get("n", 0),
        })
    # mean drop under shift (exclude clean)
    bd = [r["base_drop"] for r in rows if r["base_drop"] is not None]
    fd = [r["ft_drop"] for r in rows if r["ft_drop"] is not None]
    return {
        "rows": rows,
        "mean_base_drop": sum(bd) / len(bd) if bd else None,
        "mean_ft_drop": sum(fd) / len(fd) if fd else None,
    }


def _pct(x) -> str:
    return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"


def _dpp(x) -> str:
    return f"−{x*100:.1f}pp" if isinstance(x, (int, float)) else "—"


def to_markdown(r: Dict) -> str:
    lines = ["| Condition | Base acc | Fine-tuned acc | Base drop | Fine-tuned drop | n |",
             "|---|---|---|---|---|---|"]
    for row in r["rows"]:
        lines.append(
            f"| {row['condition']} | {_pct(row['base_acc'])} | {_pct(row['ft_acc'])} | "
            f"{_dpp(row['base_drop'])} | {_dpp(row['ft_drop'])} | {row['n']} |"
        )
    mb, mf = r["mean_base_drop"], r["mean_ft_drop"]
    if mb is not None and mf is not None:
        lines.append("")
        lines.append(f"**Mean drop under shift — baseline {_dpp(mb)} vs fine-tuned {_dpp(mf)}** "
                     f"({'more' if mf < mb else 'less'} robust).")
    return "\n".join(lines)


def to_html(r: Dict) -> str:
    head = "".join(f"<th>{h}</th>" for h in ["Condition", "Base", "Fine-tuned", "Base drop", "FT drop", "n"])
    body = ""
    for row in r["rows"]:
        better = (row["ft_drop"] is not None and row["base_drop"] is not None and row["ft_drop"] < row["base_drop"])
        color = "#16a34a" if better else "#64748b"
        body += (
            f"<tr><td>{row['condition']}</td><td>{_pct(row['base_acc'])}</td><td>{_pct(row['ft_acc'])}</td>"
            f"<td>{_dpp(row['base_drop'])}</td><td style='color:{color};font-weight:600'>{_dpp(row['ft_drop'])}</td>"
            f"<td>{row['n']}</td></tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline predictions.jsonl (category + correct)")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--out", default="results/robustness.md")
    args = ap.parse_args()
    r = robustness(_load(args.base), _load(args.finetuned))
    md = to_markdown(r)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(md + "\n")
    print(md)
    print(f"\n[robustness] wrote {args.out}")


if __name__ == "__main__":
    main()
