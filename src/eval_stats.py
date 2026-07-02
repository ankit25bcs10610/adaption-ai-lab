"""Rigorous eval statistics — make the "measurable improvement" claim bulletproof.

The challenge's #1 gate is a measurable improvement over the baseline. A single accuracy number isn't
convincing to expert judges; a bootstrapped confidence interval plus a *paired* significance test is.

This module provides:
  * bootstrap_ci(bits)                    -> mean + 95% CI for one model's accuracy
  * paired_gap(base_bits, ft_bits)        -> the base→ft gap, its 95% CI, and a bootstrap p-value
                                             (fraction of resamples where fine-tuned does NOT beat base)
  * mcnemar(base_bits, ft_bits)           -> exact McNemar test on the discordant pairs

`*_bits` are per-example 0/1 correctness aligned across models (same test examples, same order). The CLI
reads two predictions JSONL files (from src.error_analysis, which writes {gold,pred,correct,...} per
example) and prints/saves the full statistical comparison.

Pure stdlib — runs offline, no heavy deps.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple


def bootstrap_ci(
    bits: List[int], alpha: float = 0.05, n_boot: int = 2000, seed: int = 42
) -> Dict[str, float]:
    """Percentile bootstrap CI for a proportion (accuracy)."""
    n = len(bits)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = sum(bits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return {"mean": sum(bits) / n, "lo": lo, "hi": hi, "n": n}


def paired_gap(
    base_bits: List[int], ft_bits: List[int], alpha: float = 0.05, n_boot: int = 2000, seed: int = 42
) -> Dict[str, float]:
    """Paired bootstrap on the (ft - base) accuracy gap. Resample example INDICES once per iteration and
    compute both models on the same resample (correct paired inference). Returns the gap, its CI, and a
    one-sided bootstrap p-value = P(gap <= 0)."""
    assert len(base_bits) == len(ft_bits), "base/ft bits must be aligned (same examples, same order)"
    n = len(base_bits)
    if n == 0:
        return {"gap": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0, "n": 0}
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(base_bits[i] for i in idx) / n
        f = sum(ft_bits[i] for i in idx) / n
        gaps.append(f - b)
    gaps.sort()
    lo = gaps[int((alpha / 2) * n_boot)]
    hi = gaps[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    obs = sum(ft_bits) / n - sum(base_bits) / n
    p = sum(1 for g in gaps if g <= 0) / n_boot  # bootstrap prob the improvement is <= 0
    return {"gap": obs, "lo": lo, "hi": hi, "p_value": p, "n": n}


def mcnemar(base_bits: List[int], ft_bits: List[int]) -> Dict[str, float]:
    """Exact McNemar test on discordant pairs (binomial, two-sided). b = base-right/ft-wrong,
    c = base-wrong/ft-right. Small-sample-safe (no chi-square approximation)."""
    from math import comb

    b = sum(1 for x, y in zip(base_bits, ft_bits) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(base_bits, ft_bits) if x == 0 and y == 1)
    nd = b + c
    if nd == 0:
        return {"b": 0, "c": 0, "p_value": 1.0}
    k = min(b, c)
    tail = sum(comb(nd, i) for i in range(0, k + 1)) * (0.5 ** nd)
    p = min(1.0, 2 * tail)
    return {"b": b, "c": c, "p_value": p}


def _load_correct(path: str) -> List[int]:
    return [int(bool(json.loads(l)["correct"])) for l in open(path, encoding="utf-8") if l.strip()]


def compare(base_path: str, ft_path: str) -> Dict[str, object]:
    base, ft = _load_correct(base_path), _load_correct(ft_path)
    n = min(len(base), len(ft))
    base, ft = base[:n], ft[:n]  # align by position (same test set, same order)
    return {
        "n": n,
        "baseline": bootstrap_ci(base),
        "finetuned": bootstrap_ci(ft),
        "paired_gap": paired_gap(base, ft),
        "mcnemar": mcnemar(base, ft),
    }


def summary_line(cmp: Dict[str, object]) -> str:
    b, f, g = cmp["baseline"], cmp["finetuned"], cmp["paired_gap"]
    sig = "significant" if g["p_value"] < 0.05 else "NOT significant"
    return (
        f"baseline {b['mean']*100:.1f}% [{b['lo']*100:.1f}, {b['hi']*100:.1f}]  →  "
        f"fine-tuned {f['mean']*100:.1f}% [{f['lo']*100:.1f}, {f['hi']*100:.1f}]  |  "
        f"gap +{g['gap']*100:.1f}pp (95% CI [{g['lo']*100:.1f}, {g['hi']*100:.1f}], "
        f"bootstrap p={g['p_value']:.3f}, McNemar p={cmp['mcnemar']['p_value']:.3f}) — {sig}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="baseline predictions.jsonl (per-example {correct})")
    ap.add_argument("--finetuned", required=True, help="fine-tuned predictions.jsonl")
    ap.add_argument("--out", default="results/eval_stats.json")
    args = ap.parse_args()
    cmp = compare(args.base, args.finetuned)
    import os

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(cmp, open(args.out, "w"), indent=2)
    print(summary_line(cmp))
    print(f"[eval-stats] wrote {args.out}")


if __name__ == "__main__":
    main()
