"""Data-ablation study: how does accuracy scale with training-set size?

Two subcommands:
  subsets   -- write train_pc subsets at each fraction (25/50/100%) for you to train on Adaption.
  report    -- given eval JSONs (one per fraction), render a table + optional PNG plot for the card.

The point is a rigor signal for judges (find the minimum viable dataset) and a clean model-card figure.

Usage:
  python -m autoscientist_toolcaller.ablation subsets --config config.yaml --fractions 0.25 0.5 1.0
  # ... train each data/out/ablation/train_pc_<frac>.jsonl on Adaption, eval each -> results/abl_<frac>.json
  python -m autoscientist_toolcaller.ablation report --inputs results/abl_0.25.json results/abl_0.5.json results/abl_1.0.json \
      --out results/ablation.md
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

import yaml

from .format_utils import to_prompt_completion


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def cmd_subsets(args) -> None:
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["paths"]["out_dir"]
    train = _load_jsonl(os.path.join(out_dir, "train.jsonl"))
    rng = random.Random(cfg["seed"])
    rng.shuffle(train)

    abl_dir = os.path.join(out_dir, "ablation")
    os.makedirs(abl_dir, exist_ok=True)
    for frac in args.fractions:
        k = max(1, int(len(train) * frac))
        subset = train[:k]  # nested subsets (25% ⊂ 50% ⊂ 100%) so the comparison is clean
        pc = [to_prompt_completion(ex) for ex in subset]
        path = os.path.join(abl_dir, f"train_pc_{frac}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in pc:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[ablation] {frac:>5}: {k} examples -> {path}")
    print("Now train each subset on Adaption, eval on the SAME test set, save results/abl_<frac>.json")


def cmd_report(args) -> None:
    rows = []
    for path in args.inputs:
        m = json.load(open(path))
        frac = _infer_frac(path)
        rows.append((frac, m.get("overall_accuracy"), m.get("overall_stderr", 0.0),
                     m.get("hallucination_rate")))
    rows.sort(key=lambda r: (r[0] is None, r[0]))

    lines = ["| Train fraction | Overall acc | ± stderr | Hallucination rate |",
             "|---|---|---|---|"]
    for frac, acc, se, hr in rows:
        lines.append(f"| {frac} | {acc:.3f} | {se:.3f} | {hr:.3f} |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    open(args.out, "w").write(md)
    print(md)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            es = [r[2] for r in rows]
            plt.figure(figsize=(5, 3.2))
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
            plt.xlabel("Training fraction")
            plt.ylabel("Overall accuracy")
            plt.title("Data ablation")
            plt.tight_layout()
            png = os.path.splitext(args.out)[0] + ".png"
            plt.savefig(png, dpi=150)
            print(f"[ablation] plot -> {png}")
        except ImportError:
            print("[ablation] matplotlib not installed; skipped plot")


def _infer_frac(path: str):
    base = os.path.basename(path)
    for tok in base.replace(".json", "").split("_"):
        try:
            return float(tok)
        except ValueError:
            continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("subsets")
    s.add_argument("--config", default="config.yaml")
    s.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    s.set_defaults(func=cmd_subsets)

    r = sub.add_parser("report")
    r.add_argument("--inputs", nargs="+", required=True)
    r.add_argument("--out", default="results/ablation.md")
    r.add_argument("--plot", action="store_true")
    r.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
