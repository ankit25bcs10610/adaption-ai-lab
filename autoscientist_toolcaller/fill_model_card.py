"""Auto-fill the model card from results JSONs so numbers are never transcribed by hand.

Reads:
  results/baseline.json     (base model eval)
  results/eval.json         (fine-tuned eval)   -- or pass --finetuned
  results/adaption_run.json (optional; AutoScientist evaluation_summary)

Writes MODEL_CARD.md: substitutes the <!--METRICS_START-->..<!--METRICS_END--> block, the YAML
`value:` line (overall accuracy), and YOUR_USERNAME. Any missing file degrades gracefully to "n/a".

Usage:
  python -m autoscientist_toolcaller.fill_model_card --username myuser \
    --baseline results/baseline.json --finetuned results/eval.json \
    --template model_card_template.md --out MODEL_CARD.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Optional


def _load(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def _cell(v: Any, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f}"


def _row(label: str, base: Dict[str, Any], ft: Dict[str, Any], key: str, se_key: str = None) -> str:
    b, f = base.get(key), ft.get(key)
    if se_key:
        bse, fse = base.get(se_key, 0.0), ft.get(se_key, 0.0)
        bcell = f"{b:.3f} ± {bse:.3f}" if b is not None else "n/a"
        fcell = f"{f:.3f} ± {fse:.3f}" if f is not None else "n/a"
    else:
        bcell, fcell = _cell(b), _cell(f)
    return f"| {label} | {bcell} | {fcell} |"


def build_metrics_block(base: Dict[str, Any], ft: Dict[str, Any], adaption: Dict[str, Any]) -> str:
    lines = [
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        _row("Overall accuracy", base, ft, "overall_accuracy", "overall_stderr"),
        _row("Positive (tool-call) accuracy", base, ft, "positive_accuracy"),
        _row("Refusal accuracy", base, ft, "refusal_accuracy"),
        _row("Clarify accuracy", base, ft, "clarify_accuracy"),
        _row("**Hallucination rate on hard negatives** ↓", base, ft, "hallucination_rate"),
    ]
    summ = adaption.get("evaluation_summary") if adaption else None
    if isinstance(summ, dict):
        ip = summ.get("improvement_percent")
        if ip is not None:
            lines.append(f"| AutoScientist improvement_percent | — | {ip} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", required=True)
    ap.add_argument("--baseline", default="results/baseline.json")
    ap.add_argument("--finetuned", default="results/eval.json")
    ap.add_argument("--adaption", default="results/adaption_run.json")
    ap.add_argument("--template", default="model_card_template.md")
    ap.add_argument("--out", default="MODEL_CARD.md")
    args = ap.parse_args()

    base = _load(args.baseline)
    ft = _load(args.finetuned)
    adaption = _load(args.adaption)

    text = open(args.template, encoding="utf-8").read()

    block = build_metrics_block(base, ft, adaption)
    text = re.sub(
        r"<!--METRICS_START-->.*?<!--METRICS_END-->",
        f"<!--METRICS_START-->\n{block}\n<!--METRICS_END-->",
        text,
        flags=re.DOTALL,
    )

    # YAML model-index value = fine-tuned overall accuracy (replace the numeric OR __PENDING__ sentinel)
    acc = ft.get("overall_accuracy")
    if acc is not None:
        text = re.sub(r"value:\s*(?:[\d.]+|__PENDING__)", f"value: {acc:.3f}", text, count=1)

    text = text.replace("YOUR_USERNAME", args.username)

    open(args.out, "w", encoding="utf-8").write(text)
    print(f"[card] wrote {args.out}")
    if not base:
        print("  (note: baseline.json missing — run autoscientist_toolcaller.baseline first for real numbers)")
    if not ft:
        print("  (note: finetuned eval missing — run autoscientist_toolcaller.eval_harness first)")


if __name__ == "__main__":
    main()
