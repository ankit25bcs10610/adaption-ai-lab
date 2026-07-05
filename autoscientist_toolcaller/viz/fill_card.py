"""Auto-fill the Data-Visualization model card from viz eval JSONs — Entry-B parity with
`autoscientist_toolcaller.fill_model_card` (which is FC-only; the viz eval keys differ).

Reads:
  results/viz_baseline.json     (base VLM eval — the eval_chart.evaluate() output)
  results/viz_eval.json         (fine-tuned VLM eval)          -- or pass --finetuned
  results/viz_adaption_run.json (optional; AutoScientist evaluation_summary for the chart-QA set)

Writes VIZ_MODEL_CARD.md from the viz template: replaces the <!--VIZ_METRICS_START-->..<!--VIZ_METRICS_END-->
block, the model-index `value:` line (fine-tuned overall relaxed_accuracy), and YOUR_USERNAME. Any missing
file degrades gracefully to "n/a" (and leaves the model-index `__PENDING__` so preflight still blocks an
unfilled card). Numbers are never hand-transcribed.

Usage:
  python -m autoscientist_toolcaller.viz.fill_card --username pandeyankit84 \
    --baseline results/viz_baseline.json --finetuned results/viz_eval.json \
    --template autoscientist_toolcaller/viz/model_card_template.md --out VIZ_MODEL_CARD.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Optional

_START = "<!--VIZ_METRICS_START-->"
_END = "<!--VIZ_METRICS_END-->"


def _load(path: Optional[str]) -> Dict[str, Any]:
    if path and os.path.exists(path):
        return json.load(open(path))
    return {}


def _cell(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _overall(d: Dict[str, Any]) -> Optional[float]:
    v = d.get("relaxed_accuracy")
    return float(v) if isinstance(v, (int, float)) else None


def _lang(d: Dict[str, Any], lang: str) -> Optional[float]:
    b = (d.get("by_lang") or {}).get(lang)
    return b.get("accuracy") if isinstance(b, dict) else None


def _novel(d: Dict[str, Any]) -> Optional[float]:
    """Accuracy on the held-out novel chart-type split (key contains 'novel')."""
    for k, b in (d.get("by_split") or {}).items():
        if "novel" in str(k).lower() and isinstance(b, dict):
            return b.get("accuracy")
    return None


def build_metrics_block(base: Dict[str, Any], ft: Dict[str, Any]) -> str:
    f_en, f_hi = _lang(ft, "en"), _lang(ft, "hi")
    delta = f"{f_hi - f_en:+.3f}" if (f_hi is not None and f_en is not None) else "n/a"
    rows = [
        "| Metric | Base | Fine-tuned |",
        "|---|---|---|",
        f"| Relaxed accuracy (overall) | {_cell(_overall(base))} | {_cell(_overall(ft))} |",
        f"| — English (en) | {_cell(_lang(base, 'en'))} | {_cell(f_en)} |",
        f"| — Hindi (hi) | {_cell(_lang(base, 'hi'))} | {_cell(f_hi)} |",
        f"| Matched-pair Δ (hi − en) | — | {delta} |",
        f"| Novel chart-type holdout | {_cell(_novel(base))} | {_cell(_novel(ft))} |",
    ]
    return "\n".join(rows)


def fill(template_text: str, base: Dict[str, Any], ft: Dict[str, Any],
         adaption: Dict[str, Any], username: str) -> str:
    """Pure transform (unit-testable, no I/O): template + metrics -> filled card text."""
    block = build_metrics_block(base, ft)
    text = re.sub(
        re.escape(_START) + r".*?" + re.escape(_END),
        f"{_START}\n{block}\n{_END}",
        template_text,
        flags=re.DOTALL,
    )
    # model-index value = fine-tuned overall relaxed accuracy (only fill if we actually have it, so an
    # un-run card keeps __PENDING__ and stays preflight-blocked rather than shipping a fake 0.000).
    ov = _overall(ft)
    if ov is not None:
        text = text.replace("value: __PENDING__", f"value: {ov:.3f}")
    if username:
        text = text.replace("YOUR_USERNAME", username)
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--username", default="pandeyankit84")
    ap.add_argument("--baseline", default="results/viz_baseline.json")
    ap.add_argument("--finetuned", default="results/viz_eval.json")
    ap.add_argument("--adaption", default="results/viz_adaption_run.json")
    ap.add_argument("--template", default="autoscientist_toolcaller/viz/model_card_template.md")
    ap.add_argument("--out", default="VIZ_MODEL_CARD.md")
    args = ap.parse_args()

    template_text = open(args.template, encoding="utf-8").read()
    text = fill(template_text, _load(args.baseline), _load(args.finetuned),
                _load(args.adaption), args.username)
    open(args.out, "w", encoding="utf-8").write(text)
    left = text.count("__PENDING__")
    print(f"[viz.fill_card] wrote {args.out} "
          + ("(all metrics filled)" if left == 0 else f"({left} __PENDING__ remain — run viz eval first)"))


if __name__ == "__main__":
    main()
