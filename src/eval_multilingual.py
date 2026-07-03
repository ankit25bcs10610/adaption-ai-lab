"""Multilingual matched-pair evaluation — the headline HackIndia robustness signal.

The multilingual slice (`src/multilingual.py`) emits matched twins: the SAME tools + gold envelope with
the user request phrased in en / hi / hi-rom / es / fr, all sharing a `meta.pair_id`. That design exists
to measure whether the model's call / refuse / clarify decision SURVIVES a language switch. This module is
the consumer the slice was built for: it reports per-language accuracy and, crucially, the matched-pair
**Δaccuracy(lang − en)** with a paired bootstrap CI + p-value — so the cross-language gap is a real,
significance-tested number, not an artifact of which examples happened to land in each language.

Pure/offline (stdlib + the project's own eval helpers). CLI:
  python -m src.eval_multilingual --model <id> [--adapter path] --data data/out/test.jsonl \
      --out results/eval_multilingual.json [--strict]
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Tuple

from .eval_bfcl import judge_bfcl
from .eval_stats import bootstrap_ci, paired_gap
from .format_utils import build_eval_prompt


def _delta_metrics(bits: Dict[Tuple[Any, str], int]) -> Dict[str, Any]:
    """Given correctness bits keyed by (pair_id, lang), compute per-language accuracy and the
    matched-pair Δaccuracy(lang − en) with a paired bootstrap CI. No model — testable in isolation."""
    langs = sorted({lang for (_, lang) in bits})
    by_lang: Dict[str, Any] = {}
    for lang in langs:
        vals = [b for (_, l), b in bits.items() if l == lang]
        ci = bootstrap_ci(vals)
        by_lang[lang] = {
            "n": len(vals),
            "accuracy": (sum(vals) / len(vals)) if vals else None,
            "lo": ci["lo"],
            "hi": ci["hi"],
        }
    en_pairs = {pid for (pid, l) in bits if l == "en"}
    deltas: Dict[str, Any] = {}
    for lang in langs:
        if lang == "en":
            continue
        # only pairs that have BOTH an en twin and this-lang twin -> a clean matched pair
        pids = sorted((pid for (pid, l) in bits if l == lang and pid in en_pairs), key=str)
        en_bits = [bits[(pid, "en")] for pid in pids]
        lang_bits = [bits[(pid, lang)] for pid in pids]
        pg = paired_gap(en_bits, lang_bits)  # gap = acc(lang) - acc(en)
        deltas[lang] = {
            "n_pairs": len(pids),
            "delta_vs_en": pg["gap"],
            "lo": pg["lo"],
            "hi": pg["hi"],
            "p_value": pg["p_value"],
        }
    return {"languages": langs, "by_lang": by_lang, "matched_pair_delta_vs_en": deltas}


def _is_multilingual(rec: Dict[str, Any]) -> bool:
    m = rec.get("meta", {}) or {}
    return m.get("source") == "multilingual" and m.get("pair_id") is not None and m.get("lang") is not None


def evaluate_multilingual(
    records: List[Dict[str, Any]],
    generate_fn: Callable[[str], str],
    lenient: bool = True,
) -> Dict[str, Any]:
    """Score the multilingual twins in `records` and return per-language + matched-pair Δ metrics."""
    ml = [r for r in records if _is_multilingual(r)]
    bits: Dict[Tuple[Any, str], int] = {}
    for r in ml:
        v = judge_bfcl(r, generate_fn(build_eval_prompt(r)), lenient=lenient)
        bits[(r["meta"]["pair_id"], r["meta"]["lang"])] = int(v["correct"])
    metrics = _delta_metrics(bits)
    metrics["n"] = len(ml)
    metrics["mode"] = "lenient" if lenient else "strict"
    return metrics


def to_markdown(m: Dict[str, Any]) -> str:
    """Compact table for the model card / eval report."""
    if not m.get("languages"):
        return ""
    lines = ["### Multilingual robustness (matched-pair)", "", "| Lang | n | Accuracy | Δ vs en | p |", "|---|---|---|---|---|"]
    by_lang, deltas = m["by_lang"], m["matched_pair_delta_vs_en"]
    for lang in m["languages"]:
        acc = by_lang[lang]["accuracy"]
        acc_s = f"{acc:.3f}" if acc is not None else "n/a"
        if lang == "en":
            lines.append(f"| en | {by_lang[lang]['n']} | {acc_s} | — | — |")
        else:
            d = deltas.get(lang, {})
            dv = d.get("delta_vs_en")
            dv_s = f"{dv:+.3f}" if dv is not None else "n/a"
            lines.append(f"| {lang} | {by_lang[lang]['n']} | {acc_s} | {dv_s} | {d.get('p_value', float('nan')):.3f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out", default="results/eval_multilingual.json")
    ap.add_argument("--strict", action="store_true", help="use strict equality instead of lenient")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    import os
    from .eval_harness import hf_generate_fn, load_jsonl

    records = load_jsonl(args.data)
    gen = hf_generate_fn(args.model, args.max_new_tokens, args.temperature, args.adapter)
    metrics = evaluate_multilingual(records, gen, lenient=not args.strict)
    metrics["model"], metrics["adapter"] = args.model, args.adapter
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("\n" + to_markdown(metrics))


if __name__ == "__main__":
    main()
