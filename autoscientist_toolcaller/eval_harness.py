"""Evaluation harness for function calling, scoring BOTH positives and hard negatives.

The headline metric is overall accuracy, but we break it out so the improvement story is legible:

  positive accuracy   -> right tool + valid, matching arguments
  refusal accuracy     -> correctly refuses when no tool applies
  clarify accuracy      -> correctly asks when an arg is missing / choice is ambiguous
  hallucination rate    -> FRACTION of hard negatives where the model wrongly emitted a tool call
                           (this is the number a plain baseline is bad at; driving it down is the win)

Scoring is decoding-identical for base and fine-tuned models (greedy) so the comparison is fair.
Bootstrapped standard error is reported for the headline metric.

Usage (programmatic): call `evaluate(records, generate_fn)` where generate_fn(prompt)->str.
CLI: python -m autoscientist_toolcaller.eval_harness --model <hf_id_or_path> --data data/out/test.jsonl --out results/eval.json
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Callable, Dict, List, Optional

from .format_utils import build_eval_prompt, build_system_prompt, parse_model_output
from .schema_validator import validate_call


# --------------------------------------------------------------------------------------
# Per-example judging
# --------------------------------------------------------------------------------------
def _args_match(pred: Dict[str, Any], gold: Dict[str, Any]) -> bool:
    """Exact match on argument dicts, order-insensitive."""
    return pred == gold


def judge(example: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    """Return a per-example verdict dict."""
    tools = example["tools"]
    gold = example["answer"]
    gold_type = gold["type"]
    parsed = parse_model_output(output_text)

    result = {
        "hn_kind": example["meta"].get("hn_kind"),
        "sd_kind": example["meta"].get("sd_kind"),
        "gold_type": gold_type,
        "parsed_ok": parsed is not None,
        "correct": False,
        "hallucinated_call": False,
        "pred_action": None,
    }
    if parsed is None:
        return result
    action = parsed.get("action")
    result["pred_action"] = action if action in ("call", "refuse", "clarify") else None

    if gold_type == "tool_call":
        if action != "call":
            return result
        pred_calls = parsed.get("calls") or []
        gold_calls = gold["calls"]
        if len(pred_calls) != len(gold_calls):
            return result
        # order-insensitive match on (name, arguments), with schema validity required
        matched = True
        remaining = list(gold_calls)
        for pc in pred_calls:
            ok, _ = validate_call(pc, tools)
            if not ok:
                matched = False
                break
            hit = next(
                (g for g in remaining if g["name"] == pc.get("name")
                 and _args_match(pc.get("arguments", {}), g.get("arguments", {}))),
                None,
            )
            if hit is None:
                matched = False
                break
            remaining.remove(hit)
        result["correct"] = matched and not remaining
        return result

    # hard negative: gold is refuse or clarify
    if action == "call":
        result["hallucinated_call"] = True
        return result
    # accept refuse<->clarify leniency? No — grade exactly, that's the signal.
    result["correct"] = action == ("refuse" if gold_type == "refuse" else "clarify")
    return result


# --------------------------------------------------------------------------------------
# Aggregate metrics
# --------------------------------------------------------------------------------------
def _bootstrap_se(bits: List[int], n_boot: int = 1000, seed: int = 42) -> float:
    if not bits:
        return 0.0
    rng = random.Random(seed)
    n = len(bits)
    means = []
    for _ in range(n_boot):
        s = sum(bits[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    mu = sum(means) / len(means)
    var = sum((m - mu) ** 2 for m in means) / len(means)
    return var ** 0.5


def evaluate(
    records: List[Dict[str, Any]],
    generate_fn: Callable[[str], str],
    system_from_example: bool = True,
    return_records: bool = False,
) -> Dict[str, Any]:
    verdicts: List[Dict[str, Any]] = []
    details: List[Dict[str, Any]] = []
    for ex in records:
        prompt = build_eval_prompt(ex)
        out = generate_fn(prompt)
        v = judge(ex, out)
        verdicts.append(v)
        if return_records:
            details.append({"example": ex, "output": out, "verdict": v})

    def acc(subset: List[Dict[str, Any]]) -> float:
        return sum(v["correct"] for v in subset) / len(subset) if subset else 0.0

    positives = [v for v in verdicts if v["gold_type"] == "tool_call"]
    refusals = [v for v in verdicts if v["gold_type"] == "refuse"]
    clarifies = [v for v in verdicts if v["gold_type"] == "clarify"]
    hard = refusals + clarifies

    overall_bits = [int(v["correct"]) for v in verdicts]
    metrics = {
        "n": len(verdicts),
        "overall_accuracy": sum(overall_bits) / len(overall_bits) if overall_bits else 0.0,
        "overall_stderr": _bootstrap_se(overall_bits),
        "positive_accuracy": acc(positives),
        "refusal_accuracy": acc(refusals),
        "clarify_accuracy": acc(clarifies),
        "hallucination_rate": (
            sum(v["hallucinated_call"] for v in hard) / len(hard) if hard else 0.0
        ),
        "parse_failure_rate": sum(not v["parsed_ok"] for v in verdicts) / len(verdicts),
        "counts": {"positive": len(positives), "refuse": len(refusals), "clarify": len(clarifies)},
    }

    # First-class schema-drift slice: schema-awareness is a distinctive claim, so score it explicitly.
    sd_verdicts = [v for v in verdicts if v.get("sd_kind")]
    if sd_verdicts:
        by_sd: Dict[str, Any] = {}
        for kind in sorted({v["sd_kind"] for v in sd_verdicts}):
            sub = [int(v["correct"]) for v in sd_verdicts if v["sd_kind"] == kind]
            by_sd[kind] = {"n": len(sub), "accuracy": sum(sub) / len(sub), "stderr": _bootstrap_se(sub)}
        sd_bits = [int(v["correct"]) for v in sd_verdicts]
        metrics["by_sd_kind"] = by_sd
        metrics["schema_drift_accuracy"] = sum(sd_bits) / len(sd_bits)

    # Accuracy by difficulty band (easy/medium/hard) — computed on the fly from each example, so it
    # works without the build having tagged. Shows where the base→fine-tuned gap concentrates
    # (a fine-tune that helps most on HARD examples is the compelling, honest signal for criterion #1).
    from .curriculum import band as _band, difficulty as _difficulty
    dbands = [_band(_difficulty(r)) for r in records]
    by_diff: Dict[str, Any] = {}
    for b in ("easy", "medium", "hard"):
        sub = [int(verdicts[i]["correct"]) for i, bb in enumerate(dbands) if bb == b]
        if sub:
            by_diff[b] = {"n": len(sub), "accuracy": sum(sub) / len(sub), "stderr": _bootstrap_se(sub)}
    if by_diff:
        metrics["by_difficulty"] = by_diff

    # Calibration: confusion matrix (gold action × predicted action) + abstention rates. Does the model
    # abstain when it should (refuse/clarify) WITHOUT over-refusing on satisfiable requests?
    acts = ("call", "refuse", "clarify")
    confusion = {g: {p: 0 for p in (*acts, "none")} for g in acts}
    for v in verdicts:
        g = "call" if v["gold_type"] == "tool_call" else v["gold_type"]
        p = v.get("pred_action") or "none"
        confusion[g][p] += 1
    n_call = sum(confusion["call"].values())
    over_refused = confusion["call"]["refuse"] + confusion["call"]["clarify"]
    pred_abstain = sum(confusion[g][p] for g in acts for p in ("refuse", "clarify"))
    should_abstain = sum(sum(confusion[g].values()) for g in ("refuse", "clarify"))
    hit_abstain = sum(confusion[g][p] for g in ("refuse", "clarify") for p in ("refuse", "clarify"))
    metrics["calibration"] = {
        "confusion": confusion,
        "over_refusal_rate": (over_refused / n_call) if n_call else 0.0,
        "abstention_precision": (hit_abstain / pred_abstain) if pred_abstain else 0.0,
        "abstention_recall": (hit_abstain / should_abstain) if should_abstain else 0.0,
    }

    if return_records:
        return {"metrics": metrics, "records": details}
    return metrics


# --------------------------------------------------------------------------------------
# HF generation backend (used by baseline.py and post-finetune eval)
# --------------------------------------------------------------------------------------
def hf_generate_fn(
    model_id: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    adapter: Optional[str] = None,
) -> Callable[[str], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def _gen(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-6),
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    return _gen


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local path")
    ap.add_argument("--adapter", default=None, help="optional LoRA adapter path")
    ap.add_argument("--data", default="data/out/test.jsonl")
    ap.add_argument("--out", default="results/eval.json")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    records = load_jsonl(args.data)
    gen = hf_generate_fn(args.model, args.max_new_tokens, args.temperature, args.adapter)
    metrics = evaluate(records, gen)
    metrics["model"] = args.model
    metrics["adapter"] = args.adapter

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
