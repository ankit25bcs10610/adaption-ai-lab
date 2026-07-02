"""Establish the honest baseline: run the eval on the RAW base model, BEFORE any fine-tuning.

You cannot claim "measurable improvement" without this number, and it must be produced with the exact
same decoding settings the post-finetune eval uses (config.eval.temperature). This script is a thin
wrapper that pins those settings from config.yaml so base and fine-tuned runs can't drift apart.

Run:  python -m src.baseline --config config.yaml
Writes results/baseline.json
"""
from __future__ import annotations

import argparse
import json
import os

import yaml

from .eval_harness import evaluate, hf_generate_fn, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="results/baseline.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    test_path = os.path.join(cfg["paths"]["out_dir"], "test.jsonl")
    records = load_jsonl(test_path)
    gen = hf_generate_fn(
        cfg["base_model"],
        max_new_tokens=cfg["eval"]["max_new_tokens"],
        temperature=cfg["eval"]["temperature"],
    )
    metrics = evaluate(records, gen)
    metrics["model"] = cfg["base_model"]
    metrics["role"] = "baseline"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print("[baseline]", json.dumps(metrics, indent=2))
    print(
        f"\n>> Baseline overall accuracy: {metrics['overall_accuracy']:.3f} "
        f"(±{metrics['overall_stderr']:.3f}); hallucination rate on hard negatives: "
        f"{metrics['hallucination_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
