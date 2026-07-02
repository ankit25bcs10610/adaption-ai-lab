"""Baseline / post-finetune VLM eval for the Data-Viz track.

Runs a vision-language model over a chart-QA split with identical greedy decoding and scores with the
hardened relaxed-accuracy harness (eval_chart). Use for the honest 'before' number and the 'after'
number on the same split. Lazy transformers import.

Run:  python -m src.viz.baseline --model Qwen/Qwen3-VL-8B-Instruct --data data/viz/test.jsonl \
        --out results/viz_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def hf_vlm_generate_fn(model_id: str, max_new_tokens: int = 64) -> Callable[[List[Dict[str, Any]]], str]:
    """Return generate_fn(messages)->str for a HF vision-language model (Qwen3-VL / Gemma-3)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
    model.eval()

    def _to_hf(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # transformers extracts content blocks of type "image"/"video" with keys url/path/base64 —
        # NOT the OpenAI {"type":"image_url","image_url":{"url":...}} shape used for Together/Adaption.
        out = []
        for m in messages:
            c = m["content"]
            if isinstance(c, list):
                nc = []
                for blk in c:
                    if blk.get("type") == "image_url":
                        nc.append({"type": "image", "url": blk["image_url"]["url"]})
                    else:
                        nc.append(blk)
                out.append({"role": m["role"], "content": nc})
            else:
                out.append(m)
        return out

    def _gen(messages: List[Dict[str, Any]]) -> str:
        inputs = processor.apply_chat_template(
            _to_hf(messages), add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip()

    return _gen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/viz/test.jsonl")
    ap.add_argument("--out", default="results/viz_baseline.json")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    from .eval_chart import evaluate

    records = load_jsonl(args.data)
    gen = hf_vlm_generate_fn(args.model, args.max_new_tokens)
    metrics = evaluate(records, gen)
    metrics["model"] = args.model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(metrics, open(args.out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
