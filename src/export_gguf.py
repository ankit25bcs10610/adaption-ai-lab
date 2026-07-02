"""Export the fine-tuned model to GGUF quants for the open release (judges reward completeness).

If you trained a LoRA adapter, merge it into the base first (dequantize a 4-bit base to bf16 BEFORE
merging — merging into a 4-bit base degrades quality). Then convert with llama.cpp and quantize
(default Q4_K_M, the standard sweet spot).

This script orchestrates the steps and prints exact commands; it does not vendor llama.cpp.

Usage:
  python -m src.export_gguf --base Qwen/Qwen2.5-Coder-3B-Instruct --adapter out/dpo --out out/merged
  # then follow the printed llama.cpp convert/quantize commands
"""
from __future__ import annotations

import argparse
import os


def merge_adapter(base: str, adapter: str, out: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[gguf] loading base {base} in bf16 (never merge into a 4-bit base) ...")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter)
    print("[gguf] merging adapter ...")
    model = model.merge_and_unload()
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    print(f"[gguf] merged model -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to merge; omit if already merged")
    ap.add_argument("--out", default="out/merged")
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()

    src = args.out
    if args.adapter:
        merge_adapter(args.base, args.adapter, args.out)
    else:
        src = args.base
        print(f"[gguf] no adapter given; converting {src} directly")

    print(
        "\nNext, with llama.cpp checked out:\n"
        f"  python llama.cpp/convert_hf_to_gguf.py {src} --outfile out/model-f16.gguf --outtype f16\n"
        f"  ./llama.cpp/llama-quantize out/model-f16.gguf out/model-{args.quant}.gguf {args.quant}\n"
        "\nUpload the .gguf next to the safetensors repo and link it via `base_model` in the card."
    )


if __name__ == "__main__":
    main()
