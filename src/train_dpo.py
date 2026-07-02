"""Optional local DPO pass over the SFT checkpoint using TRL.

Use this only if you want to run DPO yourself (Adaption's platform may also support preference tuning —
prefer that if available, since it's the competition's core tool). Requires a GPU.

Input: data/out/pref.jsonl (from build_preference.py) with {prompt, chosen, rejected}.
Run: python -m src.train_dpo --base <sft_model_or_id> --data data/out/pref.jsonl --out out/dpo
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT checkpoint (HF id or path)")
    ap.add_argument("--data", default="data/out/pref.jsonl")
    ap.add_argument("--out", default="out/dpo")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)

    ds = load_dataset("json", data_files=args.data, split="train")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # attention + MLP (MLP matters most)
    )
    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr,
        beta=args.beta, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        warmup_ratio=0.1, logging_steps=10, save_strategy="epoch", bf16=True,
        seed=args.seed, data_seed=args.seed, report_to="trackio",
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[dpo] saved -> {args.out}")


if __name__ == "__main__":
    main()
