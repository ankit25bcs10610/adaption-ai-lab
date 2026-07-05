"""Optional local DPO pass over the SFT checkpoint using TRL — config-driven.

Use this if you want to run DPO yourself (Adaption's platform may also support preference tuning —
prefer that if available). Requires a GPU. Hyperparameters come from `config.yaml` (the `dpo:` block);
any CLI flag overrides its config value, so a reviewer reproduces from one file.

Input: data/out/pref.jsonl (from build_preference.py) with {prompt, chosen, rejected}.
Run: python -m autoscientist_toolcaller.train_dpo --base <sft_model_or_id> [--data ...] [--beta ...]
"""
from __future__ import annotations

import argparse
from typing import Any, Dict


def _dpo_params(cfg_dpo: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge config `dpo:` defaults with CLI overrides (CLI wins when explicitly set). Pure + testable."""
    def pick(cli, key, fallback):
        return cli if cli is not None else cfg_dpo.get(key, fallback)
    return {
        "beta": float(pick(args.beta, "beta", 0.1)),
        "learning_rate": float(pick(args.lr, "learning_rate", 5e-6)),
        "epochs": float(pick(args.epochs, "epochs", 1.0)),
        "per_device_batch_size": int(cfg_dpo.get("per_device_batch_size", 2)),
        "grad_accum": int(cfg_dpo.get("grad_accum", 8)),
        "lora_r": int(cfg_dpo.get("lora_r", 16)),
        "lora_alpha": int(cfg_dpo.get("lora_alpha", 32)),
        "lora_dropout": float(cfg_dpo.get("lora_dropout", 0.05)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="SFT checkpoint (HF id or path)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="data/out/pref.jsonl")
    ap.add_argument("--out", default="out/dpo")
    ap.add_argument("--epochs", type=float, default=None, help="override config dpo.epochs")
    ap.add_argument("--lr", type=float, default=None, help="override config dpo.learning_rate")
    ap.add_argument("--beta", type=float, default=None, help="override config dpo.beta")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config)) if args.config else {}
    p = _dpo_params((cfg or {}).get("dpo", {}), args)

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
        r=p["lora_r"], lora_alpha=p["lora_alpha"], lora_dropout=p["lora_dropout"],
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # attention + MLP (MLP matters most)
    )
    cfg_dpo = DPOConfig(
        output_dir=args.out, num_train_epochs=p["epochs"], learning_rate=p["learning_rate"],
        beta=p["beta"], per_device_train_batch_size=p["per_device_batch_size"],
        gradient_accumulation_steps=p["grad_accum"], warmup_ratio=0.1, logging_steps=10,
        save_strategy="epoch", bf16=True, seed=args.seed, data_seed=args.seed, report_to="trackio",
    )
    trainer = DPOTrainer(model=model, args=cfg_dpo, train_dataset=ds,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.out)
    print(f"[dpo] saved -> {args.out}  (beta={p['beta']}, lr={p['learning_rate']}, epochs={p['epochs']})")


if __name__ == "__main__":
    main()
