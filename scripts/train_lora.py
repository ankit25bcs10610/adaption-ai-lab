"""QLoRA SFT trainer for the AutoScientist tool-caller — the honest GPU path to real weights.

Produces a REAL LoRA adapter from the published dataset, using the same prompt/completion rendering
as the eval harness (autoscientist_toolcaller.format_utils.to_prompt_completion) so training matches
inference byte-for-byte. Hyperparameters mirror the AutoScientist job spec (LoRA r=64 / alpha=128,
cosine schedule, 4 epochs, SFT, train_on_inputs=False -> completion-only loss).

Run on a GPU (Colab/Kaggle/cloud). See docs/GPU_TRAIN.md.

Example:
  python scripts/train_lora.py \
    --base Qwen/Qwen2.5-Coder-3B-Instruct \
    --dataset pandeyankit84/autoscientist-toolcaller-dataset \
    --out out/lora_adapter \
    --push-to pandeyankit84/autoscientist-toolcaller     # optional; needs HF_TOKEN with write

After it finishes, get the REAL held-out number with the repo's eval:
  bash scripts/finalize.sh MODEL=<base> ADAPTER=out/lora_adapter
Then publish:
  python -m autoscientist_toolcaller.release hf-model    --repo pandeyankit84/autoscientist-toolcaller --dir out/lora_adapter
  python -m autoscientist_toolcaller.release kaggle-model --slug pandeyankit99/autoscientist-toolcaller --dir out/lora_adapter
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from autoscientist_toolcaller.format_utils import to_prompt_completion


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    # NOTE: the AutoScientist spec used mistralai/Mixtral-8x7B-Instruct-v0.1 (46.7B) — that needs an
    # A100-80GB even in 4-bit. Default here fits a free Colab/Kaggle T4/L4; override --base for Mixtral.
    ap.add_argument("--base", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    ap.add_argument("--dataset", default="pandeyankit84/autoscientist-toolcaller-dataset")
    ap.add_argument("--out", default="out/lora_adapter")
    ap.add_argument("--push-to", default=None, help="HF model repo id to push the adapter to (optional)")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit (needs more VRAM)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---- data: render with the SAME function eval uses, then mask the prompt (train_on_inputs=False)
    raw = load_dataset(args.dataset, data_files="train.jsonl", split="train")

    def encode(ex):
        pc = to_prompt_completion(ex, tokenizer=tok)
        prompt_ids = tok(pc["prompt"], add_special_tokens=False)["input_ids"]
        completion_ids = tok(pc["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = (prompt_ids + completion_ids)[: args.max_len]
        # completion-only loss: -100 over the prompt span so the model is graded on the answer only
        labels = ([-100] * len(prompt_ids) + completion_ids)[: args.max_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

    ds = raw.map(encode, remove_columns=raw.column_names, desc="rendering+tokenizing")

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids, labs, att = [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            labs.append(b["labels"] + [-100] * n)
            att.append(b["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labs),
            "attention_mask": torch.tensor(att),
        }

    # ---- model: QLoRA 4-bit by default
    quant = None
    if not args.no_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if not args.no_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",  # matches spec: lora_trainable_modules = all-linear
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        seed=args.seed,
        report_to=[],
        gradient_checkpointing=True,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate)
    trainer.train()

    # save adapter + tokenizer + a small provenance record
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "training_provenance.json"), "w") as f:
        json.dump(
            {
                "base_model": args.base,
                "dataset": args.dataset,
                "method": "qlora-sft" if not args.no_4bit else "lora-sft",
                "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                         "target_modules": "all-linear"},
                "epochs": args.epochs, "lr": args.lr, "scheduler": "cosine",
                "warmup_ratio": args.warmup_ratio, "weight_decay": args.weight_decay,
                "train_on_inputs": False, "seed": args.seed,
            },
            f, indent=2,
        )
    print(f"[train] adapter saved -> {args.out}")

    if args.push_to:
        model.push_to_hub(args.push_to, token=os.environ.get("HF_TOKEN"))
        tok.push_to_hub(args.push_to, token=os.environ.get("HF_TOKEN"))
        print(f"[train] pushed adapter -> https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
