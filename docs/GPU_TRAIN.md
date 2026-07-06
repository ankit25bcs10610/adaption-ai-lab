# GPU fine-tune runbook — the honest path to real weights

This produces a **real** LoRA adapter for the tool-caller from the published dataset, then the repo's
own eval harness computes the **real held-out number**. Nothing is fabricated. Runs on a free
Colab/Kaggle GPU. The trainer's data pipeline (prompt rendering + completion-only masking) is already
verified on this repo's data; only the GPU forward/backward needs a GPU.

## Model choice
The AutoScientist job spec used `mistralai/Mixtral-8x7B-Instruct-v0.1` (46.7B) — that needs an
**A100-80GB** even in 4-bit. On a **free T4/L4** use a smaller strong base; the script defaults to
`Qwen/Qwen2.5-Coder-3B-Instruct` (matches the repo's original model card). Pass `--base` to change it.

| GPU you have | Recommended `--base` |
|---|---|
| Free Colab/Kaggle T4 (16 GB) | `Qwen/Qwen2.5-Coder-3B-Instruct` (default) |
| L4 / A10 (24 GB) | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| A100 80 GB | `mistralai/Mixtral-8x7B-Instruct-v0.1` (the original spec) |

## Colab / Kaggle cells (paste in order)

**1. GPU + deps**
```bash
!nvidia-smi -L
!pip -q install "transformers>=4.46" "peft>=0.13" "trl>=0.11" bitsandbytes accelerate datasets huggingface_hub kagglehub
```

**2. Get the code**
```bash
!git clone https://github.com/ankit25bcs10610/adaption-ai-lab.git
%cd adaption-ai-lab
```

**3. Auth (paste your tokens)**
```python
import os
os.environ["HF_TOKEN"] = "hf_..."                 # write token
os.environ["KAGGLE_API_TOKEN"] = "KGAT_..."       # kaggle access token
```

**4. Train (produces the adapter + pushes it to HF)**
```bash
!python scripts/train_lora.py \
  --base Qwen/Qwen2.5-Coder-3B-Instruct \
  --dataset pandeyankit84/autoscientist-toolcaller-dataset \
  --out out/lora_adapter \
  --push-to pandeyankit84/autoscientist-toolcaller
```

**5. Real held-out number (the eligibility gate) — base vs fine-tuned on the repo's test set**
```bash
!bash scripts/finalize.sh MODEL=Qwen/Qwen2.5-Coder-3B-Instruct ADAPTER=out/lora_adapter
!cat HEADLINE.txt
```
This runs baseline → fine-tuned eval → significance → fills `RESULTS.md` + `MODEL_CARD.md`.

**6. Publish weights to BOTH platforms (step 5 already pushed to HF in step 4; this adds Kaggle + verifies)**
```bash
!python -m autoscientist_toolcaller.release preflight     --dir out/lora_adapter
!python -m autoscientist_toolcaller.release kaggle-model  --slug pandeyankit99/autoscientist-toolcaller --dir out/lora_adapter
```

**7. Send me back**
- the `HEADLINE.txt` line (base % → fine-tuned %), and
- confirmation the HF + Kaggle model repos now have weight files.

Then I finalize the cards/site here and you do the two human-only steps: **post** (tag @adaption_ai)
+ **submit the Part 2 form**.

## Notes
- `train_on_inputs=False` is enforced via completion-only label masking (prompt tokens = -100).
- Hyperparameters mirror the job spec: LoRA r=64 / α=128, dropout 0, all-linear, cosine, warmup 0.05,
  wd 0.01, lr 1e-4, 4 epochs, SFT.
- If you used a base other than the default, pass the SAME `--base` / `MODEL=` in steps 4–6.
