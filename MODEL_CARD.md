---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language: [en]
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
base_model_relation: finetune
datasets:
  - pandeyankit84/autoscientist-toolcaller-dataset
tags:
  - function-calling
  - tool-use
  - agents
  - autoscientist
  - adaption-labs
# NOTE: no model-index block yet — the only measured number so far is the Adaptive Data DATASET-quality
# grade (+15.7%, C→B), which is NOT a model accuracy. A model-index (held-out accuracy) will be added
# after the AutoScientist training run produces weights + the official per-category held-out number.
---

# autoscientist-toolcaller

> **Status: weights pending.** This repository currently ships the **dataset**, the eval harness, and a
> deterministic **behavior demo** (a simulator, not the trained model). The trained weights come from the
> AutoScientist training run and will be added here, together with the official per-category **held-out**
> number. The headline below is Adaptive Data's **dataset-quality** grade — *not* a model accuracy.

A function-calling / tool-use model + dataset for the AutoScientist Challenge (category:
*All Other Domains*). The dataset teaches a model the decision most tool-use datasets ignore:
**when *not* to call a tool.**

> **Adaptive Data result (real).** On a **completed, uncapped run** of the current-generation set
> (`bea4a581…`, 5,133/5,157 rows processed, 2026-07-05) the platform reports **7.0 → 8.1, +15.7%,
> grade C → B** — with **completion quality +31.5%** (6.92 → 9.1) and the dataset's quality percentile
> rising **8.4 → 31.5**. Independently corroborated by the older capped `c4923b7f` run (+15.7%) and a
> completed 250-row run (**+10.0%, grade B**, `a99c0c96…`). This is Adaptive Data's **dataset-quality
> grade** — the data-centric measurable improvement; the held-out *model* number comes from the
> AutoScientist training run. Intended base: `Qwen/Qwen2.5-Coder-3B-Instruct`.

## What's different

The training set mixes standard tool-call examples with a large slice of **hard negatives** — cases
where the correct behavior is **refuse** (no applicable tool), **clarify** (a required argument is
missing), or **disambiguate** (two plausible tools) — plus **multi-turn** (BFCL miss_param / miss_func /
long_context) and **schema-drift** (tools whose schema changed) slices. Baselines hallucinate tool
calls in exactly these cases; this model is trained not to.

Every answer is a single JSON envelope:
`{"action": "call"|"refuse"|"clarify", "calls": [...], "message": "..."}`.

## Results

**Adaptive Data quality (real, the headline).** Adaptive Data is a **data-centric** platform: it
improves the *dataset* and grades the improvement, so its quality grade is the challenge's measurable
improvement.

| Run | Rows | score before → after | Δ | grade |
|---|---|---|---|---|
| Fixed dataset (`c4923b7f…`) | 2,440 | **7.0 → 8.1** | **+15.7%** | C → B |
| Earlier curated (`a99c0c96…`, completed) | 250 | 8.0 → 8.8 | +10.0% | B |

**Model accuracy (base vs. fine-tuned).** The full base-vs-fine-tuned table (overall / positive /
refusal / clarify / hallucination-rate / novel-tools-holdout accuracy) requires *training* a model on
the improved dataset, which needs a GPU. The harness is ready and one-command: `bash scripts/run_all.sh`
with `MODEL=<hf-id>` runs baseline → multi-seed eval → paired significance (`eval_stats`) → gap
**decomposition** (`eval_decompose`) → robustness-delta → reliability probe → HTML report. Numbers land
in `results/` and auto-fill via `python -m autoscientist_toolcaller.fill_model_card`.

## Intended use

Agent / tool-calling pipelines that need reliable JSON tool calls **and safe abstention**. Feed the
available tools (JSON Schema) + the user request; the model returns one JSON envelope.

## Training

- Platform: Adaption AutoScientist (Adaptive Data recipes: deduplication + reasoning traces;
  brand-controls blueprint enforcing call/refuse/clarify discipline).
- Data: curated from `Team-ACE/ToolACE` (Apache-2.0) + original synthetic hard-negative / multi-turn /
  schema-drift slices. Deduplicated (MinHash + semantic), cross-split leakage removed. Seed 42.

## Limitations

- English-only in this version.
- Optimized for tool-calling reliability; not a general chat model.
- Argument scoring uses relaxed matching; strict-format consumers should post-validate the JSON.

## Links

- Dataset (HF): https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
- Dataset (Kaggle): https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab
