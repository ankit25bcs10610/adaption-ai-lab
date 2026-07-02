---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language: [en]
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
base_model_relation: finetune
datasets:
  - pandeyankit8456/autoscientist-toolcaller-dataset
tags:
  - function-calling
  - tool-use
  - agents
  - autoscientist
  - adaption-labs
model-index:
  - name: autoscientist-toolcaller
    results:
      - task: {type: text-generation, name: Function Calling}
        dataset: {name: autoscientist-toolcaller-test, type: pandeyankit8456/autoscientist-toolcaller-dataset}
        metrics:
          - {type: accuracy, name: overall_accuracy, value: 0.000}   # ← fill from results after eval
---

# autoscientist-toolcaller

A function-calling / tool-use model fine-tuned with **Adaption AutoScientist** for the AutoScientist
Challenge (category: *All Other Domains*). It is trained to make the decision most tool-use datasets
ignore: **when *not* to call a tool.**

> **Adaption run:** dataset ID `4bee4b34-fd6b-4343-ae68-f0175fb96ce5` ·
> AutoScientist run `dataset-4bee4b34-…-1782999081016`. Base model selected by AutoScientist
> (see the **Measure** tab); intended base `Qwen/Qwen2.5-Coder-3B-Instruct`.

## What's different

The training set mixes standard tool-call examples with a large slice of **hard negatives** — cases
where the correct behavior is **refuse** (no applicable tool), **clarify** (a required argument is
missing), or **disambiguate** (two plausible tools) — plus **multi-turn** (BFCL miss_param / miss_func /
long_context) and **schema-drift** (tools whose schema changed) slices. Baselines hallucinate tool
calls in exactly these cases; this model is trained not to.

Every answer is a single JSON envelope:
`{"action": "call"|"refuse"|"clarify", "calls": [...], "message": "..."}`.

## Results (base vs. fine-tuned)

Identical greedy decoding for both; ChartQA-style relaxed matching for arguments; bootstrapped standard
error. Reproduce with `python -m src.eval_bfcl`. Fill from `results/eval.json` after the run completes.

| Metric | Base | Fine-tuned |
|---|---|---|
| Overall accuracy | 0.000 | 0.000 |
| Positive (tool-call) accuracy | 0.000 | 0.000 |
| Refusal accuracy | 0.000 | 0.000 |
| Clarify accuracy | 0.000 | 0.000 |
| **Hallucination rate on hard negatives** ↓ | 0.000 | 0.000 |
| Novel-tools holdout accuracy | 0.000 | 0.000 |

AutoScientist `evaluation_summary`: `grade_before` → `grade_after` (`improvement_percent`). The
challenge requires a measurable improvement over the held-out baseline.

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

- Dataset: https://huggingface.co/datasets/pandeyankit8456/autoscientist-toolcaller-dataset
- Kaggle: https://www.kaggle.com/models/pandeyankit8456/autoscientist-toolcaller
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab
