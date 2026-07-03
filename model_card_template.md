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
model-index:
  - name: autoscientist-toolcaller
    results:
      - task: {type: text-generation, name: Function Calling}
        dataset: {name: autoscientist-toolcaller-test, type: pandeyankit84/autoscientist-toolcaller-dataset}
        metrics:
          - {type: accuracy, name: overall_accuracy, value: 0.000}
---

# autoscientist-toolcaller

A function-calling / tool-use model fine-tuned with **Adaption AutoScientist** for the AutoScientist
Challenge (category: *All Other Domains*). Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`. The dataset
teaches the decision most tool-use datasets ignore: **when *not* to call a tool.**

## What's different

~40% of the training set is the decisions everyone else skips — **refuse** (no applicable tool),
**clarify** (a required argument is missing), **disambiguate** (two plausible tools), **resist
over-refusal** (hedged-but-satisfiable → still call), and **complete every call** (partial-parallel) —
plus multi-turn (BFCL v4 style), execution-verified environment trajectories, and a schema-drift slice.
Every synthetic example is correct by construction; the set is deduplicated, decontaminated against
public probes, and passed a two-pass data-quality audit. Every answer is one JSON envelope:
`{"action": "call"|"refuse"|"clarify", "calls": [...], "message": "..."}`.

## Results (base vs. fine-tuned)

Held-out test set; identical greedy decoding for both; bootstrapped standard error. Reproduce with
`bash scripts/finalize.sh` (baseline → multi-seed eval → paired significance → gap decomposition).

<!--METRICS_START-->
| Metric | Base | Fine-tuned |
|---|---|---|
| Overall accuracy | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Positive (tool-call) accuracy | 0.000 | 0.000 |
| Refusal accuracy | 0.000 | 0.000 |
| Clarify accuracy | 0.000 | 0.000 |
| **Hallucination rate on hard negatives** ↓ | 0.000 | 0.000 |
<!--METRICS_END-->

**Separately** — Adaptive Data's **dataset-quality grade** (this is a *data* metric, not model accuracy):
`score_before` → `score_after` **7.0 → 8.1, +15.7%, grade C → B**.

## Depth of AutoScientist usage

Adaptive Data recipes (`deduplication` + `reasoning_traces`) + a brand-controls blueprint enforcing
call/refuse/clarify discipline; the platform-enhanced dataset was consumed. Trained via the AutoScientist
loop; where run, a second `preference_pairs` objective on the execution-labeled DPO set targets the
refuse/clarify moat directly. See `docs/AUTOSCIENTIST_USAGE.md`.

## Intended use

Agent / tool-calling pipelines that need reliable JSON tool calls **and safe abstention**. Feed the
available tools (JSON Schema) + the user request; the model returns one JSON envelope.

## Training

- Platform: Adaption AutoScientist on the adapted dataset; base `Qwen/Qwen2.5-Coder-3B-Instruct`. Seed 42.
- Data: curated from `Team-ACE/ToolACE` (Apache-2.0) + original synthetic hard-negative / multi-turn /
  schema-drift slices + execution-verified env data. Deduplicated (MinHash + semantic), decontaminated,
  cross-split leakage removed.

## Limitations

- English-only in this version.
- Optimized for tool-calling reliability; not a general chat model.
- Relaxed argument matching in eval; strict-format consumers should post-validate the JSON.

## Links

- Dataset (HF): https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
- Dataset (Kaggle): https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
- Demo: https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab
