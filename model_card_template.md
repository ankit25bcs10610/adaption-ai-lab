---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
language: [en]
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
base_model_relation: finetune
datasets:
  - YOUR_USERNAME/autoscientist-toolcall-dataset
tags:
  - function-calling
  - tool-use
  - autoscientist
  - adaption-labs
  - qlora
model-index:
  - name: autoscientist-toolcall
    results:
      - task:
          type: text-generation
          name: Function Calling
        dataset:
          name: autoscientist-toolcall-test
          type: YOUR_USERNAME/autoscientist-toolcall-dataset
        metrics:
          - type: accuracy
            name: overall_accuracy
            value: 0.000   # <- fill from results/eval.json
---

# autoscientist-toolcall

A function-calling / tool-use model fine-tuned with **Adaption AutoScientist** for the AutoScientist
Challenge. Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.

## What's different: it knows when *not* to call a tool

The training data mixes standard tool-call examples with a ~28% slice of **hard negatives** — requests
where the correct behavior is to **refuse** (no applicable tool), **ask for a missing required argument**,
or **disambiguate** between two plausible tools. Baselines hallucinate tool calls in exactly these cases;
this model is trained to handle them.

## Results (base vs. fine-tuned)

Identical greedy decoding for both; standard error is bootstrapped. Reproduce with
`python -m src.eval_harness` (see repo).

<!--METRICS_START-->
| Metric | Base | Fine-tuned |
|---|---|---|
| Overall accuracy | 0.000 ± 0.000 | 0.000 ± 0.000 |
| Positive (tool-call) accuracy | 0.000 | 0.000 |
| Refusal accuracy | 0.000 | 0.000 |
| Clarify accuracy | 0.000 | 0.000 |
| **Hallucination rate on hard negatives** ↓ | 0.000 | 0.000 |
<!--METRICS_END-->

Adaption AutoScientist `evaluation_summary` (from `results/adaption_run.json`):
`grade_before`, `grade_after`, `improvement_percent`.

## Intended use

Agent/tool-calling pipelines that need reliable JSON tool calls and safe abstention. Output is a single
JSON envelope: `{"action":"call"|"refuse"|"clarify", ...}` (see repo `src/format_utils.py`).

## Training

- Method: QLoRA (4-bit), co-optimized by AutoScientist; base `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Data: curated from `Salesforce/xlam-function-calling-60k` (CC-BY-4.0) and `Team-ACE/ToolACE`
  (Apache-2.0), plus synthesized hard negatives grounded in the real tool schemas. Deduplicated
  (MinHash + semantic) with cross-split leakage removed.
- Seed: 42. Full config in `config.yaml`.

## Reproduction

```bash
pip install -r requirements.txt
python -m src.build_dataset --config config.yaml
python -m src.baseline --config config.yaml
python -m src.train_adaption --config config.yaml
python -m src.eval_harness --model <finetuned> --data data/out/test.jsonl --out results/eval.json
```

## Dataset provenance & licensing

Derived only from permissively licensed sources (CC-BY-4.0, Apache-2.0); hard negatives are original.
Released under Apache-2.0. Attribution to xLAM (Salesforce) and ToolACE preserved in the dataset card.

## Limitations

- English-only in this version.
- Exact-match argument scoring is strict; semantically-correct-but-differently-formatted args count as
  misses in eval.
- Optimized against the Adaption held-out set for this category; generalization beyond tool-calling is
  not claimed.

## Links

- Dataset: https://huggingface.co/datasets/YOUR_USERNAME/autoscientist-toolcall-dataset
- Kaggle model: https://www.kaggle.com/models/YOUR_USERNAME/autoscientist-toolcall
- Demo Space: https://huggingface.co/spaces/YOUR_USERNAME/autoscientist-toolcall-demo
