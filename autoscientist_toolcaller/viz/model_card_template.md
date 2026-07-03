---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
language: [en, hi]
base_model: Qwen/Qwen3-VL-8B-Instruct
base_model_relation: finetune
datasets:
  - YOUR_USERNAME/autoscientist-chartqa-dataset
tags:
  - chart-understanding
  - chart-question-answering
  - data-visualization
  - multimodal
  - vision-language
  - autoscientist
  - adaption-labs
  - indic
model-index:
  - name: autoscientist-chartqa
    results:
      - task: {type: image-text-to-text, name: Chart Question Answering}
        dataset: {name: autoscientist-chartqa-test, type: YOUR_USERNAME/autoscientist-chartqa-dataset}
        metrics:
          - {type: accuracy, name: relaxed_accuracy, value: 0.000}   # fill from results/viz_eval.json
---

# autoscientist-chartqa

A chart-understanding vision-language model fine-tuned with **Adaption AutoScientist** for the Data
Visualization category. Base: `Qwen/Qwen3-VL-8B-Instruct` (or `google/gemma-3-4b-it`).

## What's different

Trained on a **self-verifying synthetic chart-QA dataset** (answers computed from the underlying data,
correct by construction) plus a **Hindi/Devanagari + romanized slice** for multilingual chart reading.
It reads values, compares categories, finds extrema, computes sums/means/differences/proportions,
counts above a threshold, and identifies trends — in English and Hindi.

## Results (base vs. fine-tuned)

ChartQA-style relaxed accuracy (±5% numeric tolerance), identical greedy decoding. Reproduce with
`python -m autoscientist_toolcaller.viz.baseline`. Fill from `results/viz_baseline.json` / `results/viz_eval.json`.

| Metric | Base | Fine-tuned |
|---|---|---|
| Relaxed accuracy (overall) | 0.000 | 0.000 |
| — reasoning split | 0.000 | 0.000 |
| — descriptive split | 0.000 | 0.000 |
| — English (en) | 0.000 | 0.000 |
| — Hindi (hi) | 0.000 | 0.000 |
| Matched-pair Δ (hi − en) ↑ | — | 0.000 |
| Novel chart-type holdout | 0.000 | 0.000 |

AutoScientist `evaluation_summary` (from `results/viz_adaption_run.json`): grade_before / grade_after /
improvement_percent.

## Intended use

Reading and reasoning over chart images (bar/line/pie/scatter/grouped/stacked), including Hindi charts.
Answer is the value/category/yes-no/trend only.

## Training

- Method: LoRA VLM fine-tune, vision encoder frozen, co-optimized by AutoScientist.
- Data: self-verifying synthetic charts + Hindi/romanized slice (+ optional ReachQA, MIT). Seed 42.

## Limitations

- Synthetic charts are matplotlib-styled; real-world (arXiv-dense) charts like CharXiv are harder.
- Devanagari conjunct fidelity depends on the render environment (raqm) — see repo README.
- Numeric answers scored with ±5% relaxed tolerance.

## Links

- Dataset: https://huggingface.co/datasets/YOUR_USERNAME/autoscientist-chartqa-dataset
- Kaggle: https://www.kaggle.com/models/YOUR_USERNAME/autoscientist-chartqa
- Demo: https://huggingface.co/spaces/YOUR_USERNAME/autoscientist-chartqa-demo
