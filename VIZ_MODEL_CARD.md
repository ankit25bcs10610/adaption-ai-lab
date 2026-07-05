---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
language: [en, hi]
base_model: Qwen/Qwen3-VL-8B-Instruct
base_model_relation: finetune
datasets:
  - pandeyankit84/autoscientist-chartqa-dataset
tags:
  - chart-understanding
  - chart-question-answering
  - data-visualization
  - multimodal
  - vision-language
  - autoscientist
  - adaption-labs
  - indic
# model-index (held-out relaxed_accuracy) is added by scripts once the AutoScientist VLM run completes —
# omitted while weights are pending so no zero/placeholder metric is ever published.
---

# autoscientist-chartqa

A chart-understanding vision-language model for the Adaption AutoScientist Challenge (**Data
Visualization** category), fine-tuned with **Adaption AutoScientist** on a self-verifying chart-QA
dataset. Base: `Qwen/Qwen3-VL-8B-Instruct` (or `google/gemma-3-4b-it`), LoRA.

> **Status: weights pending.** The dataset (below) is published and open. The trained weights + the
> held-out **relaxed-accuracy** number come from the AutoScientist VLM training run on the console; this
> card's metrics table is filled at that point (see `docs/RUN_DAY.md`). Nothing here is a fabricated
> number — cells read `__PENDING__` until the run completes.

## What's different

Trained on a **self-verifying synthetic chart-QA dataset** — every answer is *computed from the
underlying chart data* (correct by construction, no human labels to cap accuracy) — plus a
**Hindi/Devanagari + romanized** slice with matched en/hi twins for multilingual chart reading. It reads
values, compares categories, finds extrema, computes sums/means/differences/proportions, counts above a
threshold, and identifies trends — in English and Hindi.

## Results (base vs. fine-tuned)

ChartQA-style relaxed accuracy (±5% numeric tolerance), identical greedy decoding for both.

| Metric | Base | Fine-tuned |
|---|---|---|
| Relaxed accuracy (overall) | __PENDING__ | __PENDING__ |
| — English (en) | __PENDING__ | __PENDING__ |
| — Hindi (hi) | __PENDING__ | __PENDING__ |
| Matched-pair Δ (hi − en) | — | __PENDING__ |
| Novel chart-type holdout | __PENDING__ | __PENDING__ |

Separately — Adaptive Data's **dataset-quality grade** on the chart-QA set (a *data* metric, not model
accuracy) is reported from `results/viz_adaption_run.json` once graded.

## Intended use

Reading + reasoning over chart images (bar / line / pie / scatter / grouped / stacked), including Hindi
charts. The answer is the value / category / yes-no / trend only.

## Training

- Method: LoRA VLM fine-tune (vision encoder frozen), co-optimized by AutoScientist.
- Data: self-verifying synthetic charts + Hindi/romanized slice + a text-only Vega-Lite spec-reading
  modality (+ optional ReachQA, MIT). Seed 42, reproducible.

## Limitations

- Synthetic charts are matplotlib-styled; real-world dense charts (e.g. CharXiv) are harder.
- Devanagari conjunct fidelity depends on the render environment (raqm) — see the repo README.
- Numeric answers scored with ±5% relaxed tolerance.

## Links

- Dataset: https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset
- Kaggle dataset: https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab
- Live site: https://autoscientist-toolcaller.vercel.app
