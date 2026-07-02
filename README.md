<div align="center">

# Adaption AI Lab

**Two data-centric model submissions for the [Adaption AutoScientist Challenge](https://adaptionlabs.ai/blog/autoscientist-challenge) — a reliable function-calling model and a multimodal chart-understanding model, each built around an original, self-verifying dataset.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/tests-93%20passing-22C55E)](#testing)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#license)

</div>

---

## Overview

The AutoScientist Challenge automates the model training loop, so the competitive edge is **the dataset, not the compute**. This repository takes that premise literally: each track is a reproducible, seeded data pipeline whose originality lever is a dataset that is *correct by construction* and targets exactly where baselines fail.

| Track | Category | Core idea | Originality lever |
|---|---|---|---|
| **Function-Calling** (`src/`) | All Other Domains | A tool-use model that **refuses and clarifies** instead of hallucinating tool calls | Hard-negative examples (no-tool / missing-arg / ambiguous) + multi-turn + schema-drift slices |
| **Data Visualization** (`src/viz/`) | Data Visualization | A multimodal **chart-reading** model, English + Hindi | Self-verifying synthetic chart generator + Devanagari/romanized chart-QA slice (HackIndia track) |

Both tracks ship an evaluation harness, model/dataset cards, an open-release path (Hugging Face + Kaggle), and a live demo. Everything is offline-testable: heavy dependencies (`torch`, `transformers`, `datasets`, the Adaption SDK) are lazily imported so the correctness-critical logic runs with only stdlib + `numpy`.

### Live artifacts (published, open)

| | Hugging Face | Kaggle |
|---|---|---|
| Tool-calling dataset | [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset) · [model card](https://huggingface.co/pandeyankit84/autoscientist-toolcaller) | [dataset](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset) |
| Chart-QA dataset | [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset) | [dataset](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset) |

**▶️ Live demo:** https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo (call / refuse / clarify, in-browser).

Submission status + remaining steps: [`SUBMISSION.md`](SUBMISSION.md). Platform usage depth (recipes,
run IDs, evidence): [`docs/AUTOSCIENTIST_USAGE.md`](docs/AUTOSCIENTIST_USAGE.md). Data-quality audit:
[`docs/DATA_QUALITY_AUDIT.md`](docs/DATA_QUALITY_AUDIT.md).

---

## Table of contents

- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Track 1 — Function calling](#track-1--function-calling)
- [Track 2 — Data visualization](#track-2--data-visualization)
- [Frontend](#frontend)
- [Testing](#testing)
- [Reproducibility](#reproducibility)
- [How this maps to the judging criteria](#how-this-maps-to-the-judging-criteria)
- [Roadmap](#roadmap)
- [License & acknowledgements](#license--acknowledgements)

---

## Repository layout

```
.
├── src/                     # Function-calling track
│   ├── build_dataset.py       xLAM + ToolACE + Toucan → curate → hard-neg / multi-turn / schema-drift → dedup → split
│   ├── hard_negatives.py      no-tool / missing-arg / ambiguous generators (the moat)
│   ├── multiturn.py           BFCL miss_param / miss_func / long_context
│   ├── schema_drift.py        tools whose schema changed under the model
│   ├── dedup.py               MinHash + semantic near-dup + cross-split leakage check
│   ├── quality_filter.py      heuristic scorer  ·  claude_judge.py  LLM-as-judge (Anthropic SDK)
│   ├── build_preference.py    DPO pairs (chosen=refuse/clarify, rejected=hallucinated call) → train_dpo.py
│   ├── eval_harness.py        strict scorer   ·  eval_bfcl.py  BFCL-aligned + novel-tools split
│   ├── error_analysis.py      failures by category/kind   ·  eval_report.py  HTML report + confusion matrix
│   ├── train_adaption.py      Adaption AutoScientist SDK run   ·  baseline.py  honest before-number
│   ├── export_gguf.py · export_bfcl.py · release.py · fill_model_card.py
│   └── viz/                 # Data-visualization track
│       ├── synth_charts.py     self-verifying chart generator (9 chart types, GT correct by construction)
│       ├── indic_charts.py     Hindi/Devanagari + romanized slice (paired en/hi twins)
│       ├── eval_chart.py       hardened ChartQA relaxed-accuracy scorer
│       ├── format_utils.py     canonical schema + multimodal chat/base64 rendering
│       ├── build_dataset.py · train_adaption.py · baseline.py · gallery.py
│       └── model_card_template.md · dataset_card_template.md · README.md
├── web/                     # Next.js 14 + react-three-fiber landing page (in-browser tool-call playground)
├── site/                    # Zero-build single-file landing page (Three.js)
├── app/                     # Gradio ZeroGPU demo (function-calling)
├── tests/                   # Offline test suites (tests/smoke_test.py, tests/viz/test_viz.py)
├── docs/WINNING.md          # Research-backed strategy notes
├── scripts/run_all.sh       # End-to-end function-calling pipeline
├── requirements.txt · config.yaml
```

---

## Quickstart

```bash
git clone https://github.com/ankit25bcs10610/adaption-ai-lab.git
cd adaption-ai-lab

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the offline test suites (no downloads, no API keys)
python -m tests.smoke_test        # function-calling  → ALL PASS
python -m tests.viz.test_viz      # data-visualization → ALL PASS
```

Credentials are read from the environment only when you reach the training/release steps:

```bash
export ADAPTION_API_KEY=pt_live_...     # adaptionlabs.ai
export HF_TOKEN=...                      # or: hf auth login
export KAGGLE_USERNAME=... KAGGLE_KEY=...
export ANTHROPIC_API_KEY=...            # optional: Claude LLM-as-judge quality filter
```

---

## Track 1 — Function calling

**Thesis.** Most function-calling datasets only contain examples where a tool *should* be called. Real evaluations (and the Berkeley Function-Calling Leaderboard's irrelevance category) heavily penalize a model for inventing a call when none applies or guessing a missing argument — and almost nobody trains for it. That gap is the entire edge.

**Canonical example** (`src/format_utils.py`) — one JSON envelope for every decision:

```json
{
  "tools": [{ "name": "...", "description": "...", "parameters": { "...JSON Schema..." } }],
  "query": "user request",
  "answer": { "type": "tool_call | refuse | clarify", "calls": [...], "content": "..." },
  "meta": { "source": "xlam|toolace|hard_negative|multiturn|schema_drift", "hn_kind": "no_tool|missing_arg|ambiguous" }
}
```

**Pipeline**

```bash
python -m src.build_dataset   --config config.yaml   # build, curate, dedup, DECONTAMINATE, split
python -m src.build_preference --config config.yaml  # DPO pairs (+ execution-labeled env pairs)
python -m src.baseline        --config config.yaml   # honest "before" number
python -m src.train_adaption  --config config.yaml   # Adaption AutoScientist run
python -m src.eval_bfcl       --model <finetuned> --data data/out/test.jsonl
python -m src.eval_decompose  --base results/baseline_pred.jsonl --finetuned results/pred.jsonl  # gap decomposition
python -m src.reliability_probe --model <finetuned>  # legible call/refuse/clarify/over-refusal probe
python -m src.eval_report                            # HTML: significance + decomposition + confusion + schema-drift
python -m src.release preflight                       # blocks publish on missing artifacts / placeholders / LICENSE
python -m src.fill_model_card --username <you>       # auto-fill MODEL_CARD.md from results/*.json
```

Data sources are permissively licensed only: `Salesforce/xlam-function-calling-60k` (CC-BY-4.0), `Team-ACE/ToolACE` (Apache-2.0), optional `Agent-Ark/Toucan-1.5M` (Apache-2.0). Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`. Full strategy in [`docs/WINNING.md`](docs/WINNING.md).

**Data-quality audit (two adversarial passes).** Because the dataset *is* the product, the build pipeline was adversarially audited before release. Pass 1 caught the refuse/clarify moat being generated and then silently discarded by dedup — `no_tool` **8 → 239**, `miss_param` **1 → 36**, `ambiguous` **0 → 133** — plus a slice-mixing undershoot and a DPO poison-pair risk. Pass 2 caught a schema-drift poison bug (**36% of `rename` gold calls were schema-invalid** → fixed to 0) and added over-refusal-trap + partial-parallel slices, execution-verified multi-call trajectories, a leakage **decontamination** pass, and a blocking release preflight. Full write-up and before/after in [`docs/DATA_QUALITY_AUDIT.md`](docs/DATA_QUALITY_AUDIT.md). `stats.json` carries `mix` (intended-vs-realized shares + `mix_ok`) and `contamination` blocks.

---

## Track 2 — Data visualization

**Thesis.** A chart is an image, so text-only entrants can't compete — and the gap is wide and measurable (on CharXiv, GPT-4o scores ~47% on reasoning questions vs ~80% human). Two levers most competitors lack:

1. **Self-verifying synthetic charts** (`src/viz/synth_charts.py`) — matplotlib renders 9 chart types and the QA ground-truth is computed from the underlying data, with the on-chart value labels produced by the *same* formatter as the gold answer. A perfectly-reading model is never graded wrong. Hardened against ~30 adversarial edge cases (integer-quantum ties, unique-extremum guards, open-gap count thresholds, Pearson-r scatter, series-named questions on multi-series charts).
2. **Hindi/Devanagari + romanized slice** (`src/viz/indic_charts.py`) — reuses the same ground-truth with localized labels and questions, keeping numeric gold ASCII and categorical gold as the on-chart string. Paired `en`/`hi` twins share a `pair_id` for a clean matched-pair Δaccuracy — the HackIndia impact story.

```bash
python -m src.viz.build_dataset --out data/viz --n-synth 400 --n-indic 200
python -m src.viz.gallery       --data-dir data/viz --n 18     # browsable HTML gallery
python -m src.viz.baseline      --model Qwen/Qwen3-VL-8B-Instruct --data data/viz/test.jsonl
python -m src.viz.train_adaption --data data/viz/train_tab.jsonl --dry-run
```

Primary data: `hewei2001/ReachQA` (MIT). Base VLM: `Qwen/Qwen3-VL-8B-Instruct` or `google/gemma-3-4b-it` (LoRA). See [`src/viz/README.md`](src/viz/README.md).

<div align="center"><em>Example (Hindi) — “माह अनुसार तापमान”: title, axes, and categories in Devanagari; ASCII value labels that match the gold by construction.</em></div>

---

## Frontend

| App | Stack | Highlights |
|---|---|---|
| `web/` | Next.js 14 · react-three-fiber · Tailwind · framer-motion | 3D hero, **in-browser tool-call playground** (toggle a tool → a call becomes a refusal), animated benchmarks, dark/light toggle, shadcn/21st.dev-ready |
| `site/` | Single HTML file · Three.js · Tailwind CDN | Zero-build variant of the same landing page |

```bash
cd web && npm install && npm run dev     # → http://localhost:3000
```

---

## Testing

Both suites are fully offline (no model downloads, no API keys) and exercise the correctness-critical logic — scorers, ground-truth generation, determinism, edge cases surfaced by adversarial review.

```bash
python -m tests.smoke_test      # 45 checks — function-calling
python -m tests.viz.test_viz    # 53 checks — data-visualization (scorer edge cases + synth GT + split integrity)
```

Every module is `py_compile`-clean. The data-viz scorer and synthetic generator were each hardened against a dedicated adversarial-review pass (percent reconciliation, Indic digits/lakh grouping, cross-language trend matching, train/test image leakage, series-ambiguity, and more).

---

## Reproducibility

- **Seeded end to end** — every generator threads a single `random.Random(seed)`; per-example RNG is derived by hashing `(seed, index)` (collision-resistant), so output is byte-identical across runs.
- **Pinned dependencies** in `requirements.txt`; `web/package-lock.json` for the frontend.
- **No silent leakage** — MinHash + semantic dedup, cross-split leakage removal, novel-tool / novel-chart-type holdouts, and group-splitting so multi-question charts and paired en/hi twins never straddle splits.
- **Permissive licensing throughout** — clean for the mandatory dual Hugging Face + Kaggle release.

---

## How this maps to the judging criteria

| Criterion | Where it's addressed |
|---|---|
| Measurable improvement over baseline | `baseline.py` / `viz/baseline.py` + `eval_bfcl.py` / `eval_chart.py` — identical decoding, bootstrapped std error, base-vs-fine-tuned tables |
| Dataset quality & originality | Hard-negative + multi-turn + schema-drift generators; self-verifying synthetic charts; Indic slice; dedup + quality filter |
| Real-world impact | Agent tool-use reliability; multilingual (Hindi) chart understanding |
| Depth of AutoScientist usage | `train_adaption.py` — SDK end-to-end with recipes, brand controls, `estimate=True`, and evaluation-summary logging |
| Open-release quality | Auto-filled model + dataset cards, pinned deps, seeds, HTML eval report, GGUF export, live demos |

---

## Roadmap

- [ ] Confirm the per-category baseline model + exact metric in the challenge Discord (`#autoscient-challenge`) — the one external unknown.
- [ ] Run each pipeline on the platform; confirm the improvement margin; fill the cards from `results/`.
- [ ] Publish weights + datasets to Hugging Face **and** Kaggle; ship the demos; post tagging `@adaption_ai`.

---

## License & acknowledgements

Code and generated datasets released under **Apache-2.0**. Third-party data retains its upstream license
(xLAM CC-BY-4.0, ToolACE / Toucan / ReachQA Apache-2.0/MIT) with attribution preserved in the dataset cards.
Built for the Adaption AutoScientist Challenge × HackIndia.

<div align="center"><sub>🤖 Engineered with <a href="https://claude.com/claude-code">Claude Code</a></sub></div>
