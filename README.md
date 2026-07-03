<div align="center">

# Adaption AI Lab

### Data-centric submissions for the [Adaption AutoScientist Challenge](https://adaptionlabs.ai/blog/autoscientist-challenge)

**Two models, each built around an original, self-verifying dataset — a function-calling model that knows when *not* to call a tool, and a multimodal chart-reader that works in English and Hindi.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?logo=nextdotjs&logoColor=white)](https://nextjs.org/)
[![Tests](https://img.shields.io/badge/tests-198%20passing-22C55E)](#testing)
[![Data quality](https://img.shields.io/badge/Adaptive%20Data-C%20%E2%86%92%20B%20(%2B15.7%25)-8B5CF6)](#the-result-so-far)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#license--acknowledgements)

[**Live demo**](https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo) · [Datasets](#live-artifacts-published--open) · [Data-quality audit](docs/DATA_QUALITY_AUDIT.md) · [Submission status](SUBMISSION.md)

</div>

---

## The premise

The AutoScientist Challenge automates the model-training loop, so the competitive edge shifts from **compute to data**. This repo takes that literally: each track is a reproducible, seeded pipeline whose one job is to produce a dataset that is *correct by construction* and aimed squarely at where baselines fail. No hand-labeling, no noise to cap accuracy, no leakage.

| Track | Category | The bet | Originality lever |
|---|---|---|---|
| **Function-Calling** (`src/`) | All Other Domains | A tool-use model that **refuses and clarifies** instead of hallucinating a call | Hard negatives (no-tool / missing-arg / ambiguous) + multi-turn + schema-drift + a 5-language reliability slice |
| **Data Visualization** (`src/viz/`) | Data Visualization | A chart-reader that competes where text-only entrants can't — **and speaks Hindi** | Self-verifying synthetic chart generator + Devanagari/romanized slice + a text-only Vega-Lite spec-reading modality |

Everything is **offline-testable**: heavy deps (`torch`, `transformers`, `datasets`, the Adaption SDK) are lazily imported, so the correctness-critical logic runs on stdlib + `numpy` alone. `python -m tests.smoke_test` and `python -m tests.viz.test_viz` pass with **198 checks, zero downloads, zero keys**.

---

## The result so far

> **Honest status.** Adaptive Data is *data-centric* — it grades and improves the **dataset**; it does not hand back model weights. So the number below is a real, measured **data-quality grade**, not a model-accuracy claim. The base-vs-fine-tuned tables on the site are labeled *target / illustrative* and become real only after a training run (a GPU step — see [run-day](#run-day)).

The build pipeline was adversarially audited, then re-graded on the platform:

| | Before | After |
|---|---|---|
| **Adaptive Data quality grade** | C (7.0) | **B (8.1)** — **+15.7%** |
| Refuse cases (`no_tool`) | 8 | **239** |
| Clarify cases (`miss_param`) | 1 | **36** |
| Disambiguate cases | 0 | **133** |
| Schema-invalid gold calls | 36% | **0%** |

The middle rows are the whole thesis: the "refuse / clarify / disambiguate" moat was being *generated and then silently discarded by dedup* until the audit caught it. Full before/after in [`docs/DATA_QUALITY_AUDIT.md`](docs/DATA_QUALITY_AUDIT.md).

### Live artifacts (published, open)

| | Hugging Face | Kaggle |
|---|---|---|
| Tool-calling dataset | [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset) · [model card](https://huggingface.co/pandeyankit84/autoscientist-toolcaller) | [dataset](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset) |
| Chart-QA dataset | [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset) | [dataset](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset) |

**▶️ Live demo** — [call / refuse / clarify, in your browser](https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo).

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
- [Run-day](#run-day)
- [License & acknowledgements](#license--acknowledgements)

---

## Repository layout

```
.
├── src/                       # Function-calling track
│   ├── build_dataset.py         xLAM + ToolACE + Toucan → curate → hard-neg / multi-turn / drift → dedup → split
│   ├── hard_negatives.py        no-tool / missing-arg / ambiguous generators  ← the moat
│   ├── multiturn.py             BFCL miss_param / miss_func / long_context
│   ├── schema_drift.py          tools whose schema changed under the model
│   ├── multilingual.py          matched-twin reliability slice — en / hi / hi-rom / es / fr
│   ├── curriculum.py            per-example difficulty scoring + curriculum ordering
│   ├── reasoning.py             optional <think> traces distilled from gold (gated off by default)
│   ├── dedup.py · decontaminate.py    MinHash + semantic near-dup  ·  cross-split & BFCL leakage removal
│   ├── quality_filter.py · claude_judge.py    heuristic scorer  ·  LLM-as-judge (Anthropic SDK)
│   ├── build_preference.py      DPO pairs (chosen = refuse/clarify, rejected = hallucinated call)
│   ├── eval_bfcl.py · eval_harness.py · eval_decompose.py    BFCL-aligned scoring + gap decomposition
│   ├── eval_stats.py · robustness_table.py · reliability_probe.py    bootstrapped CIs, robustness, over-refusal probe
│   ├── recipe_ablation.py       recipe grid → grade-per-config table (depth of AutoScientist usage)
│   ├── train_adaption.py        Adaption AutoScientist SDK run  ·  baseline.py  honest before-number
│   ├── manifest.py · release.py · fill_model_card.py · export_gguf.py · export_bfcl.py
│   └── viz/                   # Data-visualization track
│       ├── synth_charts.py      self-verifying generator (9 chart types, GT correct by construction)
│       ├── indic_charts.py      Hindi/Devanagari + romanized slice (paired en/hi twins)
│       ├── vega_spec.py         text-only Vega-Lite spec-reading QA (second modality, no VLM needed)
│       ├── eval_chart.py        hardened ChartQA relaxed-accuracy scorer
│       ├── format_utils.py      canonical schema + multimodal chat/base64 rendering
│       └── build_dataset.py · train_adaption.py · baseline.py · gallery.py
├── web/                       # Next.js 14 + react-three-fiber landing page (in-browser tool-call playground)
├── site/                      # Zero-build single-file landing page (Three.js)
├── app/                       # Gradio demo (function-calling)
├── tests/                     # Offline suites — tests/smoke_test.py (141) · tests/viz/test_viz.py (57)
├── docs/                      # WINNING · DATA_QUALITY_AUDIT · AUTOSCIENTIST_USAGE · DATASHEET · CONSOLE_STEPS …
├── scripts/                   # run_all.sh (full pipeline) · finalize.sh (one-command run-day)
└── requirements.txt · config.yaml
```

---

## Quickstart

```bash
git clone https://github.com/ankit25bcs10610/adaption-ai-lab.git
cd adaption-ai-lab

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline test suites — no downloads, no API keys
python -m tests.smoke_test        # function-calling   → ALL PASS (141)
python -m tests.viz.test_viz      # data-visualization → ALL PASS (57)
```

Credentials are read from the environment only when you reach the training / release steps:

```bash
export ADAPTION_API_KEY=pt_live_...     # adaptionlabs.ai
export HF_TOKEN=...                      # or: hf auth login
export KAGGLE_USERNAME=... KAGGLE_KEY=...
export ANTHROPIC_API_KEY=...            # optional: Claude LLM-as-judge quality filter
```

---

## Track 1 — Function calling

**Thesis.** Most function-calling datasets contain only examples where a tool *should* be called. But real agents — and the Berkeley Function-Calling Leaderboard's irrelevance category — punish a model for inventing a call when none applies, or guessing a missing argument. Almost nobody trains for that. That gap is the entire edge.

**One JSON envelope for every decision** (`src/format_utils.py`):

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
python -m src.build_dataset    --config config.yaml   # build, curate, dedup, DECONTAMINATE, split
python -m src.build_preference --config config.yaml   # DPO pairs (+ execution-labeled env pairs)
python -m src.baseline         --config config.yaml   # honest "before" number
python -m src.train_adaption   --config config.yaml   # Adaption AutoScientist run
python -m src.eval_bfcl        --model <finetuned> --data data/out/test.jsonl
python -m src.eval_decompose   --base results/baseline_pred.jsonl --finetuned results/pred.jsonl
python -m src.reliability_probe --model <finetuned>   # legible call/refuse/clarify/over-refusal probe
python -m src.eval_report                             # HTML: significance + decomposition + confusion + drift
python -m src.release preflight                       # blocks publish on missing artifacts / placeholders
python -m src.fill_model_card  --username <you>       # auto-fill MODEL_CARD.md from results/*.json
```

**What makes the data original**

- **Hard negatives** (`hard_negatives.py`) — no-tool, missing-arg, and ambiguous cases the model must *not* answer with a call. This is the moat.
- **Multilingual reliability** (`multilingual.py`) — matched twins across **English, Hindi, romanized Hindi, Spanish, and French**, correct by construction, so you can measure whether "should I refuse?" survives a language switch.
- **Curriculum** (`curriculum.py`) — every example is scored for difficulty and can be emitted in easy→hard order.
- **Optional reasoning traces** (`reasoning.py`) — short `<think>` rationales distilled from the gold decision, gated **off** by default so they never contaminate the strict-format baseline.

**Data-quality audit — two adversarial passes.** Because the dataset *is* the product, the build was audited before release. Pass 1 caught the refuse/clarify moat being generated and then silently discarded by dedup (`no_tool` **8 → 239**, `miss_param` **1 → 36**, `ambiguous` **0 → 133**), plus a slice-mixing undershoot and a DPO poison-pair risk. Pass 2 caught a schema-drift poison bug (**36% of `rename` gold calls were schema-invalid → 0%**), added over-refusal-trap and partial-parallel slices, execution-verified multi-call trajectories, a leakage **decontamination** pass, and a blocking release preflight. `stats.json` carries `mix` (intended-vs-realized shares + `mix_ok`) and `contamination` blocks. Full write-up: [`docs/DATA_QUALITY_AUDIT.md`](docs/DATA_QUALITY_AUDIT.md).

Sources are permissively licensed: `Salesforce/xlam-function-calling-60k` (CC-BY-4.0), `Team-ACE/ToolACE` (Apache-2.0), optional `Agent-Ark/Toucan-1.5M` (Apache-2.0). Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`. Full strategy: [`docs/WINNING.md`](docs/WINNING.md).

---

## Track 2 — Data visualization

**Thesis.** A chart is an image, so text-only entrants can't compete — and the gap is wide and measurable (on CharXiv, GPT-4o scores ~47% on reasoning questions vs ~80% human). Three levers most competitors lack:

1. **Self-verifying synthetic charts** (`viz/synth_charts.py`) — matplotlib renders 9 chart types and the QA ground-truth is *computed from the underlying data*, with on-chart value labels produced by the **same formatter** as the gold answer. A perfectly-reading model is never graded wrong. Hardened against ~30 adversarial edge cases (integer-quantum ties, unique-extremum guards, Pearson-r scatter, series-named questions on multi-series charts), including compound multi-hop QA (`compare_then_compute`).
2. **Hindi/Devanagari + romanized slice** (`viz/indic_charts.py`) — the same ground-truth with localized labels and questions; numeric gold stays ASCII, categorical gold is the on-chart string. Paired `en`/`hi` twins share a `pair_id` for a clean matched-pair Δaccuracy — the HackIndia impact story.
3. **Vega-Lite spec-reading** (`viz/vega_spec.py`) — a *text-only* modality where the "chart" is a Vega-Lite JSON spec and the model answers from `data.values`. No pixels, no VLM, same relaxed scorer — a second, cheaper axis of chart comprehension.

```bash
python -m src.viz.build_dataset --out data/viz --n-synth 400 --n-indic 600 --n-vega 150
python -m src.viz.gallery       --data-dir data/viz --n 18     # browsable HTML gallery
python -m src.viz.baseline      --model Qwen/Qwen3-VL-8B-Instruct --data data/viz/test.jsonl
python -m src.viz.train_adaption --data data/viz/train_tab.jsonl --dry-run
```

Primary data: `hewei2001/ReachQA` (MIT). Base VLM: `Qwen/Qwen3-VL-8B-Instruct` or `google/gemma-3-4b-it` (LoRA). See [`src/viz/README.md`](src/viz/README.md).

---

## Frontend

| App | Stack | Highlights |
|---|---|---|
| `web/` | Next.js 14 · react-three-fiber · Tailwind · framer-motion | 3D hero, **in-browser tool-call playground** (toggle a tool off → a call becomes a refusal), animated benchmarks, dark/light toggle |
| `site/` | Single HTML file · Three.js · Tailwind CDN | Zero-build variant of the same landing page |

```bash
cd web && npm install && npm run dev     # → http://localhost:3000
```

The site keeps **measured** numbers (the +15.7% grade, the audit) strictly separate from **projected** ones (base-vs-fine-tuned bars, labeled *target*), so nothing on the page can be read as a claim it can't back.

---

## Testing

Both suites are fully offline (no model downloads, no API keys) and exercise the correctness-critical logic — scorers, ground-truth generation, determinism, and every edge case surfaced by adversarial review.

```bash
python -m tests.smoke_test      # 141 checks — function-calling
python -m tests.viz.test_viz    #  57 checks — data-visualization (scorer edge cases + synth GT + split integrity)
```

Every module is `py_compile`-clean. The data-viz scorer and synthetic generator were each hardened against a dedicated adversarial pass (percent reconciliation, Indic digits / lakh grouping, cross-language trend matching, train/test image leakage, series ambiguity, and more).

---

## Reproducibility

- **Seeded end to end** — every generator threads one `random.Random(seed)`; per-example RNG is derived by hashing `(seed, index)`, so output is byte-identical across runs.
- **A written manifest** (`manifest.py`) records seeds, source versions, and slice counts alongside each build.
- **No silent leakage** — MinHash + semantic dedup, cross-split leakage removal, BFCL decontamination, novel-tool / novel-chart-type holdouts, and group-splitting so multi-question charts and paired en/hi twins never straddle splits.
- **Permissive licensing throughout** — clean for the mandatory dual Hugging Face + Kaggle release.

---

## How this maps to the judging criteria

| Criterion | Where it's addressed |
|---|---|
| **Measurable improvement over baseline** | `baseline.py` / `viz/baseline.py` + `eval_bfcl.py` / `eval_chart.py` — identical decoding, bootstrapped std error, base-vs-fine-tuned tables, gap decomposition |
| **Dataset quality & originality** | Hard-negative + multi-turn + schema-drift generators; 5-language reliability slice; self-verifying synthetic charts + Vega spec-reading; dedup + decontamination + LLM-as-judge |
| **Real-world impact** | Agent tool-use reliability (knowing when *not* to call); multilingual (Hindi) chart understanding |
| **Depth of AutoScientist usage** | `train_adaption.py` (SDK end-to-end with recipes, brand controls, `estimate=True`) + `recipe_ablation.py` grade-per-config grid + the measured C→B grade |
| **Open-release quality** | Auto-filled model + dataset cards, pinned deps, seeds + manifest, HTML eval report, GGUF export, live demos on HF + Kaggle |

---

## Run-day

The one remaining step is a model-training run on the AutoScientist console (produces weights + a held-out number). Console steps: [`docs/CONSOLE_STEPS.md`](docs/CONSOLE_STEPS.md). Once you have weights, a single command fills every real number and rewrites the model card:

```bash
MODEL=<weights> bash scripts/finalize.sh
# baseline → multi-seed eval → significance → decomposition → robustness → probe → report → model card → preflight
```

Submission status and remaining checklist: [`SUBMISSION.md`](SUBMISSION.md) · platform-usage depth (recipes, run IDs, evidence): [`docs/AUTOSCIENTIST_USAGE.md`](docs/AUTOSCIENTIST_USAGE.md).

---

## License & acknowledgements

Code and generated datasets are released under **Apache-2.0**. Third-party data retains its upstream license (xLAM CC-BY-4.0; ToolACE / Toucan / ReachQA Apache-2.0 / MIT) with attribution preserved in the dataset cards. Built for the Adaption AutoScientist Challenge × HackIndia.

<div align="center"><sub>🤖 Engineered with <a href="https://claude.com/claude-code">Claude Code</a></sub></div>
