# Submission checklist — Adaption AutoScientist Challenge (Part 2)

Cross-checked against the official problem statement. This maps every required step and judging
criterion to its status and the exact artifact, so nothing is missed at submission.

**Part 2 window:** July 6 – Aug 3 (form shared before July 6). Categories: Science, Agriculture,
**Data Visualization**, Math & Code, **All Other Domains**.

**Category:** two complete tracks are ready **and both datasets are published** — pick at submission
(you may enter both if allowed):
- **All Other Domains** → function-calling / tool-use. Dataset on
  [HF](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset) +
  [Kaggle](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset).
- **Data Visualization** → multimodal chart-QA (Hindi/Devanagari). Dataset on
  [HF](https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset) +
  [Kaggle](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset).

---

## The 8 required steps

| # | Step | Status | Evidence / what's left |
|---|------|--------|------------------------|
| 1 | Sign up at adaptionlabs.ai (1,000 credits) | ✅ done | account active; credits used for runs |
| 2 | Pick a category | ⚠️ **you choose** | recommend **Data Visualization** (less crowded, visually native) or **All Other Domains** |
| 3 | Build dataset with **Adaptive Data** | ✅ done | ran `datasets.run()`; **+15.7% quality, grade C→B on a COMPLETED, uncapped run** of the current-gen set (`bea4a581`, 5,133/5,157 rows, 2026-07-05; completion quality **+31.5%**, percentile 8.4→31.5). Corroborated by the older capped `c4923b7f` run (+15.7%) and the completed 250-row `a99c0c96` run (+10%, B); enhanced dataset downloaded to `data/adaptive_out/` |
| 4 | Train model with **AutoScientist** | ⚠️ **console step** | the Python SDK is dataset-only; the model-training loop + weights are in the **web console** (adaptionlabs.ai/app). Run AutoScientist on the adapted dataset there → produces weights + the official held-out number |
| 5 | Beat the baseline on the held-out test set | ⚠️ **needs step 4** | the platform reports the held-out improvement when AutoScientist training completes. Adaptive Data already shows **+15.7%** dataset-quality gain (criterion #2); the held-out *model* number comes from step 4 |
| 6 | Release **weights + dataset** on **HF and Kaggle** | 🟡 **dataset on BOTH done; weights pending** | **Dataset published to both:** HF [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset) + [model card](https://huggingface.co/pandeyankit84/autoscientist-toolcaller); Kaggle [dataset](https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset) (public). Remaining: **model weights** (from the AutoScientist console run) uploaded to the HF model repo + a Kaggle model |
| 7 | Post on LinkedIn + X, tag @adaption_ai | ⚠️ **you post** | ready-to-paste drafts in `docs/social_posts.md`; **live demo already deployed** (bonus): https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo |
| 8 | Submit (Part 2 form) | ⚠️ **you submit** | after steps 4–7; form opens before July 6 |

*(The AutoScientist Challenge requires no WhatsApp/Discord join — Discord is optional/community. HackIndia's
"Adaptive Data Track", deadline ~June 15, is a separate, earlier event.)*

---

## Judging criteria → where we're strong

*Official judging = a measurable held-out % improvement over a baseline per category (the hard gate) + bonus
points for releasing a demo. The numbered list below is our own quality framework mapping (not the verbatim
official rubric).*

1. **Measurable improvement over baseline (held-out)** — ⏳ **pending the AutoScientist training run** (that produces the official per-category held-out model number). Note: the **+15.7% (C→B)** figure is Adaptive Data's **dataset-quality grade**, *not* a model accuracy — we do not claim the eligibility gate on it.
2. **Dataset quality & originality** — the strongest axis: refuse/clarify/over-refusal/partial-parallel moat, execution-verified env data, schema-drift slice, decontamination, a documented **two-pass data-quality audit** (`docs/DATA_QUALITY_AUDIT.md`) that found + fixed real defects (no_tool 8→284; 36%→0% invalid golds, now enforced by a build-time drop-guard).
3. **Real-world impact** — reliable tool-calling (safe abstention) is a core agent-safety problem; the viz track adds Hindi/Devanagari chart-QA.
4. **Depth of AutoScientist usage** — Adaptive Data recipes (dedup + reasoning traces) + brand-controls blueprint; enhanced dataset consumed; reproducible one-command pipeline.
5. **Open-release quality** — model + dataset cards, README, audit, reproducibility **manifest** (SHA-256 + seeds + versions), blocking release **preflight**, GGUF export, and a live demo site.

---

## What's blocking (and who unblocks it)

| Blocker | Owner | Unblocks |
|---|---|---|
| ~~HF write token~~ | ✅ done | dataset + model card **published to HF** |
| ~~Kaggle API creds~~ | ✅ done | dataset **published to Kaggle** (public) |
| **AutoScientist console run** for weights + official held-out number | **you** (web console) | Steps 4–5 and the weights half of Step 6 (add weights to the HF model repo) |
| LinkedIn/X posts (tag @adaption_ai + adaption-labs) | **you** | Step 7 — a required submission item |
| **Rotate** the pasted credentials (HF write, Adaption key) | **you** | security — they were shared in chat |

Everything else — dataset, audit, cards, eval harness, reproducibility, and the live demo — is built and
committed. Once you drop a HF write token + Kaggle creds, publishing both artifacts to both platforms is
a single command each (see `README.md` → pipeline, gated by `python -m autoscientist_toolcaller.release preflight`).
