# Submission checklist — Adaption AutoScientist Challenge (Part 2)

Cross-checked against the official problem statement. This maps every required step and judging
criterion to its status and the exact artifact, so nothing is missed at submission.

**Part 2 window:** July 6 – Aug 3 (form shared before July 6). Categories: Science, Agriculture,
**Data Visualization**, Math & Code, **All Other Domains**.

**Category:** two complete tracks are ready — pick at submission (you may enter both if allowed):
- **All Other Domains** → the function-calling / tool-use model + dataset (`src/`).
- **Data Visualization** → the multimodal chart-QA dataset (`src/viz/`).

---

## The 8 required steps

| # | Step | Status | Evidence / what's left |
|---|------|--------|------------------------|
| 1 | Sign up at adaptionlabs.ai (1,000 credits) | ✅ done | account active; credits used for runs |
| 2 | Pick a category | ⚠️ **you choose** | recommend **Data Visualization** (less crowded, visually native) or **All Other Domains** |
| 3 | Build dataset with **Adaptive Data** | ✅ done | ran `datasets.run()`; **+15.7% quality, grade C→B** on the fixed set (`c4923b7f`); enhanced dataset downloaded to `data/adaptive_out/` |
| 4 | Train model with **AutoScientist** | ⚠️ **console step** | the Python SDK is dataset-only; the model-training loop + weights are in the **web console** (adaptionlabs.ai/app). Run AutoScientist on the adapted dataset there → produces weights + the official held-out number |
| 5 | Beat the baseline on the held-out test set | ⚠️ **needs step 4** | the platform reports the held-out improvement when AutoScientist training completes. Adaptive Data already shows **+15.7%** dataset-quality gain (criterion #2); the held-out *model* number comes from step 4 |
| 6 | Release **weights + dataset** on **HF and Kaggle** | 🟡 **HF done; Kaggle + weights pending** | **Published to HF:** [dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset) + [model card](https://huggingface.co/pandeyankit84/autoscientist-toolcaller). Remaining: **Kaggle** (`python -m src.release kaggle-dataset/kaggle-model` — needs Kaggle API creds) and the **weights** (from the AutoScientist console run) added to the HF model repo |
| 7 | Post on LinkedIn + X, tag @adaption_ai | ⚠️ **you post** | drafts in `docs/social_posts.md`; add the live-demo URL after deploy (bonus points) |
| 8 | Submit (Part 2 form) | ⚠️ **you submit** | after steps 4–7; form opens before July 6 |

Plus (HackIndia track, necessary): **join the WhatsApp channel** and the **Discord** (`#autoscient-challenge`).

---

## Judging criteria → where we're strong

1. **Measurable improvement over baseline** — Adaptive Data **+15.7% (C→B)** now; official held-out number after the console AutoScientist run.
2. **Dataset quality & originality** — the strongest axis: refuse/clarify/over-refusal/partial-parallel moat, execution-verified env data, schema-drift slice, decontamination, a documented **two-pass data-quality audit** (`docs/DATA_QUALITY_AUDIT.md`) that found + fixed real defects (no_tool 8→239; 36%→0% invalid golds).
3. **Real-world impact** — reliable tool-calling (safe abstention) is a core agent-safety problem; the viz track adds Hindi/Devanagari chart-QA.
4. **Depth of AutoScientist usage** — Adaptive Data recipes (dedup + reasoning traces) + brand-controls blueprint; enhanced dataset consumed; reproducible one-command pipeline.
5. **Open-release quality** — model + dataset cards, README, audit, reproducibility **manifest** (SHA-256 + seeds + versions), blocking release **preflight**, GGUF export, and a live demo site.

---

## What's blocking (and who unblocks it)

| Blocker | Owner | Unblocks |
|---|---|---|
| ~~HF write token~~ | ✅ done | dataset + model card **published to HF** |
| **Kaggle API creds** | **you** | Kaggle half of Step 6 — then `python -m src.release kaggle-dataset` |
| **AutoScientist console run** for weights + official held-out number | **you** (web console) | Steps 4–5 and the weights half of Step 6 (add weights to the HF model repo) |
| LinkedIn/X posts, WhatsApp + Discord join | **you** | Step 7 + HackIndia eligibility |
| **Rotate** the pasted credentials (HF write, Adaption key) | **you** | security — they were shared in chat |

Everything else — dataset, audit, cards, eval harness, reproducibility, and the live demo — is built and
committed. Once you drop a HF write token + Kaggle creds, publishing both artifacts to both platforms is
a single command each (see `README.md` → pipeline, gated by `python -m src.release preflight`).
