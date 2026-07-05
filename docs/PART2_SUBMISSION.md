# Part 2 submission — answer sheets (paste when the form opens ~July 6)

Part 2 runs **July 6 → Aug 3**; winners **Aug 10**. The AutoScientist Challenge has **10 categories**
with awards **per category**, so you have **two complete tracks → submit BOTH** to double your shots:

- **Entry A — All Other Domains** → function-calling / tool-use.
- **Entry B — Data Visualization** (official: "data visualization and chart interpretation") → multimodal
  chart-QA (English + Hindi/Devanagari).

> **Official requirements (verified 2026-07-05 against adaptionlabs.ai/blog/autoscientist-challenge):**
> release **both the dataset AND the trained weights to Hugging Face AND Kaggle**; a **social post on
> LinkedIn + X tagging @adaption_ai and adaption-labs**; judging is a **measurable % improvement over a
> baseline on Adaption's held-out in-house test set per category** (the hard gate), plus **bonus points
> for releasing a demo**. There is **no WhatsApp/Discord eligibility gate** — the HackIndia "Adaptive Data
> Track" (deadline ~June 15) is a *separate, earlier* event, not the AutoScientist Challenge.

> ⚠️ **First, confirm whether one participant may submit to multiple categories** (the form selects
> category per submission; ask the organizers / office hours). If yes, submit both. If only one, submit the
> one with the completed AutoScientist run + strongest held-out number.

`✅ ready to paste · ✍️ you fill · ⏳ needs the AutoScientist training run first.`

## You fill (shared across both entries)
- ✍️ **First / Last name** · **Email** · **Phone** (+91)
- ✍️ **Job Title** · **Company** (student → college / "Independent")
- ✍️ **Address** (real — they mail swag) · **Team members / captain** (or just you)
- ✍️ If the form asks it: category + team details. *(No WhatsApp/Discord join is required for the
  AutoScientist Challenge; Discord is optional/community. Only mark "part of HackIndia" if you actually
  entered that separate track before its ~June 15 deadline.)*

---

## Entry A — All Other Domains (function-calling)
- **Which category:** `All Other Domains`
- **Specify dataset ID:** `d92279d3-90c5-4da7-aef3-e506aa291cd6`
  *(cleaned + diversified; use once it shows **Job Completed** — fallback `a99c0c96-…` is completed at +10%)*
- **Final dataset (HF):** `https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset`
- **Kaggle:** `https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset`
- **Hugging Face model:** `https://huggingface.co/pandeyankit84/autoscientist-toolcaller`
- **Live demo:** `https://autoscientist-toolcaller.vercel.app` (+ HF Space)
- ⏳ **Training Model ID / weights link / held-out %** — from the AutoScientist run (Step 4).

## Entry B — Data Visualization (chart-QA, en + hi)
- **Which category:** `Data Visualization`
- **Final dataset (HF):** `https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset`
- **Kaggle:** `https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset`
- **Originality:** self-verifying synthetic chart generator (9 types, GT correct-by-construction) +
  Devanagari/romanized slice with matched en/hi twins + a text-only Vega-Lite spec-reading modality.
- **Release cards (ready):** [`VIZ_DATASET_CARD.md`](../VIZ_DATASET_CARD.md) (filled, 471 rows) +
  [`VIZ_MODEL_CARD.md`](../VIZ_MODEL_CARD.md) (weights-pending, honest — metrics fill on run-day).
- ⏳ **Dataset ID** — upload the chart-QA set to **Adaptive Data** + grade it (not yet run; the FC set was).
- ⏳ **Training Model ID / weights link / held-out %** — run **AutoScientist** on the chart-QA set
  (base VLM `Qwen/Qwen3-VL-8B-Instruct` or `google/gemma-3-4b-it`, LoRA). Create HF model repo
  `pandeyankit84/autoscientist-chartqa` + a Kaggle model, publish weights.

---

## Before you submit (both entries)
1. **Confirm multi-category eligibility** with the organizers (office hours / the challenge page).
2. **AutoScientist training** — Entry A on `d92279d3`; Entry B on the chart-QA set (`docs/RUN_DAY.md`,
   `docs/CONSOLE_STEPS.md`). Each gives weights + the held-out % (criterion #1 — the eligibility gate).
3. `MODEL=<weights> bash scripts/finalize.sh` (Entry A) → fills MODEL_CARD + report with real numbers.
4. Publish weights to HF + Kaggle for each entry (I can run `autoscientist_toolcaller/release.py`).
5. Post on LinkedIn + X, tag @adaption_ai / Adaption — one post per entry (`docs/social_posts.md`).
6. Paste each sheet into the Part 2 form.
