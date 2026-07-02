# Social posts — AutoScientist Challenge submission

Fill the `<…>` slots after training + publishing:
- `<IMPROVEMENT>` — improvement_percent from the Adaption Measure tab / eval (e.g. "34%").
- `<HF_URL>` — Hugging Face model URL · `<KAGGLE_URL>` — Kaggle URL.
- Tag **@adaption_ai** on X and **Adaption** (Adaption Labs) on LinkedIn. Post from your account, then
  paste both post URLs into the submission form.

---

## X / Twitter (concise)

> Taught a small open model the hardest part of tool use: **when *not* to call a tool.**
>
> Built for the @adaption_ai AutoScientist Challenge — a data-centric function-calling model that
> refuses, clarifies, and calls instead of hallucinating. Trained on @adaption_ai AutoScientist,
> +<IMPROVEMENT> over baseline.
>
> Weights + dataset, fully open 👇
> HF: <HF_URL>
> Kaggle: <KAGGLE_URL>
> Code: https://github.com/ankit25bcs10610/adaption-ai-lab
>
> #AutoScientist #HackIndia #opensource #LLM #functioncalling

**Alt (Data-Viz / HackIndia angle):**

> Charts that read in English *and* Hindi 📊 — a multimodal chart-QA model on a self-verifying dataset
> (answers correct by construction) with a Devanagari slice for #HackIndia. Trained with @adaption_ai
> AutoScientist. Open weights + data: <HF_URL> · <KAGGLE_URL>

---

## LinkedIn (narrative)

> **When should an AI agent NOT call a tool?**
>
> That question is where most function-calling models fail — they invent a tool call when none applies,
> or guess a missing argument instead of asking. For the **Adaption AutoScientist Challenge**, I built a
> data-centric submission that targets exactly that gap.
>
> The idea: since Adaption's AutoScientist automates the training loop, the edge is the **dataset**. So I
> built one around **hard negatives** — examples where the right answer is to *refuse* (no applicable
> tool), *clarify* (a required argument is missing), or *disambiguate* (two plausible tools) — alongside
> multi-turn and "schema-drift" slices. Every synthetic example is correct by construction and
> deduplicated with train/test leakage removed.
>
> Trained end-to-end on **AutoScientist**, it beats the held-out baseline by **<IMPROVEMENT>**, and it's
> released fully open — weights + dataset on Hugging Face and Kaggle, with a reproducible pipeline, an
> eval harness, and a live demo.
>
> I also built a second track: a **multimodal chart-understanding** model that reads charts in **English
> and Hindi** (Devanagari), on a self-verifying synthetic chart dataset — part of the HackIndia track.
>
> 🔗 Hugging Face: <HF_URL>
> 🔗 Kaggle: <KAGGLE_URL>
> 🔗 Code + write-up: https://github.com/ankit25bcs10610/adaption-ai-lab
>
> Huge thanks to **Adaption** and **HackIndia** for the challenge and the platform.
>
> #AutoScientist #HackIndia #AI #MachineLearning #OpenSource #LLM #Agents

---

## Posting checklist
- [ ] Publish weights + dataset to HF **and** Kaggle; paste URLs above.
- [ ] Fill `<IMPROVEMENT>` from the Measure tab.
- [ ] Post on X (tag @adaption_ai) and LinkedIn (tag Adaption); attach a chart/demo image or the live demo link.
- [ ] Paste both post URLs into the submission form.
