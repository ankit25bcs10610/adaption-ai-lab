# Social posts — AutoScientist Challenge (ready to post)

Post from your own accounts, tag **@adaption_ai** on X and **Adaption** (Adaption Labs) on LinkedIn,
then paste both post URLs into the submission form. The only slot left is `<DEMO_URL>` (add it after you
deploy the site to Vercel; delete that line if you skip the demo).

**Published links (real):**
- HF model: https://huggingface.co/pandeyankit84/autoscientist-toolcaller
- HF dataset (tool-calling): https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
- HF dataset (chart-QA): https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset
- Kaggle dataset (tool-calling): https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
- Kaggle dataset (chart-QA): https://www.kaggle.com/datasets/pandeyankit99/autoscientist-chartqa-dataset
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab

---

## X / Twitter (single post)

> Taught a small open model the hardest part of tool use: **when *not* to call a tool.**
>
> Built for the @adaption_ai AutoScientist Challenge — a data-centric function-calling model that
> refuses, clarifies, and calls instead of hallucinating tools.
>
> Adaptive Data quality **+15.7% (grade C→B)**. Weights + dataset fully open 👇
>
> 🤗 https://huggingface.co/pandeyankit84/autoscientist-toolcaller
> 📊 https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
> 💻 https://github.com/ankit25bcs10610/adaption-ai-lab
>
> #AutoScientist #HackIndia #AI

---

## X / Twitter (thread, optional)

1/ Most function-calling datasets only teach a model to *call* tools. Real agents fail on the opposite:
inventing a tool call when none fits, or guessing a missing argument. I trained an open model on exactly
those failure modes. @adaption_ai AutoScientist Challenge 🧵

2/ The moat is ~40% "when NOT to call" data: refuse (no tool applies), clarify (missing arg / ambiguous),
over-refusal traps (hedged-but-doable → still call), and partial-parallel (two intents → two calls). All
correct-by-construction.

3/ I ran a 2-pass adversarial audit on my own build pipeline. It caught the moat nearly shipping empty —
refuse examples 8 → 239, and 36% → 0% schema-invalid gold calls. Wrote it all up in the repo.

4/ Adaptive Data (@adaption_ai) graded the dataset **7.0 → 8.1, +15.7%, C→B**. Everything's open —
weights, dataset, eval harness, reproducibility manifest, and a second Data-Viz (Hindi chart-QA) track.

5/ Model + data 👇
🤗 https://huggingface.co/pandeyankit84/autoscientist-toolcaller
📊 https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
💻 https://github.com/ankit25bcs10610/adaption-ai-lab
#AutoScientist #HackIndia

---

## LinkedIn (professional)

**Teaching an AI model when *not* to call a tool.**

I built an open-source, data-centric function-calling model for the Adaption AutoScientist Challenge
(with HackIndia). Most tool-use datasets only teach a model to call tools — but real agents break on the
opposite decision: inventing a tool call when none applies, or guessing a missing argument. That gap is
the whole edge.

So ~40% of the training set is the decisions everyone else skips — refuse, clarify, disambiguate, resist
over-refusal, and complete every call — all synthesized correct-by-construction from real tool schemas.

A few things I'm proud of:
• **Adaptive Data quality +15.7% (grade C → B)** — measured by Adaption's platform.
• A **two-pass adversarial audit** of my own build pipeline that caught the moat nearly shipping empty
  (refuse examples 8 → 239; 36% → 0% schema-invalid gold calls) — documented, with regression tests.
• Execution-verified environments, a decontamination pass, and a full reproducibility manifest.
• A second **Data-Visualization** track: a self-verifying chart-QA dataset with a Hindi/Devanagari slice.

Everything is open — weights, both datasets, the eval harness, and a live demo.

🤗 Model: https://huggingface.co/pandeyankit84/autoscientist-toolcaller
🤗 Dataset: https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
📊 Kaggle: https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
💻 Code: https://github.com/ankit25bcs10610/adaption-ai-lab
🔗 Demo: <DEMO_URL>

Huge thanks to Adaption for AutoScientist and Adaptive Data.
#AutoScientist #HackIndia #OpenSource #AI #MachineLearning #LLM #Agents

---

## Posting checklist
- [x] Publish datasets to HF **and** Kaggle (both tracks) — done.
- [ ] Add model **weights** to the HF model repo after the AutoScientist console run.
- [ ] Deploy the demo (Vercel) and fill `<DEMO_URL>`.
- [ ] Post on X (tag @adaption_ai) and LinkedIn (tag Adaption); attach a chart/demo image.
- [ ] Paste both post URLs into the submission form.
