# Social posts — AutoScientist Challenge (ready to post)

Post from your own accounts, tagging **@adaption_ai and adaption-labs** on **both** X and LinkedIn (the
official challenge page requires tagging both handles), then paste both post URLs into the submission
form. A social post is a required submission item. Everything below is ready to copy-paste.

> **One thing to fill on run-day:** after the AutoScientist console run gives you the held-out number,
> prepend this line to the X post and add the bullet to LinkedIn (honest model-accuracy claim):
> `📈 Held-out tool-calling accuracy: base __%  →  fine-tuned __% ([+__pts]).`
> Until then, lead with the **dataset-quality** grade (+15.7%, C→B) below — which is what's measured today.

**Published links (real):**
- **Live site (Vercel):** https://autoscientist-toolcaller.vercel.app  ← the polished demo; lead with this
- Live demo (HF Space): https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo
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
> Adaptive Data quality **+15.7% (grade C→B)**. Live demo + open weights + dataset 👇
>
> Thanks @adaption_ai (Adaption Labs — tag adaption-labs too).
> ▶️ Try it: https://autoscientist-toolcaller.vercel.app
> 🤗 https://huggingface.co/pandeyankit84/autoscientist-toolcaller
> 💻 https://github.com/ankit25bcs10610/adaption-ai-lab
>
> #AutoScientist #AI

---

## X / Twitter (thread, optional)

1/ Most function-calling datasets only teach a model to *call* tools. Real agents fail on the opposite:
inventing a tool call when none fits, or guessing a missing argument. I trained an open model on exactly
those failure modes. @adaption_ai AutoScientist Challenge 🧵

2/ The moat is ~40% "when NOT to call" data: refuse (no tool applies), clarify (missing arg / ambiguous),
over-refusal traps (hedged-but-doable → still call), and partial-parallel (two intents → two calls). All
correct-by-construction.

3/ I ran a 2-pass adversarial audit on my own build pipeline. It caught the moat nearly shipping empty —
refuse examples 8 → 644, and 36% → 0% schema-invalid gold calls. Wrote it all up in the repo.

4/ Adaptive Data (@adaption_ai) graded the dataset **7.0 → 8.1, +15.7%, C→B**. Everything's open —
weights, dataset, eval harness, reproducibility manifest, and a second Data-Viz (Hindi chart-QA) track.

5/ Model + data 👇
🤗 https://huggingface.co/pandeyankit84/autoscientist-toolcaller
📊 https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
💻 https://github.com/ankit25bcs10610/adaption-ai-lab
#AutoScientist
---

## LinkedIn (professional)

**Teaching an AI model when *not* to call a tool.**

I built an open-source, data-centric function-calling model for the Adaption AutoScientist Challenge.
Most tool-use datasets only teach a model to call tools — but real agents break on the
opposite decision: inventing a tool call when none applies, or guessing a missing argument. That gap is
the whole edge.

So ~40% of the training set is the decisions everyone else skips — refuse, clarify, disambiguate, resist
over-refusal, and complete every call — all synthesized correct-by-construction from real tool schemas.

A few things I'm proud of:
• **Adaptive Data quality +15.7% (grade C → B)** — measured by Adaption's platform.
• A **two-pass adversarial audit** of my own build pipeline that caught the moat nearly shipping empty
  (refuse examples 8 → 644; 36% → 0% schema-invalid gold calls) — documented, with regression tests.
• Execution-verified environments, a decontamination pass, and a full reproducibility manifest.
• A second **Data-Visualization** track: a self-verifying chart-QA dataset with a Hindi/Devanagari slice.

Everything is open — weights, both datasets, the eval harness, and a live demo.

🤗 Model: https://huggingface.co/pandeyankit84/autoscientist-toolcaller
🤗 Dataset: https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
📊 Kaggle: https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset
💻 Code: https://github.com/ankit25bcs10610/adaption-ai-lab
🔗 Live site: https://autoscientist-toolcaller.vercel.app
▶️ In-browser demo: https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo

Huge thanks to **@adaption_ai / Adaption Labs** for AutoScientist and Adaptive Data. *(Tag both @adaption_ai and Adaption Labs when you post.)*
#AutoScientist #OpenSource #AI #MachineLearning #LLM #Agents

---

## Data-Visualization entry — post separately (second category)

**X / Twitter**
> Taught a small open VLM to **read charts — in English AND Hindi** 📊
>
> Built for the @adaption_ai AutoScientist Challenge: a **self-verifying** synthetic chart-QA dataset —
> every answer computed from the underlying data, so labels are correct by construction — plus a
> Devanagari/romanized slice with matched en/hi twins for a clean cross-language Δ.
>
> Thanks @adaption_ai (Adaption Labs — tag adaption-labs too).
> 🤗 https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset
> 💻 https://github.com/ankit25bcs10610/adaption-ai-lab
> #AutoScientist #AI

**LinkedIn**
> **Charts are images — so text-only models can't read them, and the gap is wide** (on CharXiv, GPT-4o
> scores ~47% on chart reasoning vs ~80% human). For the Adaption AutoScientist Challenge I built a
> data-centric chart-understanding dataset: a self-verifying synthetic generator (9 chart types, ground
> truth computed from the data), a Hindi/Devanagari + romanized slice with matched en/hi twins, and a
> text-only Vega-Lite spec-reading modality. Open on Hugging Face + Kaggle.
> Huge thanks to **@adaption_ai / Adaption Labs** for AutoScientist and Adaptive Data. *(Tag both
> @adaption_ai and Adaption Labs when you post.)*
> 🤗 https://huggingface.co/datasets/pandeyankit84/autoscientist-chartqa-dataset · 💻 https://github.com/ankit25bcs10610/adaption-ai-lab
> #AutoScientist #OpenSource #AI #MachineLearning #DataVisualization

## Posting checklist
- [x] Publish datasets to HF **and** Kaggle (both tracks) — done.
- [x] Live site + demo deployed: https://autoscientist-toolcaller.vercel.app (+ HF Space).
- [ ] Add model **weights** to the HF model repo after the AutoScientist console run.
- [ ] Fill the **held-out accuracy** line (base __% → fine-tuned __%) once the console run returns it.
- [ ] Post on X and LinkedIn (tag **@adaption_ai and adaption-labs** on both); attach a chart/demo screenshot.
- [ ] Paste both post URLs into the submission form.
