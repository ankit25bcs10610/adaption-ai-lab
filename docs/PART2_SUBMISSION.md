# Part 2 submission — answer sheet (paste when the form opens ~July 6)

Category: **All Other Domains** (function-calling / tool-use). Part 2 runs **July 6 → Aug 3**.
Fields mirror the Part 1 form. ✅ = ready to paste · ✍️ = you fill · ⏳ = needs the AutoScientist run first.

## You fill (personal / team)
- ✍️ **First name / Last name**
- ✍️ **Email**  ·  ✍️ **Phone** (+91)
- ✍️ **Job Title**  ·  ✍️ **Company Name** (student? put your college / "Independent")
- ✍️ **Street / City / State / Postal Code / Country** (real — they mail swag)
- ✍️ **Team members** (first/last/email each) + **team captain** — or just you

## Ready to paste ✅
- **Which category:** `All Other Domains`
- **Specify dataset ID:** `d92279d3-90c5-4da7-aef3-e506aa291cd6`
  *(the cleaned + diversified set — use this once it shows **Job Completed**; until then the completed
  fallback is `a99c0c96-ff5b-490a-9aa9-372ea62d79d4`)*
- **Insert final dataset:** `https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset`
- **Kaggle URL (required):** `https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset`
- **Hugging Face URL:** `https://huggingface.co/pandeyankit84/autoscientist-toolcaller`
- **Are you submitting as part of the HackIndia challenge?** `Yes`

## Needs the AutoScientist training run first ⏳ (see docs/CONSOLE_STEPS.md)
- ⏳ **Training Model ID (optional):** from the **Measure** tab after training — leave blank if not done.
- ⏳ **Insert model weights link:** publish weights first, then paste
  `https://huggingface.co/pandeyankit84/autoscientist-toolcaller` (weights added to that repo).
- ⏳ **LinkedIn + X post links:** post the drafts in `docs/social_posts.md` (tag @adaption_ai / Adaption),
  then paste both URLs. Live demo to include: `https://huggingface.co/spaces/pandeyankit84/autoscientist-toolcaller-demo`

## Before you submit
1. Run **AutoScientist training** on `d92279d3` → weights + held-out number (`docs/CONSOLE_STEPS.md`).
2. `MODEL=<weights> bash scripts/finalize.sh` → fills MODEL_CARD + report with real numbers.
3. Publish weights to HF + Kaggle (I can run `src/release.py` for you).
4. Post on LinkedIn + X; join the WhatsApp channel (necessary).
5. Paste this sheet into the Part 2 form.
