# Run-day runbook — from weights to submitted

One ordered sequence for the day the AutoScientist console run finishes. Everything up to step 0 is
already done, tested, and committed; this is the only remaining path. Detailed refs:
[`CONSOLE_STEPS.md`](CONSOLE_STEPS.md) · [`AUTOSCIENTIST_USAGE.md`](AUTOSCIENTIST_USAGE.md) ·
[`PART2_SUBMISSION.md`](PART2_SUBMISSION.md) · [`social_posts.md`](social_posts.md).

## 0. Get the weights + held-out number  ·  *(you, web console — the one gate)*
1. **adaptionlabs.ai/app** → open dataset **`d92279d3`** (cleaned/diversified) → **AUTOSCIENTIST** tab → **Launch / Train**.
2. Wait for **Job Completed**. Note the **held-out number** and the **weights** (HF id or a download).

## 0b. Depth-of-AutoScientist evidence (criterion #4)  ·  *(you, console)*
Run AutoScientist **2–3 times with different recipes** (e.g. `deduplication` on/off, `reasoning_traces`
on/off, a `preference_pairs` objective) and note each run's held-out %. Tabulate them so a judge sees you
drove the platform, not one click:
```bash
python -m autoscientist_toolcaller.recipe_ablation --out results/recipe_ablation.md   # fill from the run grades
```
The winning recipe's weights go forward to step 1.

## 1. Fill every real number  ·  *(one command)*
```bash
MODEL=<hf-id-or-local-weights> [ADAPTER=<lora-path>] SEEDS="41 42 43" bash scripts/finalize.sh
```
Produces (no hand-transcription): baseline vs fine-tuned eval · multi-seed significance (bootstrap CI +
McNemar) · gap decomposition · **by-difficulty** breakdown · **multilingual Δaccuracy(lang−en)** ·
robustness table · reliability probe · `results/report.html` · an auto-filled **`MODEL_CARD.md`** · then a
blocking **release preflight**. Copy the printed `HEADLINE.txt` line — that's your measured result.

## 2. Publish the weights (both platforms are mandatory)
```bash
python -m autoscientist_toolcaller.release preflight      --dir data/out        # must print no blockers
python -m autoscientist_toolcaller.release hf-model       --repo pandeyankit84/autoscientist-toolcaller --dir <weights>
python -m autoscientist_toolcaller.release kaggle-model   --slug pandeyankit99/autoscientist-toolcaller --dir <weights>
```
Then re-push the filled `MODEL_CARD.md` to the HF model repo. *(I can run all of this for you.)*

## 3. Update the public numbers  ·  *(measured, no longer projected)*
- Paste the `HEADLINE.txt` line into `README.md` (the "result so far" table) + flip the site's
  `benchmarks` from projected→measured (`web/lib/results.ts`, `projected = false`) and redeploy.
- Fill the held-out line in [`social_posts.md`](social_posts.md) (`base __% → fine-tuned __%`).
  *(I can do all of this once you give me the numbers.)*

## 4. Post + submit  ·  *(you)*
1. Post the X + LinkedIn drafts ([`social_posts.md`](social_posts.md)); tag **@adaption_ai** / Adaption.
2. Join the HackIndia **WhatsApp** channel + **Discord** (`#autoscient-challenge`) — eligibility.
3. Fill the Part 2 form from [`PART2_SUBMISSION.md`](PART2_SUBMISSION.md): dataset id, HF + Kaggle
   dataset URLs, **weights link**, **Training Model ID**, both post URLs, live site.

## Data-Visualization entry (second category — same shape, chart-QA)
Awards are per-category, so run the chart-QA track too:
```bash
python -m autoscientist_toolcaller.viz.build_dataset --out data/viz --n-synth 400 --n-indic 600 --n-vega 150
python -m autoscientist_toolcaller.viz.train_adaption --data data/viz/train_tab.jsonl   # upload + grade on Adaptive Data
# then on the console: AutoScientist-train the chart-QA set (base VLM Qwen3-VL-8B or gemma-3-4b, LoRA)
python -m autoscientist_toolcaller.viz.baseline --model <vlm> --data data/viz/test.jsonl   # honest 'before'
# eval the fine-tuned VLM with the hardened relaxed scorer (autoscientist_toolcaller.viz.eval_chart)
```
Publish weights to a new `pandeyankit84/autoscientist-chartqa` HF repo + a Kaggle model; fill the
Entry-B fields in [`PART2_SUBMISSION.md`](PART2_SUBMISSION.md) and post its social variant.

## Housekeeping (do now, independent of run-day)
- **Revoke** any credentials pasted in chat (GitHub PAT, HF, Adaption) — GitHub: <https://github.com/settings/tokens>.
- Optional: enable the Vercel **Analytics** + **Speed Insights** tabs; `twine upload` to PyPI ([`PUBLISHING.md`](PUBLISHING.md)).
