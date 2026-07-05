# AutoScientist console — step by step (weights + held-out number)

The SDK is dataset-only; the **model-training loop that produces weights + the official held-out number
runs in the web console** at https://adaptionlabs.ai/app. This is the one remaining eligibility action.
Free compute is provided. Do it once on the best dataset, then run one command locally.

## 0. Unblock the queue (do first)
The free tier runs ~3 jobs at once. Two stale pre-cleaning runs are hogging slots and starving the good
dataset. In **/app/datasets**, open the **Action** menu and **cancel/delete**:
- `c4923b7f…` (2,440 rows) and `3ef74056…` (2,440 rows) — old, superseded.
Keep: **`d92279d3…`** (2,557 rows — the cleaned + diversified set) and `a99c0c96…` (reference).

## 1. Let Adaptive Data finish on the good dataset
`d92279d3` should show Adaptive Data reaching **Job Completed** (grade before→after). If it stalls at
"Data Uploading" or 0%, it's queue-blocked — cancelling the two above frees it.

## 2. Run the AutoScientist training loop (produces WEIGHTS + held-out number)
1. Open `bea4a581-2ef4-44fa-b6ae-47ba5fcaf36f` (the new 6,522-row set that matches the published HF/Kaggle
   dataset; its grade is finishing server-side) → the **AUTOSCIENTIST** column/tab → **Launch / Train**.
   *(fallback: the older `d92279d3` 2,557-row set.)*
2. Objective: **instruction_dataset** (prompt → completion).
3. Column mapping: `prompt` → prompt, `completion` → completion (already set for this dataset).
4. Base model: leave AutoScientist's default (or Qwen2.5-Coder-3B if offered).
5. Launch. Free compute runs it end-to-end; wait for **Job Completed**.
6. **Record two things:** (a) the **held-out improvement %** the console reports (this is the Step-5
   eligibility number), and (b) **Download the trained weights** (adapter/checkpoint).

## 3. Second objective — preference tuning (depth + often a higher moat gain)
Run a *second* AutoScientist job to demonstrate two training modes (judging criterion #4) and often beat
SFT on the refuse/clarify moat:
1. Upload **`data/out/pref.jsonl`** (2,701 preference pairs: `{prompt, chosen, rejected}`) — or reuse the
   dataset if the console can point at it.
2. Objective: **preference_pairs**. Column mapping: `prompt`→prompt, `chosen`→chosen, `rejected`→rejected.
3. Launch → wait → record the held-out number. Keep whichever objective scores higher for the submission.

## 4. (Optional edge) Recipe ablation for "depth of AutoScientist usage"
Re-run Adaptive Data with different recipe toggles and note the grade delta per config — evidence you
*compared* recipes, not just ran one:
- `reasoning_traces` on/off · `prompt_rephrase` on · `hallucination_mitigation` on (on-thesis for
  "no hallucinated calls") · `brand_controls.length` = concise vs detailed.

## 5. Back to local — one command fills every real number
Once you have the weights (a local path or an HF repo id):
```bash
MODEL=<weights-path-or-hf-id>  bash scripts/finalize.sh          # single seed
# or, stronger:
SEEDS="41 42 43" MODEL=<weights>  bash scripts/finalize.sh       # multi-seed ± CI
```
This writes real numbers into `MODEL_CARD.md`, `results/report.html`, the decomposition + robustness
tables, and prints the **HEADLINE** (`+X pp ± CI, p<0.05, hallucination A%→B%`).

## 6. Publish the weights (I can run these for you)
```bash
python -m autoscientist_toolcaller.release hf-model     --repo pandeyankit84/autoscientist-toolcaller --dir <weights>
python -m autoscientist_toolcaller.release kaggle-model --slug pandeyankit99/autoscientist-toolcaller --dir <weights>
```
`release preflight` blocks publishing until the card has real numbers (no placeholders).

## 7. Finish the submission
- Paste the HEADLINE line as the first line of the LinkedIn/X posts (`docs/social_posts.md`).
- Post on LinkedIn + X (tag **@adaption_ai** and **adaption-labs**), pick **All Other Domains**, submit
  the Part-2 form. *(No WhatsApp/Discord join is required for the AutoScientist Challenge.)*

**Send me the weights (or the HF repo id) and I'll run steps 5–6 for you.**
