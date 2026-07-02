# AutoScientist Challenge — Function-Calling Entry

A **data-centric** submission for the Adaption AutoScientist Challenge (category: *All Other Domains* /
tool-use). The whole thesis: the Adaption platform automates the training loop, so the win comes from the
**dataset**, not the model. Our edge is a curated function-calling set that deliberately mixes in
**hard-negative examples** — cases where the correct behavior is *not* to call a tool (no applicable tool,
missing argument, or ambiguous choice). Baselines fail these constantly; training on them produces a
measurable, provable improvement.

## Why this wins

- **Weak, public baseline.** Even good 8B models score ~50% on the Berkeley Function-Calling Leaderboard,
  and its irrelevance/hallucination category is where most training data is silent.
- **Fits the platform.** Pure text in / JSON out — exactly the `prompt`/`completion` post-training Adaption
  runs on Together AI. No audio/vision pipeline.
- **The moat is data, not compute.** ~2–5k curated examples with a ~25% hard-negative slice beats a large
  noisy set (LIMA effect), and the hard negatives are original contribution judges reward.

## Pipeline

```
Core
1. build_dataset.py   download xLAM + ToolACE -> quality filter -> curate -> hard negatives ->
                      novel-tool holdout -> dedup -> split
2. dedup.py           MinHash + semantic near-dup removal + cross-split leakage check
3. baseline.py        run eval on the RAW base model -> honest "grade_before" number
4. train_adaption.py  upload to Adaption, run AutoScientist -> grade_before/after/improvement_percent
5. eval_harness.py    re-run the SAME eval on the fine-tuned model -> confirm margin
6. release.py         push weights + dataset to HF AND Kaggle
7. fill_model_card.py auto-populate MODEL_CARD.md from results/*.json
8. app/app.py         Gradio ZeroGPU demo: refusal vs. hallucination side-by-side

Strength add-ons
- eval_bfcl.py        BFCL-style category breakdown (simple/multiple/parallel/irrelevance/clarify) +
                      lenient AST-style arg matching; eval test_novel.jsonl for generalization
- quality_filter.py   heuristic (offline) or LLM-judge scoring; set dataset.quality_keep_frac < 1.0
- build_preference.py DPO pairs (chosen=correct refuse/call, rejected=hallucinated call) -> train_dpo.py
- ablation.py         subsets at 25/50/100% -> accuracy-vs-size table + plot
- error_analysis.py   group failures by category + hard-negative kind -> data-iteration loop
```

Run the core end-to-end: `bash scripts/run_all.sh`. See `dataset_card_template.md` for the HF dataset card.

## The canonical example format

Every example is stored as one JSON object (see `src/format_utils.py`):

```json
{
  "tools": [ { "name": "...", "description": "...", "parameters": { ...JSON Schema... } } ],
  "query": "user request text",
  "answer": {
    "type": "tool_call" | "refuse" | "clarify",
    "calls": [ { "name": "...", "arguments": { ... } } ],   // for tool_call
    "content": "..."                                          // for refuse/clarify
  },
  "meta": { "source": "xlam|toolace|hard_negative", "hn_kind": "no_tool|missing_arg|ambiguous|null" }
}
```

`format_utils.to_prompt_completion()` renders this into the `prompt` / `completion` columns Adaption expects,
using the base model's own chat template.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...            # huggingface
export ADAPTION_API_KEY=...    # pt_live_...  (from adaptionlabs.ai)
export KAGGLE_USERNAME=... KAGGLE_KEY=...
```

## Judging-criteria coverage

| Criterion | Where we address it |
|---|---|
| Measurable improvement | `baseline.py` + `eval_harness.py` produce a base-vs-finetuned table with std error |
| Dataset quality / originality | hard-negative generator (`hard_negatives.py`) + dedup + schema validation |
| Real-world impact | tool-use / agent reliability is a core production pain point |
| Depth of AutoScientist usage | `train_adaption.py` uses the SDK end-to-end and logs its evaluation summary |
| Open-release quality | `model_card_template.md`, pinned `requirements.txt`, seeds, repro steps, Gradio demo |

## Status / TODO before submit

- [ ] Confirm base model + exact metric with Adaption in `#autoscient-challenge` (Discord) — de-risks everything.
- [ ] Run pipeline, confirm improvement margin.
- [ ] Publish to HF + Kaggle, fill model card.
- [ ] Ship Gradio Space, post on LinkedIn/X tagging @adaption_ai.
