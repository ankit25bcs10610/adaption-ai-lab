# Winning strategy — research-backed

This repo now holds **two** data-centric submissions:

1. **Function-calling** ("All Other Domains") — the original pipeline (`src/`).
2. **Data Visualization** (multimodal chart understanding) — `src/viz/`, the recommended stronger,
   less-crowded, visually-native category. See `src/viz/README.md`. It adds two originality levers most
   competitors lack: a **self-verifying synthetic chart generator** (answers correct by construction) and
   a **Hindi/Devanagari + romanized chart-QA slice** that unlocks the $2k HackIndia track (matched-pair
   en/hi Δaccuracy). Eval is a hardened ChartQA relaxed-accuracy harness; the win story is closing the
   wide CharXiv gap (GPT-4o ~47% reasoning vs ~80% human).

Below is the strategy distilled from current (2026) research on what actually wins function-calling
benchmarks, and exactly how the function-calling repo implements each lever.

## The 5 things that win — and where we do them

1. **Small proven base + LoRA.** Qwen2.5-Coder-3B-Instruct (swap to 1.5B or Qwen3-1.7B is fine). The 1–3B
   band is the documented sweet spot; sub-1B genuinely fails multi-turn/parallel. → `config.yaml: base_model`.

2. **Original, self-verifying dataset.** Every positive target is schema-validated; hard negatives and
   multi-turn are synthesized from real tool schemas with labels correct by construction.
   → `src/schema_validator.py`, `src/hard_negatives.py`, `src/multiturn.py`, `src/quality_filter.py`.

3. **Aim data where the score concentrates and models are on the floor.** BFCL v4 = Agentic 40% +
   Multi-Turn 30% + Live 10% + Non-Live 10% + Hallucination 10%. So:
   - **Multi-turn (`multiturn_ratio: 0.20`)** — `miss_param`, `miss_func`, `long_context`. Small fine-tunes
     provably beat GPT-4o here (xLAM-2-8b 70.5 vs GPT-4o 51).
   - **Irrelevance/abstention** — the Hallucination bucket; `no_tool` hard negatives held near ~10% of the
     set (the swept optimum; more degrades positive calling).
   → `src/multiturn.py`, `src/hard_negatives.py`, `config.yaml` ratios.

4. **Prove it with the official harness + honest stats.** Identical greedy decoding for base vs fine-tuned,
   bootstrapped standard error, per-category breakdown, AND cross-checked against the official
   `bfcl-eval`. → `src/baseline.py`, `src/eval_bfcl.py`, `src/export_bfcl.py`.

5. **Complete, permissive open release.** Apache-2.0 weights/code, CC-BY-4.0 data, auto-filled model +
   dataset cards, seeds, error bars, GGUF quants, HF Space demo. Completeness is measurably rewarded.
   → `src/release.py`, `src/fill_model_card.py`, `dataset_card_template.md`, `src/export_gguf.py`, `app/`, `web/`.

## Data sources (permissive only)

| Source | License | Role |
|---|---|---|
| `Salesforce/xlam-function-calling-60k` | CC-BY-4.0 ⚠ re-verify | single-turn positives |
| `Team-ACE/ToolACE` | Apache-2.0 | single-turn positives |
| `Agent-Ark/Toucan-1.5M` | Apache-2.0 | large real MCP trajectories (opt-in) |
| hard negatives | original (Apache-2.0) | refuse / clarify (the moat) |
| multi-turn | original (Apache-2.0) | miss_param / miss_func / long_context |

## Optional but high-upside

- **Claude LLM-as-judge quality filter** (research's #2 data lever) — `src/claude_judge.py`. Set
  `dataset.quality_judge: claude` + `quality_keep_frac < 1.0` in `config.yaml` to drop the bottom
  fraction of positives by a Claude-scored quality rubric (uses the Anthropic Python SDK).
- **Schema-drift slice** (distinctive originality) — `src/schema_drift.py`, `schema_drift_ratio: 0.08`.
  Tools whose schemas changed under the model (param added/retyped/renamed) — a fine-tuning slice with
  no existing dataset. Teaches schema-awareness (clarify on drift, remap on rename).
- **HTML eval report** — `src/eval_report.py` renders a self-contained report: base-vs-ft table,
  per-category bars, and a call/refuse/clarify **confusion matrix** (fed by `error_analysis`'s
  `predictions.jsonl`). Judges skim in minutes; this makes the win instantly legible.
- **DPO for restraint** after SFT — `src/build_preference.py` + `src/train_dpo.py` (chosen = correct
  refuse/clarify, rejected = a plausible hallucinated call).
- **Function masking** during training (up to +51 F1 on weak bases) — apply at the training layer if the
  platform exposes it.

## Advanced upgrades (research-backed, implemented)

From a 6-stream advanced-research pass. All offline, seeded, tested (72 checks).

1. **Execution-verified tool environments** (`src/envs.py`) — the highest-leverage data work. Deterministic
   cart/calendar domains with state-diff checkers generate **execution-verified** multi-turn examples
   (correct by construction — the call is *run* and the resulting state checked) and **execution-labeled
   DPO pairs** (chosen = verified, rejected = checker-proven-wrong). This is the data-centric stand-in for
   RL-with-verifiable-rewards (the platform can't run online RL). Enable with `dataset.env_examples: 400`.
2. **Robustness-delta table** (`src/robustness_table.py`) — shows the fine-tuned model's accuracy *drop*
   under distribution shift (multi-turn / clarify / irrelevance / …) is **smaller** than the baseline's.
   BFCL authors report 11–19% drops from mere paraphrasing, so this is what makes a large hidden-baseline
   gain *believable* — the single biggest credibility lever.
3. **Statistical honesty kit** (`src/eval_stats.py`) — bootstrapped 95% CIs, a paired base-vs-ft gap CI +
   bootstrap p-value, and an exact McNemar test. Report "+X pp ± CI, p<0.05" instead of a bare number.
4. **Interactive eval dashboard** (`src/eval_report.py`) — now adds a significance banner + an SVG radar
   of per-category accuracy on top of the confusion matrix.

Bigger bets still open (need an HF token / more time): a real in-browser base-vs-fine-tuned playground
(ONNX + transformers.js), and unblocking the multimodal/Indic track by hosting chart images as an HF
Image dataset. **Trap to avoid:** don't promise online RL (GRPO/PPO) — the platform is data-centric;
express RLVR as the execution-labeled DPO pairs above.

## Uncertainties to double-check before relying on them

- xLAM-60k license can flip (NC↔BY) — confirm at use time.
- Toucan aggregate license is Apache-2.0 but verify subsets if you slice it.
- The exact Adaption baseline model + metric per category is **not public** — confirm in Discord
  `#autoscient-challenge` office hours; it lets you match `base_model` and eval to their target.

Full sourced research lives in the conversation that generated this repo.
