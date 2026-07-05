# Teaching a Small Model When *Not* to Call a Tool: A Data-Centric Function-Calling Submission for the Adaption AutoScientist Challenge

**Author:** Ankit Pandey
**Artifacts:** code <https://github.com/ankit25bcs10610/adaption-ai-lab> · dataset (HF) <https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset> · Kaggle <https://www.kaggle.com/datasets/pandeyankit99/autoscientist-toolcaller-dataset> · live demo <https://autoscientist-toolcaller.vercel.app>
**License:** Apache-2.0

> **Status note (honesty).** The dataset-quality result (§6.1) is *measured* on Adaption's Adaptive Data
> platform. The held-out *model-accuracy* improvement (§6.2) — the challenge's eligibility gate — is
> produced by the AutoScientist console training run and is reported as `__PENDING__` here; the full
> evaluation harness that fills it is built, tested, and one command away (`scripts/finalize.sh`). No
> model-accuracy number in this paper is fabricated. Convert to PDF with `pandoc docs/paper.md -o paper.pdf`.

## Abstract

Most function-calling datasets contain only examples where a tool *should* be invoked, which trains models
to call a tool even when none applies or when required arguments are missing — the dominant failure mode of
real agents. We take the opposite view: the hardest and most valuable decision in tool use is **when *not* to
call a tool**. We construct an original, **correct-by-construction** function-calling dataset (7,566 examples)
in which ~40% of the data is *structured negative supervision* — refuse (no applicable tool), clarify
(missing argument / ambiguous intent), over-refusal traps (hedged-but-satisfiable requests that must still
call), and partial-parallel (two intents → two calls) — synthesized from real tool schemas so every gold
label is verifiable. We add execution-verified multi-turn trajectories from deterministic tool environments,
a schema-drift slice, and a five-language reliability slice. A two-pass adversarial audit of our own build
pipeline caught the negative-supervision "moat" being silently discarded by deduplication (refuse examples
8 → 644) and 36% → 0% schema-invalid gold calls, the latter now enforced by a build-time drop-guard. Graded
on Adaption's Adaptive Data platform, dataset quality improved **+15.7% (grade C → B)**. We release the
dataset, an execution-verified evaluation harness (BFCL-v4-weighted, calibration/abstention metrics,
bootstrapped confidence intervals + McNemar significance), reproducibility manifests, and a live demo.

## 1. Introduction

Tool-using LLM agents fail less on *how* to format a call than on *whether* to call at all. The Berkeley
Function-Calling Leaderboard's "irrelevance" category and its multi-turn/agentic categories exist precisely
because models hallucinate calls, invent arguments, or fail to ask for missing ones. Yet the datasets that
dominate fine-tuning optimize almost exclusively for successful invocation. The Adaption AutoScientist
Challenge automates the training loop, shifting the competitive edge from compute to **data**. This
reframes the problem: the winning move is not a bigger model or more calls, but a dataset that is *correct
by construction* and aimed at the decisions everyone else omits.

**Contributions.** (1) An original function-calling dataset whose distinguishing feature is structured
negative supervision — refuse / clarify / over-refusal / partial-parallel — all label-verifiable from tool
schemas. (2) Execution-verified data: multi-turn and multi-call trajectories generated against deterministic
tool environments whose state transitions are the oracle, plus execution-labeled preference pairs (a
data-centric stand-in for RL-with-verifiable-rewards). (3) A reproducible, adversarially-audited build
pipeline with a build-time schema-validity drop-guard guaranteeing 0% invalid gold calls. (4) A rigorous,
offline-testable evaluation harness with calibration/abstention metrics and honest statistics.

## 2. Related work

**Function-calling datasets.** ToolACE (Apache-2.0) and xLAM/APIGen supply large single-turn positive
corpora; we build on permissively-licensed positives (ToolACE) and add the negative and multi-turn
supervision they lack. **Benchmarks.** BFCL v3/v4 weight multi-turn and agentic categories heavily and
include an irrelevance/abstention category; we mirror its category structure and weighting in evaluation.
**Data-centric AI.** Rather than model-centric scaling, we treat the dataset as the artifact under
optimization, consistent with the AutoScientist/Adaptive Data premise.

## 3. Dataset construction

Each example is `{tools, query, [history], answer}` where `answer` is a single JSON envelope with action
`call` / `refuse` / `clarify`. All slices are seeded (seed 42) and reproducible via
`python -m autoscientist_toolcaller.build_dataset`.

- **Positives** — curated from ToolACE; every gold call is schema-validated (`schema_validator.py`).
- **Hard negatives** (`hard_negatives.py`) — `no_tool` → refuse (held near ~9% of the total, the swept
  optimum); `missing_arg` / `ambiguous` → clarify; `over_refusal` → **must call** (counterweight to
  refusal bias); `partial_parallel` → **two calls**.
- **Multi-turn** (`multiturn.py`) — BFCL-style `miss_param`, `miss_func`, `long_context`.
- **Execution-verified environments** (`envs.py`) — deterministic cart/calendar domains whose
  state-diff checkers *run* each candidate call and verify the resulting state, yielding correct-by-
  construction multi-call trajectories and checker-proven-wrong preference pairs.
- **Agentic trajectories** (`agentic.py`) — observation-in-the-loop multi-step rollouts with error-recovery
  steps (impossible → clarify; already-satisfied → stop), plus **fault-injection** variants: a scripted
  transient tool error (503 / 429 / timeout / malformed payload) whose gold continuation is to *retry the
  same call* — failure realism per BFCL-v4's injected errors and PALADIN (arXiv:2509.25238).
- **Schema-drift** (`schema_drift.py`) — tools whose schema changed under the model (param added /
  retyped / renamed), teaching schema-awareness.
- **Multilingual** (`multilingual.py`) — matched English/Hindi/romanized twins sharing a `pair_id` for a
  clean cross-language Δ.
- **Format-invariance twins** (`format_twins.py`) — the same example with tool docs re-rendered as Python
  signatures / XML / a compact list, gold identical — targeting BFCL-v4's *format sensitivity* bucket,
  where tool-specialized fine-tunes notoriously overfit one documentation format.
- **Masked twins** (`format_twins.py`) — Hammer-style function masking (arXiv:2410.04587): neutral
  `func_i`/`arg_j` names with descriptions kept and gold renamed consistently, killing naming-convention
  shortcuts (select-by-description).

**Hygiene.** MinHash + semantic deduplication, cross-split leakage removal, and a decontamination pass
against BFCL/ToolACE-style probes protect the held-out claim. A final **drop-guard** validates every
`tool_call` gold against its Draft-7 schema (`additionalProperties:false`), guaranteeing 0 invalid golds
in the shipped artifacts (`stats.json:schema_invalid_dropped`).

**Composition (published set: 7,566 examples, 7,315 unique tools).** 7,323 across train/val/test + 243 in
a `test_novel` holdout using tool names never seen in training. Realized source shares: positives ≈ 50%,
hard-negative ≈ 17%, multi-turn ≈ 13%, schema-drift ≈ 8%, multilingual ≈ 3%, format-twins ≈ 10%, masked-twins ≈ 4%; `no_tool` ≈ 8.8% of total.

## 4. AutoScientist / Adaptive Data usage

We use the platform as a **data-centric loop** (`train_adaption.py`, reproduced offline by
`demo_platform.py`): upload → grade → improve (recipes: deduplication + reasoning traces, plus a
`brand_controls` blueprint that encodes the call/refuse/clarify discipline) → re-grade → train. Depth is
demonstrated by comparing recipe configurations (`recipe_ablation.py`) rather than a single click. The
console training leg produces the model weights and the official held-out number.

## 5. Evaluation methodology

Base and fine-tuned models are scored with **identical greedy decoding** on the same held-out split.
Metrics (`eval_harness.py`, `eval_bfcl.py`, `eval_stats.py`): overall / positive / refusal / clarify
accuracy; hallucination rate on hard negatives; a **BFCL-v4-weighted** aggregate (simple / multiple /
parallel / multi-turn / irrelevance / clarify / agentic); a **calibration/abstention** block (confusion
matrix, over-refusal rate, abstention precision/recall); a **novel-tools holdout** for generalization; and
matched-pair multilingual Δ. Significance is reported with **bootstrapped 95% CIs, a paired base-vs-fine-
tuned gap CI, and an exact McNemar test** — i.e. "+X pp ± CI, p<0.05", not a bare number. A
**robustness-delta** table reports the accuracy drop under distribution shift for base vs. fine-tuned. The
harness is offline-testable (291 checks) so its correctness is verified independently of any model run.

## 6. Results

### 6.1 Dataset-quality improvement (measured)

Adaptive Data graded the audited dataset **7.0 → 8.1, +15.7%, grade C → B** on the fixed set (`c4923b7f`,
1,000/2,440 rows under the free-tier cap); a completed 250-row reference run (`a99c0c96`) corroborates at
**+10.0%, grade B**. The audit before/after (the whole thesis): `no_tool` 8 → 644, `miss_param` 1 → 70,
`ambiguous` 0 → 167, schema-invalid gold calls 36% → 0%.

### 6.2 Held-out model improvement (pending the console run)

| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Overall accuracy | `__PENDING__` | `__PENDING__` | `__PENDING__` |
| Refusal accuracy | `__PENDING__` | `__PENDING__` | `__PENDING__` |
| Clarify accuracy | `__PENDING__` | `__PENDING__` | `__PENDING__` |
| Hallucination rate on hard negatives ↓ | `__PENDING__` | `__PENDING__` | `__PENDING__` |
| Weighted BFCL-v4 accuracy | `__PENDING__` | `__PENDING__` | `__PENDING__` |

These cells are filled by `scripts/finalize.sh` from the AutoScientist-trained weights; the eligibility
gate is the Δ column.

## 7. Limitations

The dataset-quality grade is a *data* metric, not model accuracy; we do not claim the eligibility gate on
it. Positives derive from a single permissive source (ToolACE); other sources are disabled by default for
license safety. Argument scoring uses relaxed matching. The current release is English-centric outside the
five-language reliability slice.

## 8. Future work

Online RL is out of scope for a data-centric platform; we express verifiable-reward signal as execution-
labeled preference pairs instead. Planned: broader multilingual coverage (Adaptive Data supports 242
languages), a multimodal chart-understanding track (see the companion Data-Visualization submission), and
larger real-MCP agentic trajectories.

## References

- Berkeley Function-Calling Leaderboard (BFCL v3/v4). Gorilla/UC Berkeley.
- Team-ACE, *ToolACE* (Apache-2.0).
- Salesforce, *xLAM / APIGen* function-calling data.
- Adaption Labs, *AutoScientist Challenge* and *Adaptive Data*. <https://adaptionlabs.ai/blog/autoscientist-challenge>
