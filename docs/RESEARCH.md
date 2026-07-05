# Winning strategy — deep research (2026)

*Grounding: verified against the live repo state (`autoscientist_toolcaller/*.py`) and fact-checked per-facet research. Every external claim cites a real, resolvable source. No fabricated benchmarks. Committed as `docs/RESEARCH.md`; also the work-order for the next implementation round.*

---

## 1. Executive verdict

On raw BFCL-v4 model accuracy this submission is **not** near the frontier and must not claim to be: a Qwen2.5-Coder-3B fine-tune lands below the ~0.50–0.66 sub-10B band now set by Qwen3.5-4B/9B, and far below the 0.729–0.750 leaders (Qwen3.7 Max, Qwen3.5-397B) on the [BFCL v4 aggregator board](https://llm-stats.com/benchmarks/bfcl-v4). Where it is genuinely defensible — and where the frontier under-serves — is the **when-NOT-to-call / abstention axis**: structured negative supervision, execution-verified correctness, and honest calibration reporting with CIs + McNemar, an axis that [Hammer](https://arxiv.org/html/2410.04587v2) and [AbstentionBench](https://arxiv.org/abs/2506.09038) prove is both unsolved and actively degraded by naive positive-only / reasoning tuning. The 2–3 highest-leverage moves left are all offline and verified-absent from the current code: (1) a **format-invariance twin slice** (targets ~1/3 of BFCL-v4's 40% agentic bucket, the exact brittleness that sinks tool-specialized models); (2) a **v4-aligned eval realignment** (the current `BFCL_WEIGHTS` is v3-shaped and measures a benchmark that no longer exists); (3) a **hybrid tool retriever** (TF-IDF-only today; the field reports 2–3× selection gains from dense+lexical fusion the repo already has the embeddings for). These convert "comprehensive v3-era abstention recipe" into "v4-aware abstention + robustness recipe."

---

## 2. SOTA positioning — lead / match / lag

| Facet | Standing | Evidence & honest read |
|---|---|---|
| **Function-calling data (negatives/abstention)** | **LEAD** | Correct-by-construction refuse/clarify/over-refusal/partial-parallel + execution-verified DPO is exactly the axis Toucan ([1.5M traj](https://arxiv.org/abs/2510.01179)), APIGen-MT/xLAM-2, ToolACE explicitly *exclude* (they optimize successful invocation; Toucan dropped servers needing keys and "can't guarantee refusal cases"). [Hammer](https://arxiv.org/html/2410.04587v2) proves positive-only tuning inversely degrades irrelevance detection — direct validation of the bet. |
| **Function-calling data (positive scale)** | **LAG (by design)** | 6,522 ex vs 2–3 orders of magnitude more at the frontier. This is *fine*: the [Microsoft SLM study](https://medium.com/data-science-at-microsoft/optimizing-function-calling-with-small-language-models-data-quality-quantity-and-practical-353be49b7a00) and an ACM survey show BFCL fine-tuning **plateaus ~400 samples** and excess noisy data *degrades* small models. Volume is not the lever; composition/difficulty is. |
| **Agentic / multi-turn** | **MATCH on structure, LAG on failure realism** | Observation-in-the-loop trajectories + recovery-to-clarify exist, but from a **deterministic clean-observation oracle** (`envs.py` returns only `"ok"` or a logical error). No injected transient faults (503/429/timeout/malformed), which [BFCL v4 web-search injects as 6 real errors](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html) and [PALADIN](https://arxiv.org/pdf/2509.25238)/[Failure-Makes-the-Agent-Stronger](https://arxiv.org/html/2509.18847v2) treat as central. `eval_agentic.py` runs each rollout **once at temp 0** — no pass^k reliability metric ([tau-bench](https://arxiv.org/abs/2406.12045)'s headline contribution). |
| **Multilingual** | **LEAD (rare)** | 5-language matched twins (en/hi/hi-rom/es/fr), post-dedup, leak-free, with a matched-pair Δaccuracy signal — almost no challenge entry uses Adaption's 242-language feature. Table-stakes risk: cross-lingual **leakage** — n-gram+cosine decontam is [provably bypassed by translation](https://arxiv.org/abs/2311.04850). |
| **Chart-QA (track 2)** | **MATCH / niche-LEAD** | Self-verifying synthetic chart-QA with a Devanagari/romanized-Hindi slice is a genuine niche; not a frontier contest, so it reads as breadth, not a headline claim. |
| **Eval rigor** | **LEAD on stats, LAG on v4 shape** | Bootstrap 95% CI + paired McNemar + robustness-delta + novel-tools holdout exceed most entries. But the headline `BFCL_WEIGHTS = {multi_turn 0.20, agentic 0.20, …}` is **v3-shaped**: it has no web-search / memory / format-sensitivity slice and does not match the confirmed v4 **40% Agentic / 30% Multi-Turn / 10% Live / 10% Non-Live / 10% Hallucination** ([Gorilla BFCL v4 blog](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)). Abstention is a **single point** (precision/recall), not the swept operating-point curve [AbstentionBench](https://arxiv.org/abs/2506.09038) motivates. |

**Table-stakes vs novel.** Table-stakes now: irrelevance/refuse supervision, dedup/decontam, CIs, a BFCL-weighted number. Genuinely novel for a solo entry: format-invariance twins, real-model-error-mined negatives, graded-noise robustness curves, and honest v4-aligned + swept-threshold calibration. The strategy leans entirely on the second list.

---

## 3. Ranked new improvements to IMPLEMENT (offline, no GPU/console)

Ranked by impact/effort. Every item verified **absent** from the current code and **SUPPORTED** by a real source. Where a lever's final *number* needs the console model, the offline artifact is built now and the plot point is marked target-until-run.

### #1 — Format-invariance twin slice `format_twins.py` *(impact: high / effort: medium)*
- **Build:** New module `autoscientist_toolcaller/format_twins.py`. For a subset of gold call/refuse/clarify examples, emit matched twins sharing a `pair_id`: tool docs rendered as `{python, json, xml}` (`function_doc_format`) × expected output as `{python-call, json, verbose_xml, concise_xml}` (`return_format`) × `tool_call_tag {on, off}`; **gold semantics identical**. Reuse the existing schema validator and the matched-twin machinery from `multilingual.py`. Add a `format_delta` metric to `eval_bfcl.py` (accuracy spread across formats).
- **Basis:** BFCL v4 Format Sensitivity is part of the 40% agentic bucket; the SAME model varies wildly by I/O format, and tool-specialized fine-tunes catastrophically overfit ONE format (watt-tool-70B outputs Python when asked JSON; CoALM-70B collapses to ~0 with tool-call tags) — [Gorilla BFCL v4 blog](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html). The repo ships a single JSON envelope (`format_utils.render_tools` → `json.dumps`) and is at direct risk.
- **Effect:** Targets ~1/3 of the 40% agentic weight; a base-vs-fine-tuned `format_delta` is a v4-relevant robustness claim reviewers can check. **Highest ROI single lever.**

### #2 — Realign the eval to confirmed v4 composition `eval_bfcl.py` *(impact: high / effort: medium)*
- **Build:** Replace `BFCL_WEIGHTS` with two-level aggregation matching v4 (percentage weights *between* main categories 40/30/10/10/10; equal-weight *within* agentic; test-count weight *within* live). Map existing slices into the v4 tree, add a `format_sensitivity` sub-bucket fed by #1, and **label `web_search`/`memory` as not-covered** (user-run) rather than silently absent.
- **Basis:** Current weights `{multi_turn 0.20, agentic 0.20, irrelevance 0.12,…}` are v3-shaped and mismatch the confirmed formula — [Gorilla BFCL v4 blog](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html). Verified in `eval_bfcl.py:179` (the code even comments "the real leaderboard weights differ").
- **Effect:** Makes the headline number *defensible and honest*. Pure offline eval code. Pairs with #1 (the format bucket needs the twin slice to be non-empty).

### #3 — Hybrid tool retriever `tool_retrieval.py` *(impact: high / effort: medium)*
- **Build:** Upgrade `ToolRetriever` (currently TF-IDF only, verified `tool_retrieval.py:31`) to weighted fusion of BM25-style lexical + **model2vec cosine** (reuse `dedup._embed`, already in the repo but unused for retrieval), with field weighting over name/description/params. Re-run `eval_tool_selection` recall@k to report the delta.
- **Basis:** Retrieval is a 2–3× selection lever: [RAG-MCP](https://arxiv.org/abs/2505.03275) 13.6%→43.1% + >50% fewer prompt tokens; [Redis](https://redis.io/blog/from-reasoning-to-retrieval-solving-the-mcp-tool-overload-problem) 42%→85%; [Tool-to-Agent Retrieval](https://arxiv.org/abs/2511.01854) Recall@5 +19.4% avg (best per-model 0.66→0.85). Pure semantic alone is noisy; hybrid dense+lexical+metadata is the fix.
- **Effect:** Moves an already-evaluated slice (recall@k) with **no training** — the delta is fully offline. Strongest *no-console-dependency* ROI.

### #4 — Error→reflect→corrected-call recovery slice + fault injection `agentic.py` + `envs.py` *(impact: high / effort: medium)*
- **Build:** (a) Add a stochastic fault layer to `envs.py` returning realistic observations (`503/429/403/ConnectTimeout/ReadTimeout/ConnectionError` + malformed JSON) — `envs.py.apply` already returns `(state, ok, reason)`, so the failure cause is known. (b) In `agentic.py`, synthesize correct-by-construction **error → `<reflect>` diagnosis → corrected `<call>` → suffix** trajectories using 4 perturbation ops (call-order swap, redundant call, missing call, argument error), teaching retry→switch→abstain. Current recovery path handles only ONE case (`agentic.py:75` recovery = final step impossible → clarify).
- **Basis:** [Failure Makes the Agent Stronger](https://arxiv.org/html/2509.18847v2) lifts Qwen2.5-7B on BFCL v3 multi-turn 11.0→14.88 (+35%), Miss_Param +50% via exactly this structure; [PALADIN](https://arxiv.org/pdf/2509.25238) truncate-at-first-failure + repair; BFCL v4 web-search injects the 6 errors above ([Gorilla blog](https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html)). *Note:* the RL variant [Fission-GRPO](https://arxiv.org/abs/2601.15625) is a console/training method — cite as motivation, not a deliverable.
- **Effect:** Targets the 30% multi-turn + 40% agentic weight; extends the abstention thesis into the failure regime the frontier now emphasizes.

### #5 — Model-error-mined hard-negative slice `hard_negatives.py` (+ `error_analysis.py`) *(impact: high / effort: medium; partial console dependency)*
- **Build:** A harness that takes base-Qwen2.5-Coder-3B completions on held-out prompts, diffs each vs gold, buckets into the FC-RewardBench 8-way taxonomy, obfuscates fn/param names to random strings, and emits DPO pairs (chosen=gold, rejected=real error) tagged by error type. **Offline pipeline built now; generating completions is the console step (§4).**
- **Basis:** [ToolRM](https://arxiv.org/html/2509.11963): mined real errors are the highest-value negatives; dominant failure is **Incorrect-Parameter-Value on the correct tool** (650) and Incorrect-Fn-Name (403), not missing-arg (45 req) — a slice current template negatives barely cover. Best-of-n lifted Qwen3-0.6B 39.5%→64.4%.
- **Effect:** Targets the documented dominant 3B failure mode. High impact; flagged partial-console.

### #6 — K-sample answer-consistency filter `quality_filter.py` *(impact: medium-high / effort: medium; partial console dependency)*
- **Build:** New `score_fn` for the existing `filter_examples(keep_frac, score_fn)` (verified `quality_filter.py:100`): canonicalize each of K completions (reuse `build_preference._canon`), keep the example only if the **plurality call matches gold**. Ship the filter + a consistency report now; wire K completions on run day.
- **Basis:** [CoT-Self-Instruct](https://arxiv.org/html/2507.23751v2): answer-consistency beats self-consistency (55.1→57.2 avg) and 2,926 filtered rows beat 5,000 unfiltered — "better to have less high-quality synthetic data." Stronger and cheaper than the current optional 1-shot judge. *Caveat:* the paper's benchmarks are math/reasoning; transfer to tool-call plurality is reasonable but untested.
- **Effect:** Cleaner training set, higher held-out delta per row.

### #7 — Graded multi-level noise + robustness-vs-noise curve `schema_drift.py` + `robustness_table.py` *(impact: medium / effort: medium)*
- **Build:** Extend `schema_drift.py` (currently discrete kinds: rename/add_required/retype_enum) into graded perturbation **levels** {slight/medium/heavy/union} (typo/synonym tool names, reordered/renamed params, distractor injection), labels correct-by-construction. Plot held-out accuracy vs noise level in `robustness_table.py`.
- **Basis:** [RoTBench](https://arxiv.org/html/2401.08326v2): GPT-4 drops 80.0→58.1 Clean→Union with no change in human accuracy; RoTTuning's diverse-env training gave +16.1 avg. Turns one schema-drift number into a defensible curve. *(PA-Tool's +17pts sub-claim unverified — do not cite.)*
- **Effect:** Stronger robustness claim than a single number. Curve points need eval runs (offline eval of the trained model, or console if retraining).

### #8 — Swept-threshold abstention F1 curve + answerable/unanswerable pairs `eval_report.py` *(impact: medium / effort: medium)*
- **Build:** Sweep the refuse/clarify decision threshold (or an abstention-margin proxy) to plot over-refusal-vs-miss and report abstention-F1 at the optimum; add matched answerable/unanswerable pairs (does the model flip correctly when the enabling tool is removed?).
- **Basis:** [AbstentionBench](https://arxiv.org/abs/2506.09038): abstention is unsolved, scale barely helps, and reasoning-tuning *hurts* it — over/under-refusal is a trade-off, not a point. Repo reports single-point precision/recall only.
- **Effect:** Elevates the calibration story from a number to a curve. Cite the "reasoning-tuning hurts abstention" finding as motivation.

### #9 — LLM-decontaminator pass (paraphrase + cross-lingual) `decontaminate.py` *(impact: medium / effort: medium)*
- **Build:** Two-stage embedding-retrieve → strong-LLM "too close?" judgment over cached BFCL-v4/ToolACE probe text, run **cross-lingually** against the multilingual twins; record dropped counts.
- **Basis:** n-gram+cosine vs ~70 decoy probes is [provably bypassed by rephrase/translation](https://arxiv.org/abs/2311.04850) — a live risk because this repo's own multilingual twins can introduce translated near-duplicates.
- **Effect:** Strengthens the "held-out gain is not leakage" claim the paper rests on.

### #10 — Function-masking / name-perturbation augmentation `hard_negatives.py` *(impact: medium / effort: medium)*
- **Build:** For a subset, replace tool/param names with neutral tokens (`func_0`, `arg_1`) keeping descriptions; add twins where a decoy tool's NAME lexically matches the query but its DESCRIPTION does not (gold = pick-by-description or refuse).
- **Basis:** [Hammer](https://arxiv.org/html/2410.04587v2): "function masking" kills naming-convention shortcuts and improves both selection and irrelevance detection. Composes with the schema-drift slice.
- **Effect:** Anti-shortcut hardening of the abstention thesis. Solid but incremental next to #1–#5.

### #11 — Self-conditioning diagnostic `eval_agentic.py` *(impact: medium / effort: low)*
- **Build:** Replay each agentic trajectory with a controlled fraction of deliberately-wrong prior assistant turns (env oracle knows gold) and measure next-step accuracy drop. Cheap, novel diagnostic.
- **Basis:** [Illusion of Diminishing Returns](https://arxiv.org/html/2509.09677v3): per-step accuracy degrades as a model's own prior errors accumulate — distinct from long-context limits; scale doesn't fix it.
- **Effect:** Surfaces a publishable failure mode no current metric captures. Diagnostic, not a training lever.

### #12 — pass^k reliability metric `eval_agentic.py` *(impact: medium / effort: low)*
- **Build:** Sample k rollouts per trajectory with the existing env oracle; report the unbiased pass^k estimator (success on ALL k trials) alongside the single success rate. `eval_agentic.rollout` already supports `n_samples` — extend the aggregation, don't rebuild.
- **Basis:** [tau-bench](https://arxiv.org/abs/2406.12045): pass^k captures consistency/variance a single greedy run hides (GPT-4o pass^1 <50%, pass^8 <25% retail). *Caveat:* needs k stochastic completions from the console model, so the metric plumbing is offline but the number is run-day.
- **Effect:** Large credibility upgrade for near-zero code cost.

### Positioning / paper update `docs/paper.md` *(impact: medium / effort: low — do alongside)*
Add a one-page positioning table vs the 2025–26 frontier (Toucan, APIGen-MT/xLAM-2, ToolACE-R, Hammer, AbstentionBench) stating exactly what this recipe adds that each lacks, plus the confirmed v4 40/30/10/10/10 weighting. Reviewers will check these. **Correct two nits before citing:** AbstentionBench is arXiv:2506.09038, *not* "NeurIPS 2025" — drop the venue; the #1-vs-top-open-weight gap is ~2.1 pts (0.750 vs 0.729), so say "~2 points," not "3–4."

### Marginal — do NOT prioritize
- **Dual-preference calibration-neutrality delta:** over-refusal contrasts (`build_preference`) and abstention/over-refusal metrics already exist; only *joining* them into one reported neutrality delta is new. Marginal — a small `eval_report.py` addition at most, not a headline.

---

## 4. User-only actions that most raise win odds (console/GPU)

These gate eligibility (§6.2 held-out accuracy is PENDING) and unlock the offline pipelines above.

1. **Run the fine-tune with the three new slices enabled** — format twins (#1) + error-recovery (#4) + function-masking (#10) in the recipe. This is what converts the offline data work into the one number that gates eligibility, plus a v4-relevant robustness claim.
2. **Dump base-model completions on the held-out pool** — a single console step ("sample K greedy+temperature completions on held-out prompts → JSONL") unlocks **both** #5 (error mining) and #6 (consistency filter) and the #12 pass^k number. Highest-leverage one-liner.
3. **Objective:** keep SFT + execution-labeled DPO; the frontier evidence ([Reasoning Trap](https://arxiv.org/html/2510.22977v1)) shows abstention DPO can silently erode call-utility (0.45→0.34) — so report a **paired restraint/utility delta** on the same run, not abstention gain alone.
4. **Base model:** stay on Qwen2.5-Coder-3B for the headline (data-recipe story), but the [Microsoft SLM result](https://medium.com/data-science-at-microsoft/optimizing-function-calling-with-small-language-models-data-quality-quantity-and-practical-353be49b7a00) shows FC ability scales cleanly 4B→7B after LoRA — if a second run is cheap, a 7B ablation strengthens "recipe transfers."
5. **`finalize.sh` should emit:** (a) the v4-aligned weighted aggregate (#2), (b) base-vs-fine-tuned **format-Δ** (#1), (c) the abstention operating-point curve (#8). These three make the report v4-shaped.
6. **Two-category strategy:** keep function-calling as the headline (differentiated on abstention) and chart-QA as breadth; do **not** claim leaderboard parity — claim "best-in-class data recipe for the abstention/when-not-to-call axis the frontier under-serves."
7. **Live web-search / memory execution eval** (real v4 agentic buckets) needs live tools/backends — a console action, out of scope offline; flag as not-covered rather than faking it.

---

## 5. Sources

- BFCL v4 composition, format sensitivity, injected errors — https://gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html
- BFCL v4 leaderboard (aggregator) — https://llm-stats.com/benchmarks/bfcl-v4
- Hammer (irrelevance inversion + function masking) — https://arxiv.org/html/2410.04587v2
- AbstentionBench (arXiv:2506.09038) — https://arxiv.org/abs/2506.09038
- ToolRM / FC-RewardBench (error mining + reward selection) — https://arxiv.org/html/2509.11963
- CoT-Self-Instruct (answer-consistency filter) — https://arxiv.org/html/2507.23751v2
- The Reasoning Trap (abstention DPO erodes utility) — https://arxiv.org/html/2510.22977v1
- RoTBench (graded noise robustness) — https://arxiv.org/html/2401.08326v2
- Failure Makes the Agent Stronger (error→reflect→correct) — https://arxiv.org/html/2509.18847v2
- PALADIN (self-correcting tool-failure agents) — https://arxiv.org/pdf/2509.25238
- Fission-GRPO (RL recovery — motivation only) — https://arxiv.org/abs/2601.15625
- Illusion of Diminishing Returns (self-conditioning) — https://arxiv.org/html/2509.09677v3
- tau-bench (pass^k reliability) — https://arxiv.org/abs/2406.12045
- Toucan (1.5M real-MCP trajectories) — https://arxiv.org/abs/2510.01179
- LLM-Decontaminator (paraphrase/translation leakage) — https://arxiv.org/abs/2311.04850
- RAG-MCP (tool retrieval) — https://arxiv.org/abs/2505.03275
- Redis MCP tool-overload (retrieval 42%→85%) — https://redis.io/blog/from-reasoning-to-retrieval-solving-the-mcp-tool-overload-problem
- Tool-to-Agent Retrieval — https://arxiv.org/abs/2511.01854
- Microsoft SLM function-calling (data quality > quantity) — https://medium.com/data-science-at-microsoft/optimizing-function-calling-with-small-language-models-data-quality-quantity-and-practical-353be49b7a00

*Verification notes: `BFCL_WEIGHTS` v3-shape confirmed at `eval_bfcl.py:179`; TF-IDF-only retriever at `tool_retrieval.py:31`; no format twins, pass^k, consistency filter, error-mining, or fault injection present in the tree (grep-verified). AbstentionBench venue and the leaderboard gap figure corrected per fact-check.*
