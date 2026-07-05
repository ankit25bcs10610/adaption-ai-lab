---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - function-calling
  - tool-use
  - agents
  - hard-negatives
  - autoscientist
pretty_name: AutoScientist Tool-Calling (with hard negatives, multi-turn & schema-drift)
size_categories:
  - 1K<n<10K
---

# AutoScientist Tool-Calling Dataset

A curated function-calling / tool-use dataset for the Adaption AutoScientist Challenge. Its
distinguishing feature is a large slice of **hard negatives** and reliability-focused cases — where the
correct behavior is *not* a plain tool call.

> **Adaptive Data quality (real):** on the **fixed** set (`c4923b7f…`, graded on 1,000 of 2,440 rows
> under the free-tier cap) the platform reports **7.0 → 8.1, +15.7%, grade C → B**; an earlier 250-row
> run (`a99c0c96…`, **completed**) scored **8.0 → 8.8, +10.0%, grade B**. This is the platform's
> dataset-quality grade — the data-centric "measurable improvement." Full dataset lineage (incl. the
> cleaned `d92279d3…` set staged for submission) is in [`docs/AUTOSCIENTIST_USAGE.md`](docs/AUTOSCIENTIST_USAGE.md).

## Composition

Each row is one example (canonical format in `autoscientist_toolcaller/format_utils.py`):

```json
{"tools": [ {"name","description","parameters"} ],
 "query": "...",
 "answer": {"type": "tool_call"|"refuse"|"clarify", "calls": [...], "content": "..."},
 "meta": {"source": "toolace|hard_negative|multiturn|schema_drift", "hn_kind": "...", "mt_kind": "...", "sd_kind": "..."}}
```

Splits: `train` / `val` / `test`, plus `test_novel` (examples using tools **never seen in training** — a
generalization test). **3,346 examples** in the published set (HF + Kaggle) — 3,234 across
`train`/`val`/`test` + 112 in the `test_novel` holdout; platform-graded snapshots were smaller (see the
lineage table in `docs/AUTOSCIENTIST_USAGE.md`). Every `tool_call` gold is schema-validated at build by a
correct-by-construction drop-guard (**0 schema-invalid golds shipped**; see `stats.json:schema_invalid_dropped`).
Per-source/kind counts and a `mix` block (intended-vs-realized shares + a `mix_ok` guard) are in `stats.json`.

Slices (counts across all splits):
- **positives** — real tool-call examples curated from ToolACE (schema-validated), plus a small slice
  of **execution-verified** examples from deterministic tool environments (`autoscientist_toolcaller/envs.py`).
- **hard negatives** — `no_tool` → refuse (**284**, held at ~**8.8% of the total set**, the research
  optimum); `missing_arg` / `ambiguous` → clarify; `over_refusal` → **must call** (hedged-but-satisfiable
  requests, the counterweight to refusal bias); `partial_parallel` → **two calls** (call completeness).
- **multi-turn** — `miss_param`, `miss_func`, `long_context` (BFCL v3/v4 style), plus verified
  **multi-call** env trajectories (2–3 order-independent calls).
- **schema-drift** — a tool's schema changed under the model: `add_required` / `retype_enum` (→ clarify)
  and `rename` (→ remap to the new field). Every drift gold is validated against its drifted schema.

Exact per-kind counts and realized-vs-intended shares are in `stats.json` (`mix` block). A companion
**preference set** (`pref.jsonl`) includes execution-labeled DPO pairs (chosen = verified call,
rejected = checker-proven-wrong).

**Decontamination:** every training query is checked (n-gram + embedding) against public
BFCL/ToolACE-style probes and dropped on overlap; a `contamination` block is recorded in `stats.json`.

**Platform-enhanced variant.** Adaptive Data (data-centric) rewrites each row into a richer
`enhanced_prompt` / `enhanced_completion` — in the completed run it revised **100%** of completions
(e.g. flagging *both* an invalid enum *and* a missing required argument where the original flagged only
one). Fetch it with `python -m autoscientist_toolcaller.fetch_improved --dataset-id <id>` to get a clean enhanced
prompt/completion file to train on; this is the artifact behind the platform's quality-grade gain.

> **Data-quality audit.** An adversarial audit of the build pipeline caught the refuse/clarify moat
> being generated and then silently discarded by query-only dedup (`no_tool` had collapsed to **8**
> rows, `miss_param` to **1**). The generators, the slice-mixing math, and dedup were fixed, and
> regression-tested; the counts above are the post-fix set. Full write-up:
> [`docs/DATA_QUALITY_AUDIT.md`](https://github.com/ankit25bcs10610/adaption-ai-lab/blob/main/docs/DATA_QUALITY_AUDIT.md).

## Provenance & licensing

Positives are derived and curated from **`Team-ACE/ToolACE`** (Apache-2.0). All hard-negative,
multi-turn, and schema-drift examples are **original**, synthesized from the real tool schemas in the
corpus (template-based, seeded, reproducible). Every example is schema-validated; the set is
deduplicated (MinHash + semantic) with train/val/test leakage removed. Released under **Apache-2.0**.

## Intended use

Supervised fine-tuning (and preference tuning) of function-calling models that must reliably decide
*whether* and *how* to call tools — including safe abstention.

## Limitations

- English-only in this version.
- Synthetic hard-negative/multi-turn phrasing is template-derived (labels correct by construction;
  wording is less varied than fully natural queries).

## Reproduction

`python -m autoscientist_toolcaller.build_dataset --config config.yaml`  (repo: https://github.com/ankit25bcs10610/adaption-ai-lab)
