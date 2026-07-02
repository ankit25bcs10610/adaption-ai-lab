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

> **Adaption dataset IDs:** `a99c0c96-ff5b-490a-9aa9-372ea62d79d4` (completed: quality **8.0 → 8.8,
> +10.0%**) · `4bee4b34-fd6b-4343-ae68-f0175fb96ce5` (1,949-row, dedup + reasoning_traces).

## Composition

Each row is one example (canonical format in `src/format_utils.py`):

```json
{"tools": [ {"name","description","parameters"} ],
 "query": "...",
 "answer": {"type": "tool_call"|"refuse"|"clarify", "calls": [...], "content": "..."},
 "meta": {"source": "toolace|hard_negative|multiturn|schema_drift", "hn_kind": "...", "mt_kind": "...", "sd_kind": "..."}}
```

Splits: `train` / `val` / `test`, plus `test_novel` (examples using tools **never seen in training** — a
generalization test). **~2,935 examples** total after dedup. Per-source/kind counts and a `mix` block
(intended-vs-realized shares + a `mix_ok` guard) are in `stats.json`.

Slices (counts across all splits):
- **positives** — real tool-call examples curated from ToolACE (schema-validated). ~57.6% of the set.
- **hard negatives** (~18.9%) — `no_tool` → refuse (**239**); `missing_arg` → clarify (**183**);
  `ambiguous` → clarify (**133**). `no_tool` sits at **8.1% of the total set** (research optimum ~10%).
- **multi-turn** (~14.3%) — `miss_param` (**36**), `miss_func` (**192**), `long_context` (**193**), BFCL v3/v4 style.
- **schema-drift** (~9.1%) — a tool's schema changed under the model: `add_required` (**123**),
  `retype_enum` (**59**), `rename` (**86**).

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

`python -m src.build_dataset --config config.yaml`  (repo: https://github.com/ankit25bcs10610/adaption-ai-lab)
