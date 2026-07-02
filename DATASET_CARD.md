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
generalization test). ~2,294 examples total after dedup. Per-source/kind counts in `stats.json`.

Slices:
- **positives** — real tool-call examples curated from ToolACE (schema-validated).
- **hard negatives** — `no_tool` → refuse; `missing_arg` → clarify; `ambiguous` → clarify.
- **multi-turn** — `miss_param`, `miss_func`, `long_context` (BFCL v3/v4 style).
- **schema-drift** — a tool's schema changed under the model (`add_required` / `retype_enum` / `rename`).

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
