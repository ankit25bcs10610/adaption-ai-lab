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
pretty_name: AutoScientist Tool-Calling (with hard negatives)
size_categories:
  - 1K<n<10K
---

# AutoScientist Tool-Calling Dataset

A curated function-calling / tool-use dataset for the Adaption AutoScientist Challenge. Its distinguishing
feature is a ~28% slice of **hard negatives** — cases where the correct behavior is *not* to call a tool.

## Composition

Each row is one example (canonical format in the repo's `src/format_utils.py`):

```json
{"tools": [ {"name","description","parameters"} ],
 "query": "...",
 "answer": {"type": "tool_call"|"refuse"|"clarify", "calls": [...], "content": "..."},
 "meta": {"source": "xlam|toolace|hard_negative", "hn_kind": "no_tool|missing_arg|ambiguous"|null}}
```

Splits: `train` / `val` / `test`, plus `test_novel` (examples using tools **never seen in training**, to
measure generalization). Counts and per-source/per-kind breakdown are in `stats.json`.

Hard-negative kinds:
- **no_tool** — no available tool can satisfy the request → the model must *refuse*.
- **missing_arg** — a required argument is absent → the model must *ask* (clarify), not guess.
- **ambiguous** — two plausible tools → the model must *disambiguate* (clarify).

## Provenance & licensing

Positive examples are derived and curated from:
- `Salesforce/xlam-function-calling-60k` — **CC-BY-4.0**
- `Team-ACE/ToolACE` — **Apache-2.0**

Hard negatives are **original**, synthesized from the real tool schemas in the corpus (template-based,
seeded, reproducible). All examples are schema-validated; the set is deduplicated (MinHash + semantic) with
train/val/test leakage removed. Released under **Apache-2.0**. Attribution to xLAM and ToolACE preserved.

## Intended use

Supervised fine-tuning and preference tuning of function-calling models that must reliably decide *whether*
and *how* to call tools — including safe abstention.

## Limitations

- English-only in this version.
- Hard-negative query phrasing is template-derived (labels are guaranteed correct; wording is less varied
  than natural queries). Optionally paraphrase for fluency.

## Reproduction

Built with the pipeline in the accompanying repo: `python -m src.build_dataset --config config.yaml`.
