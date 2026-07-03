---
license: apache-2.0
task_categories:
  - image-text-to-text
  - visual-question-answering
language:
  - en
  - hi
tags:
  - chart-question-answering
  - data-visualization
  - synthetic
  - self-verifying
  - indic
  - autoscientist
pretty_name: AutoScientist Chart-QA (self-verifying + Hindi)
size_categories:
  - 1K<n<10K
---

# AutoScientist Chart-QA Dataset

A chart-understanding dataset for the Adaption AutoScientist Challenge (Data Visualization). Its
distinguishing features: answers are **correct by construction** (computed from the underlying chart
data, not human-labeled) and it includes a **Hindi/Devanagari + romanized** slice.

## Composition

Each row is one (chart image, question, answer). Splits: `train` / `val` / `test`, plus `test_novel`
(chart types held out entirely, to measure generalization). Indic en/hi twins share a `pair_id` and are
kept within the same split. Counts and per-source/type/kind/lang breakdown are in `stats.json`.

Chart types: bar, hbar, grouped_bar, stacked_bar, line, multiline, pie, scatter, area.
QA kinds: value_lookup, max, min, compare, sum, count, trend, proportion, difference, mean.
Languages: en, hi (Devanagari), hi-romanized.

## How answers are guaranteed correct

Every value drawn on the chart is produced by the *same* formatter as the gold answer, and all
ground-truth is computed with integer-quantum arithmetic (exact tie/threshold handling), unique-extremum
guards, Pearson-r for scatter correlation, and pie proportions labeled from the computed percentages.
See the repo's `autoscientist_toolcaller/viz/synth_charts.py` (invariants I1–I7).

## Provenance & licensing

Synthetic and Indic charts are original, released under **Apache-2.0** / CC-BY-4.0. Optional ReachQA
rows are MIT (attribution preserved). Devanagari rendering uses OFL Noto (or system) fonts. No scraped
or non-commercial data — clean for the mandatory Hugging Face + Kaggle release.

## Intended use

Supervised fine-tuning of vision-language models for chart reading and reasoning, including multilingual
(Hindi) chart understanding.

## Limitations

- matplotlib-styled synthetic charts (clean); complements but doesn't replace real-world chart corpora.
- Numeric answers are graded with ±5% relaxed tolerance.

## Reproduction

`python -m autoscientist_toolcaller.viz.build_dataset --out data/viz --n-synth 400 --n-indic 200 --seed 42`
