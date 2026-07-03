# Data-Visualization (Chart Understanding) Track

A **multimodal** submission for the Adaption AutoScientist Challenge Part 2 category *Data Visualization
& Chart Interpretation*. Data-centric: the platform automates VLM fine-tuning, so the edge is the
**dataset**. Two originality levers most competitors won't have:

1. **A self-verifying synthetic chart generator** (`synth_charts.py`) — renders charts with matplotlib
   and produces QA whose answers are *correct by construction* (computed from the underlying data; the
   number drawn on the chart is produced by the same formatter as the gold). No labeling, no noise.
2. **A Hindi/Devanagari + romanized chart-QA slice** (`indic_charts.py`) — unlocks the $2k HackIndia
   track. PolyChartQA showed Hindi among the worst chart-QA languages and no Tamil/Telugu set exists,
   so this is genuine white space. Paired mode emits en+hi twins sharing a `pair_id` for a clean
   matched-pair Δaccuracy(hi−en).

## Why it wins

- **Wide, measurable gap.** On CharXiv (realistic charts) GPT-4o scores ~47% on reasoning vs ~80% human;
  open VLMs are far lower. A targeted fine-tune closes a chunk of that — a big, honest improvement number.
- **Visually native.** A chart is an image; text-only entrants can't compete in this category.
- **Correct-by-construction data** sidesteps the label-noise that caps most chart datasets.

## Pipeline

```
build_dataset.py   synth + Indic (+ optional ReachQA) -> dedup -> novel-chart-type holdout ->
                   group-split (en/hi twins together) -> canonical JSONL + chat JSONL + tab JSONL + stats
baseline.py        run a base VLM (Qwen3-VL-8B / Gemma-3-4B) on the test split -> honest 'before'
train_adaption.py  upload tab JSONL, run Adaptive Data multimodal (column_mapping.image + recipes +
                   chart-analyst blueprint), estimate=True first -> grade_before/after
eval_chart.py      hardened ChartQA relaxed accuracy (±5%), per chart_type / qa_kind / lang / split
```

Generate a sample:  `python -m autoscientist_toolcaller.viz.build_dataset --out data/viz --n-synth 400 --n-indic 200`

## Modules

- `format_utils.py` — canonical chart-QA example, base64 data-URI helper, chat-message builder (single
  source of truth for train + eval), fixed-point answer stringifier (never scientific notation).
- `synth_charts.py` — chart renderer (bar/hbar/grouped/stacked/line/multiline/pie/scatter/area) + hardened
  ground-truth (integer-quantum ties, unique-extremum guard, open-gap count thresholds, Pearson-r scatter,
  pie proportion labels drawn from GT, label==answer by construction). Invariants I1–I7 from the review.
- `eval_chart.py` — relaxed-accuracy scorer hardened against ~30 edge cases (Indic digits, lakh grouping,
  explicit-percent reconciliation, exact-zero gold, yes/no vs count collision, range parsing, proper
  non-greedy fuzzy list matching, garbage/range rejection).
- `indic_charts.py` — Hindi/romanized decorator: localized labels + question templates, Devanagari font
  application, ASCII numeric gold, Devanagari categorical gold, paired en/hi twins.

## Requirements

`matplotlib`, `Pillow`, `numpy` (rendering + pure logic); `datasets` (optional, ReachQA); `transformers`
+ `torch` (baseline VLM); `adaption` (training). For correct Devanagari conjuncts in released images,
render with `matplotlib[raqm]` (+ `libraqm`) on Linux; macOS system Devanagari fonts shape correctly.

## Data sources & licensing

Synthetic + Indic charts are original (release CC-BY-4.0). Optional ReachQA is MIT. Fonts (if bundled)
are OFL Noto. All permissive — clean for the mandatory HF + Kaggle open release.

## Tests

`python -m tests.viz.test_viz` — 40 offline checks (scorer edge cases, synth ground-truth correctness,
determinism, format). No model download.
