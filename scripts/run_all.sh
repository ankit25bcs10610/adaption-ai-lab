#!/usr/bin/env bash
# End-to-end pipeline for the function-calling track — one command, deterministic order.
#
# Offline stages (data build, preference, stats, report, manifest, preflight) always run.
# Model stages (baseline, eval, error-analysis, reliability probe) run only when $MODEL is set —
# they need weights (a GPU box or an OpenAI-compatible endpoint), so they no-op on a laptop.
#
# Usage:
#   bash scripts/run_all.sh                      # offline build + (platform) train + offline reports
#   MODEL=path/to/ft ADAPTER=path/to/lora bash scripts/run_all.sh   # + baseline/eval/probe
#   SEEDS="41 42 43" MODEL=... bash scripts/run_all.sh              # multi-seed eval aggregation
#   SKIP_TRAIN=1 bash scripts/run_all.sh         # skip the Adaption platform run
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
RESULTS="results"; DATA="data/out"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SEEDS="${SEEDS:-42}"
mkdir -p "$RESULTS"

echo "==> [1/9] Build dataset (curate, dedup, decontaminate, split)"
python3 -m src.build_dataset --config config.yaml

echo "==> [2/9] Build preference pairs (+ execution-labeled env DPO)"
python3 -m src.build_preference --config config.yaml --split train

if [ -z "${SKIP_TRAIN:-}" ] && [ -n "${ADAPTION_API_KEY:-}" ]; then
  echo "==> [3/9] Adaption AutoScientist run"
  python3 -m src.train_adaption --config config.yaml || echo "   (train_adaption failed; retry later)"
else
  echo "==> [3/9] Adaption run SKIPPED (set ADAPTION_API_KEY and unset SKIP_TRAIN to enable)"
fi

if [ -n "${MODEL:-}" ]; then
  ADAPTER_ARG=""; [ -n "${ADAPTER:-}" ] && ADAPTER_ARG="--adapter ${ADAPTER}"

  echo "==> [4/9] Baseline (untuned $BASE_MODEL) — honest 'before'"
  python3 -m src.baseline --config config.yaml --model "$BASE_MODEL" --out "$RESULTS/baseline.json" || true
  python3 -m src.error_analysis --model "$BASE_MODEL" --data "$DATA/test.jsonl" --out-dir "$RESULTS/base" || true

  echo "==> [5/9] Multi-seed eval of the fine-tuned model (seeds: $SEEDS)"
  PAIR_ARGS=""
  for s in $SEEDS; do
    echo "   -- seed $s"
    python3 -m src.eval_bfcl --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" --out "$RESULTS/eval_bfcl_$s.json" || true
    python3 -m src.error_analysis --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" --out-dir "$RESULTS/ft_$s" || true
    PAIR_ARGS="$PAIR_ARGS --pair $RESULTS/base/predictions.jsonl,$RESULTS/ft_$s/predictions.jsonl"
  done

  echo "==> [6/9] Statistics: paired gap + decomposition + robustness (multi-seed)"
  python3 -m src.eval_stats --base "$RESULTS/base/predictions.jsonl" --finetuned "$RESULTS/ft_42/predictions.jsonl" --out "$RESULTS/eval_stats.json" || true
  python3 -m src.eval_decompose --base "$RESULTS/base/predictions.jsonl" --finetuned "$RESULTS/ft_42/predictions.jsonl" $PAIR_ARGS --out "$RESULTS/eval_decompose.json" || true
  python3 -m src.robustness_table --base "$RESULTS/base/predictions.jsonl" --finetuned "$RESULTS/ft_42/predictions.jsonl" --out "$RESULTS/robustness.md" || true

  echo "==> [7/9] Reliability probe"
  python3 -m src.reliability_probe --model "$MODEL" $ADAPTER_ARG --out "$RESULTS/reliability_probe.md" || true
else
  echo "==> [4-7/9] Model eval SKIPPED (set MODEL=<hf-id-or-path> to run baseline/eval/probe)"
fi

echo "==> [8/9] HTML report + BFCL export"
python3 -m src.eval_report --out "$RESULTS/report.html" || true
python3 -m src.export_bfcl --data "$DATA/test.jsonl" --out-dir "$RESULTS/bfcl" || true

echo "==> [9/9] Reproducibility manifest + release preflight"
python3 -m src.manifest --out-dir "$DATA" --config config.yaml --manifest "$RESULTS/manifest.json"
python3 -m src.release preflight --dir "$DATA" || echo "   (preflight reported blockers — fill cards / eval before publishing)"

echo "==> DONE. Artifacts in $RESULTS/ (report.html, eval_decompose.json, manifest.json, bfcl/)"
