#!/usr/bin/env bash
# RUN-DAY FINALIZER — one command to turn AutoScientist-trained weights into every real number.
#
# The moment the console training loop gives you weights, this produces: baseline vs fine-tuned eval,
# multi-seed aggregation, paired significance (bootstrap CI + McNemar), the gap DECOMPOSITION, the
# robustness-delta table, the reliability probe, an HTML report, and a MODEL_CARD.md auto-filled with the
# real held-out numbers — then a release preflight. Numbers are never transcribed by hand.
#
# Usage:
#   MODEL=<hf-id-or-local-weights> [ADAPTER=<lora-path>] [SEEDS="41 42 43"] bash scripts/finalize.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
RESULTS=results; DATA=data/out
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SEEDS="${SEEDS:-42}"; S1="$(echo "$SEEDS" | awk '{print $1}')"
[ -z "${MODEL:-}" ] && { echo "Set MODEL=<hf-id-or-local-weights> (the AutoScientist-trained model)"; exit 1; }
mkdir -p "$RESULTS"
ADAPTER_ARG=""; [ -n "${ADAPTER:-}" ] && ADAPTER_ARG="--adapter ${ADAPTER}"

echo "==> [1/7] Baseline (untuned $BASE_MODEL) — the honest 'before'"
python3 -m src.baseline --config config.yaml --model "$BASE_MODEL" --out "$RESULTS/baseline.json"
python3 -m src.error_analysis --model "$BASE_MODEL" --data "$DATA/test.jsonl" --out-dir "$RESULTS/base"

echo "==> [2/7] Fine-tuned eval (multi-seed: $SEEDS)"
PAIRS=""
for s in $SEEDS; do
  python3 -m src.eval_harness   --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" --out "$RESULTS/eval_$s.json"
  python3 -m src.eval_bfcl      --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" --out "$RESULTS/eval_bfcl_$s.json" || true
  python3 -m src.error_analysis --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" --out-dir "$RESULTS/ft_$s"
  PAIRS="$PAIRS --pair $RESULTS/base/predictions.jsonl,$RESULTS/ft_$s/predictions.jsonl"
done
cp "$RESULTS/eval_$S1.json" "$RESULTS/eval.json"
cp "$RESULTS/eval_bfcl_$S1.json" "$RESULTS/eval_bfcl.json" 2>/dev/null || true
FT="$RESULTS/ft_$S1/predictions.jsonl"; BASEP="$RESULTS/base/predictions.jsonl"

echo "==> [2b/7] Multilingual matched-pair Δaccuracy(lang−en) — the HackIndia robustness number"
python3 -m src.eval_multilingual --model "$MODEL" $ADAPTER_ARG --data "$DATA/test.jsonl" \
  --out "$RESULTS/eval_multilingual.json" | tee "$RESULTS/multilingual.txt" || true

echo "==> [3/7] Novel-tools holdout (generalization)"
[ -f "$DATA/test_novel.jsonl" ] && python3 -m src.eval_harness --model "$MODEL" $ADAPTER_ARG --data "$DATA/test_novel.jsonl" --out "$RESULTS/eval_novel.json" || true

echo "==> [4/7] Significance + gap decomposition + robustness"
python3 -m src.eval_stats --base "$BASEP" --finetuned "$FT" --out "$RESULTS/eval_stats.json" | tee "$RESULTS/HEADLINE.txt"
python3 -m src.eval_decompose --base "$BASEP" --finetuned "$FT" $PAIRS --out "$RESULTS/eval_decompose.json"
python3 -m src.robustness_table --base "$BASEP" --finetuned "$FT" --out "$RESULTS/robustness.md"

echo "==> [5/7] Reliability probe"
python3 -m src.reliability_probe --model "$MODEL" $ADAPTER_ARG --out "$RESULTS/reliability_probe.md" || true

echo "==> [6/7] HTML report + auto-filled model card (real numbers)"
python3 -m src.eval_report --out "$RESULTS/report.html"
python3 -m src.fill_model_card --username pandeyankit84 --template model_card_template.md --out MODEL_CARD.md

echo "==> [7/7] Reproducibility manifest + release preflight"
python3 -m src.manifest --out-dir "$DATA" --config config.yaml --manifest "$RESULTS/manifest.json"
python3 -m src.release preflight --dir "$DATA" || echo "   (preflight blockers above)"

echo ""
echo "===================== HEADLINE ====================="
cat "$RESULTS/HEADLINE.txt"
echo "===================================================="
echo "Filled: MODEL_CARD.md · $RESULTS/report.html · eval_decompose.json · robustness.md · eval_multilingual.json"
echo "Publish weights next:"
echo "  python -m src.release hf-model     --repo pandeyankit84/autoscientist-toolcaller --dir <weights>"
echo "  python -m src.release kaggle-model --slug pandeyankit99/autoscientist-toolcaller --dir <weights>"
echo "Then paste the HEADLINE line into README + the social posts, and re-push MODEL_CARD.md to HF."
