#!/usr/bin/env bash
# Entry-B (Data-Visualization / chart-QA) finalize — the analog of scripts/finalize.sh for the FC track.
# Given a base VLM and the AutoScientist-fine-tuned VLM weights, it produces the honest before/after
# relaxed-accuracy numbers, auto-fills VIZ_MODEL_CARD.md (no hand-transcription), and refreshes the viz
# reproducibility manifest. Model stages need a GPU (transformers), so they no-op without the vars.
#
# Usage:
#   BASE_VLM=Qwen/Qwen3-VL-8B-Instruct MODEL=<ft-weights-or-hf-id> bash scripts/finalize_viz.sh
#   DATA=data/viz/test.jsonl BASE_VLM=... MODEL=... bash scripts/finalize_viz.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
RESULTS="results"; VIZ="data/viz"
DATA="${DATA:-$VIZ/test.jsonl}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3-VL-8B-Instruct}"
mkdir -p "$RESULTS"

if [ -n "${MODEL:-}" ]; then
  echo "==> [1/4] Base VLM ($BASE_VLM) — honest 'before' on $DATA"
  python3 -m autoscientist_toolcaller.viz.baseline --model "$BASE_VLM" --data "$DATA" --out "$RESULTS/viz_baseline.json"

  echo "==> [2/4] Fine-tuned VLM ($MODEL) — 'after' on the same split"
  python3 -m autoscientist_toolcaller.viz.baseline --model "$MODEL" --data "$DATA" --out "$RESULTS/viz_eval.json"
else
  echo "==> [1-2/4] VLM eval SKIPPED (set MODEL=<ft-weights> and BASE_VLM=<base> to run before/after)"
fi

echo "==> [3/4] Auto-fill VIZ_MODEL_CARD.md from the eval JSONs"
python3 -m autoscientist_toolcaller.viz.fill_card --username pandeyankit84 \
  --baseline "$RESULTS/viz_baseline.json" --finetuned "$RESULTS/viz_eval.json" \
  --adaption "$RESULTS/viz_adaption_run.json" \
  --template autoscientist_toolcaller/viz/model_card_template.md --out VIZ_MODEL_CARD.md

echo "==> [4/4] Viz reproducibility manifest + stage cards into publish dirs"
python3 -m autoscientist_toolcaller.manifest --viz --out-dir "$VIZ" --config config.yaml --manifest "$RESULTS/viz_manifest.json"
# Stage the viz dataset card so neither the HF nor Kaggle viz dataset ships card-less.
cp VIZ_DATASET_CARD.md "$VIZ/README.md"
mkdir -p data/kaggle_viz && python3 -c "import re; t=open('VIZ_DATASET_CARD.md').read(); m=re.match(r'^---\n.*?\n---\n+',t,re.DOTALL); open('data/kaggle_viz/README.md','w').write(t[m.end():] if m else t)"
# If a weights dir is given, ship the (now-filled) viz model card as its README so `release hf-model`'s
# preflight gate sees it (weights-pending state keeps an honest, model-index-omitted card).
if [ -n "${WEIGHTS:-}" ]; then cp VIZ_MODEL_CARD.md "$WEIGHTS/README.md"; echo "staged VIZ_MODEL_CARD.md -> $WEIGHTS/README.md"; fi

echo ""
echo "Filled: VIZ_MODEL_CARD.md · $VIZ/README.md · $RESULTS/viz_baseline.json · $RESULTS/viz_eval.json · $RESULTS/viz_manifest.json"
echo "Publish the chart-QA weights next (gates on the now-filled card):"
echo "  python -m autoscientist_toolcaller.release hf-model     --repo pandeyankit84/autoscientist-chartqa --dir <weights>"
echo "  python -m autoscientist_toolcaller.release kaggle-model --slug pandeyankit99/autoscientist-chartqa --dir <weights>"
