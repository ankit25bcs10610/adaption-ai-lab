#!/usr/bin/env bash
# End-to-end pipeline. Stops on first error. Assumes venv active + env vars set:
#   HF_TOKEN, ADAPTION_API_KEY, (KAGGLE_USERNAME/KAGGLE_KEY for release).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${1:-config.yaml}

echo "==> 1/4  Build dataset (download, curate, hard negatives, dedup, split)"
python -m src.build_dataset --config "$CONFIG"

echo "==> 2/4  Baseline eval on the raw base model (the honest 'before' number)"
python -m src.baseline --config "$CONFIG"

echo "==> 3/4  Train via Adaption AutoScientist (cost estimate prints first)"
python -m src.train_adaption --config "$CONFIG"

echo "==> 4/5  Build DPO preference pairs (optional, targets the moat)"
python -m src.build_preference --config "$CONFIG"

echo "==> 5/5  Post-finetune eval + card (set FINETUNED_MODEL first)"
cat <<'EOF'
    Set FINETUNED_MODEL (HF id/path from the AutoScientist run), then:

    python -m src.eval_harness --model $FINETUNED_MODEL --data data/out/test.jsonl      --out results/eval.json
    python -m src.eval_bfcl    --model $FINETUNED_MODEL --data data/out/test.jsonl      --out results/eval_bfcl.json
    python -m src.eval_harness --model $FINETUNED_MODEL --data data/out/test_novel.jsonl --out results/eval_novel.json
    python -m src.error_analysis --model $BASELINE_MODEL   --data data/out/test.jsonl --out-dir results/base
    python -m src.error_analysis --model $FINETUNED_MODEL   --data data/out/test.jsonl --out-dir results/ft
    python -m src.eval_stats --base results/base/predictions.jsonl --finetuned results/ft/predictions.jsonl
    #   -> bootstrapped 95% CIs + paired base-vs-ft gap CI + bootstrap/McNemar p-values (significance)
    python -m src.eval_report      # -> results/report.html (base-vs-ft table, per-category bars, confusion matrix)
    python -m src.fill_model_card --username YOURUSER

    Then publish:
    python -m src.release hf-dataset   --repo YOURUSER/autoscientist-toolcall-dataset --dir data/out
    python -m src.release hf-model     --repo YOURUSER/autoscientist-toolcall         --dir <weights>
    python -m src.release kaggle-model --slug YOURUSER/autoscientist-toolcall         --dir <weights>
EOF

echo "Done with the automated stages."
