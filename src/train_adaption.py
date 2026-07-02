"""Train via the Adaption AutoScientist SDK.

Flow (per Adaption docs):
  1. Upload the prompt/completion dataset (data/out/train_pc.jsonl).
  2. Preview cost with estimate=True.
  3. Run adaptation/training, wait for completion.
  4. Print the evaluation_summary (grade_before / grade_after / improvement_percent) -- this is the
     platform's own proof of improvement, and goes straight into the model card.

NOTE: the Adaption SDK surface is read from docs.adaptionlabs.ai; method names may shift between SDK
versions. Keep this file thin and adjust the two calls (`upload` and `run`) if the SDK differs. Requires
ADAPTION_API_KEY in the environment (pt_live_...).

Run:  python -m src.train_adaption --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost only, don't train")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    api_key = os.environ.get("ADAPTION_API_KEY")
    if not api_key:
        raise SystemExit("Set ADAPTION_API_KEY (pt_live_...) in your environment first.")

    try:
        from adaption import Adaption
    except ImportError:
        raise SystemExit("Install the SDK: pip install adaption  (see adaptionlabs.ai)")

    client = Adaption(api_key=api_key)

    train_pc = os.path.join(cfg["paths"]["out_dir"], "train_pc.jsonl")
    if not os.path.exists(train_pc):
        raise SystemExit(f"{train_pc} not found — run: python -m src.build_dataset first")

    # 1. Upload -----------------------------------------------------------------
    print(f"[adaption] uploading {train_pc} ...")
    dataset = client.datasets.upload_file(train_pc)  # CSV/JSON/JSONL/Parquet supported
    dataset_id = getattr(dataset, "dataset_id", None) or getattr(dataset, "id", dataset)
    print(f"[adaption] dataset_id = {dataset_id}")

    # Wait for the async import to populate row_count before running.
    for _ in range(60):
        st = client.datasets.get_status(dataset_id)
        if getattr(st, "row_count", None) is not None or getattr(st, "status", None) == "failed":
            break
        time.sleep(3)

    column_mapping = cfg["adaption"]["column_mapping"]
    # The reliable-tool-calling objective lives in the brand_controls blueprint (system prompt applied
    # to every generated completion). run() has no base_model/objective — AutoScientist selects the model.
    recipe_specification = {"recipes": {"deduplication": True, "reasoning_traces": True}}
    brand_controls = {
        "length": "concise",
        "blueprint": (
            "You are a reliable function-calling assistant. Emit schema-correct JSON tool calls with all "
            "required arguments; REFUSE when no available tool applies; ASK for clarification when a "
            "required argument is missing or the tool choice is ambiguous. Never hallucinate a call or "
            "guess an argument."
        ),
    }
    job_specification = {}

    # 2. Cost estimate ----------------------------------------------------------
    est = client.datasets.run(
        dataset_id,
        column_mapping=column_mapping,
        recipe_specification=recipe_specification,
        brand_controls=brand_controls,
        job_specification=job_specification,
        estimate=True,
    )
    print("[adaption] estimate:", _summ(est))
    if args.dry_run:
        return

    # 3. Run + wait -------------------------------------------------------------
    print("[adaption] launching AutoScientist run ...")
    run = client.datasets.run(
        dataset_id,
        column_mapping=column_mapping,
        recipe_specification=recipe_specification,
        brand_controls=brand_controls,
        job_specification=job_specification,
    )
    run_id = getattr(run, "id", run)
    result = client.datasets.wait_for_completion(run_id)

    # 4. Report -----------------------------------------------------------------
    summary = getattr(result, "evaluation_summary", None) or _get(result, "evaluation_summary")
    print("[adaption] evaluation_summary:", json.dumps(summary, indent=2, default=str))

    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["results_dir"], "adaption_run.json")
    json.dump(
        {"dataset_id": str(dataset_id), "run_id": str(run_id), "evaluation_summary": summary},
        open(out, "w"),
        indent=2,
        default=str,
    )
    print(f"[adaption] saved -> {out}")
    print(
        "\n>> Download the trained weights via the SDK/console, then publish to HF + Kaggle "
        "(see model_card_template.md)."
    )


def _summ(obj):
    for k in ("estimated_minutes", "estimated_credits_consumed"):
        v = _get(obj, k)
        if v is not None:
            print(f"    {k}: {v}")
    return obj


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


if __name__ == "__main__":
    main()
