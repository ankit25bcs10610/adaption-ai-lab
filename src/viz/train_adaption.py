"""Train the chart-QA VLM via the Adaption AutoScientist SDK (multimodal).

Uploads the tabular dataset (prompt/completion/image data-URI), then runs Adaptive Data with the
dedicated multimodal `image` column mapping plus recipes (dedup, reasoning_traces for chart reasoning)
and a chart-analyst `blueprint`. estimate=True first to budget the 1,000 credits.

Run:  python -m src.viz.train_adaption --data data/viz/train_tab.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/viz/train_tab.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--dry-run", action="store_true", help="cost estimate only")
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows for a cheap estimate pass")
    args = ap.parse_args()

    if not os.environ.get("ADAPTION_API_KEY"):
        raise SystemExit("Set ADAPTION_API_KEY (pt_live_...) first.")
    try:
        from adaption import Adaption
    except ImportError:
        raise SystemExit("pip install adaption")

    client = Adaption(api_key=os.environ["ADAPTION_API_KEY"])
    if not os.path.exists(args.data):
        raise SystemExit(f"{args.data} not found — run: python -m src.viz.build_dataset")

    print(f"[viz-train] uploading {args.data} ...")
    ds = client.datasets.upload_file(args.data)
    dataset_id = getattr(ds, "id", ds)

    column_mapping = {"prompt": "prompt", "completion": "completion", "image": "image"}
    recipe_specification = {"recipes": {"deduplication": True, "reasoning_traces": True}}
    brand_controls = {
        "length": "concise",
        "blueprint": (
            "You are a meticulous chart analyst. Read values directly off the chart. Answer with only "
            "the value, category, yes/no, or trend word requested — no explanation."
        ),
    }
    job_specification = {"base_model": args.base_model}
    if args.max_rows:
        job_specification["max_rows"] = args.max_rows

    est = client.datasets.run(
        dataset_id, column_mapping=column_mapping, recipe_specification=recipe_specification,
        brand_controls=brand_controls, job_specification=job_specification, estimate=True,
    )
    print("[viz-train] estimate:",
          getattr(est, "estimated_credits_consumed", "?"), "credits;",
          getattr(est, "estimated_minutes", "?"), "min")
    if args.dry_run:
        return

    run = client.datasets.run(
        dataset_id, column_mapping=column_mapping, recipe_specification=recipe_specification,
        brand_controls=brand_controls, job_specification=job_specification,
    )
    run_id = getattr(run, "id", run)
    result = client.datasets.wait_for_completion(dataset_id)
    summary = getattr(result, "evaluation_summary", None)
    print("[viz-train] evaluation_summary:", json.dumps(summary, indent=2, default=str))
    os.makedirs("results", exist_ok=True)
    json.dump({"dataset_id": str(dataset_id), "run_id": str(run_id), "evaluation_summary": summary},
              open("results/viz_adaption_run.json", "w"), indent=2, default=str)
    print("[viz-train] saved -> results/viz_adaption_run.json")


if __name__ == "__main__":
    main()
