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

Run:  python -m autoscientist_toolcaller.train_adaption --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import time

import yaml


def _patch_httpx_timeout(seconds: float = 900.0) -> None:
    """Widen httpx timeouts so multi-MB dataset uploads don't abort mid-stream (httpx.WriteTimeout).

    The SDK's ``do_upload_file`` PUTs the whole file to S3 via the module-level ``httpx.put(url,
    content=...)`` with NO timeout, so httpx applies its 5s per-request write timeout — too short for
    a ~10MB body. That per-request default can't be raised by widening the Client constructor, so we
    wrap ``httpx.put``/``httpx.request`` directly. We also widen the Client constructors for the async
    upload path (which builds its own client) and the SDK's own client.
    """
    try:
        import httpx
    except ImportError:
        return
    # 1. Wrap the top-level convenience functions used by the S3 PUT upload.
    for fn_name in ("put", "request", "post"):
        fn = getattr(httpx, fn_name, None)
        if fn is None or getattr(fn, "_timeout_patched", False):
            continue

        def _wrapped(*a, _fn=fn, **kw):
            kw.setdefault("timeout", httpx.Timeout(seconds))
            return _fn(*a, **kw)

        _wrapped._timeout_patched = True
        setattr(httpx, fn_name, _wrapped)
    # 2. Widen the Client constructors (covers the async upload path + secondary clients).
    for cls_name in ("Client", "AsyncClient"):
        cls = getattr(httpx, cls_name, None)
        if cls is None or getattr(cls.__init__, "_timeout_patched", False):
            continue
        orig = cls.__init__

        def _init(self, *a, _orig=orig, **kw):
            kw.setdefault("timeout", httpx.Timeout(seconds))
            _orig(self, *a, **kw)

        _init._timeout_patched = True
        cls.__init__ = _init


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="estimate cost only, don't train")
    ap.add_argument("--preference", action="store_true",
                    help="second objective: train on pref.jsonl with training_type=preference_pairs (DPO)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    api_key = os.environ.get("ADAPTION_API_KEY")
    if not api_key:
        raise SystemExit("Set ADAPTION_API_KEY (pt_live_...) in your environment first.")

    try:
        from adaption import Adaption
    except ImportError:
        raise SystemExit("Install the SDK: pip install adaption  (see adaptionlabs.ai)")

    # The SDK's default timeout (60s total) aborts multi-MB dataset uploads. Raise it via the
    # supported constructor arg; also patch httpx as a fallback for any secondary (e.g. storage) client.
    import httpx

    _patch_httpx_timeout(900.0)
    client = Adaption(api_key=api_key, timeout=httpx.Timeout(900.0, connect=10.0))

    # Objective: instruction (train_pc.jsonl) or preference-pairs (pref.jsonl: prompt/chosen/rejected).
    if args.preference:
        upload_path = os.path.join(cfg["paths"]["out_dir"], "pref.jsonl")
        training_type = "preference_pairs"
    else:
        upload_path = os.path.join(cfg["paths"]["out_dir"], "train_pc.jsonl")
        training_type = "instruction_dataset"
    if not os.path.exists(upload_path):
        raise SystemExit(f"{upload_path} not found — run build_dataset / build_preference first")

    # 1. Upload -----------------------------------------------------------------
    print(f"[adaption] uploading {upload_path} (objective: {training_type}) ...")
    dataset = client.datasets.upload_file(upload_path)  # CSV/JSON/JSONL/Parquet supported
    dataset_id = getattr(dataset, "dataset_id", None) or getattr(dataset, "id", dataset)
    print(f"[adaption] dataset_id = {dataset_id}")

    # Wait for the async import to populate row_count before running.
    for _ in range(60):
        st = client.datasets.get_status(dataset_id)
        if getattr(st, "row_count", None) is not None or getattr(st, "status", None) == "failed":
            break
        time.sleep(3)

    # preference pairs carry prompt/chosen/rejected (the platform maps chosen/rejected by convention);
    # instruction data uses the configured prompt/completion mapping.
    column_mapping = {"prompt": "prompt"} if args.preference else cfg["adaption"]["column_mapping"]
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
        training_type=training_type,
        estimate=True,
    )
    print("[adaption] estimate:", _summ(est))
    if args.dry_run:
        return

    # 3. Run + wait -------------------------------------------------------------
    print(f"[adaption] launching AutoScientist run (training_type={training_type}) ...")
    run = client.datasets.run(
        dataset_id,
        column_mapping=column_mapping,
        recipe_specification=recipe_specification,
        brand_controls=brand_controls,
        job_specification=job_specification,
        training_type=training_type,
    )
    run_id = getattr(run, "id", run)
    # wait_for_completion polls by DATASET id (the SDK tracks the run against its dataset), not run id.
    result = client.datasets.wait_for_completion(dataset_id, timeout=7200)

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
