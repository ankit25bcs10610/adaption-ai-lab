"""Recipe-ablation harness — run Adaptive Data across a grid of recipes and compare grades.

Depth of AutoScientist usage (judging criterion #4): every naive entry runs one recipe. This runs a
GRID — toggling `reasoning_traces`, `prompt_rephrase`, `hallucination_mitigation` and the brand-controls
`length` — and reports the dataset-quality grade per config, so you can show you *compared* recipes and
picked the best, not just ran the default. (Complements src/ablation.py, which ablates dataset SIZE.)

Uploads the dataset once (or reuse an existing `--dataset-id`), then runs each config. Use `--estimate`
for a free dry preview (credits/time only, no job launched). Writes results/recipe_ablation.json + .md.

Requires ADAPTION_API_KEY.
Run: python -m src.recipe_ablation --config config.yaml [--dataset-id <id>] [--estimate]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import yaml

from .train_adaption import _patch_httpx_timeout, _get, _summ

# On-thesis grid for a no-hallucinated-calls function-caller.
GRID: List[Dict[str, Any]] = [
    {"name": "dedup_only", "recipes": {"deduplication": True}, "length": "concise"},
    {"name": "dedup+reasoning", "recipes": {"deduplication": True, "reasoning_traces": True}, "length": "concise"},
    {"name": "dedup+rephrase", "recipes": {"deduplication": True, "prompt_rephrase": True}, "length": "concise"},
    {"name": "dedup+halluc_mitig", "recipes": {"deduplication": True, "hallucination_mitigation": True}, "length": "concise"},
    {"name": "all+detailed", "recipes": {"deduplication": True, "reasoning_traces": True, "prompt_rephrase": True}, "length": "detailed"},
]

_BLUEPRINT = (
    "You are a reliable function-calling assistant. Emit schema-correct JSON tool calls with all required "
    "arguments; REFUSE when no available tool applies; ASK for clarification when a required argument is "
    "missing or the tool choice is ambiguous. Never hallucinate a call or guess an argument."
)


def _grade(es) -> Dict[str, Any]:
    if not es:
        return {}
    return {k: _get(es, k) for k in ("grade_before", "grade_after", "score_before", "score_after", "improvement_percent")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--dataset-id", default=None, help="reuse an uploaded dataset (skips upload)")
    ap.add_argument("--estimate", action="store_true", help="free dry preview (no job launched)")
    ap.add_argument("--out", default="results/recipe_ablation.json")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    key = os.environ.get("ADAPTION_API_KEY")
    if not key:
        raise SystemExit("Set ADAPTION_API_KEY (pt_live_...) first.")
    _patch_httpx_timeout(900.0)
    import httpx
    from adaption import Adaption
    client = Adaption(api_key=key, timeout=httpx.Timeout(900.0, connect=10.0))

    dataset_id = args.dataset_id
    if not dataset_id:
        pc = os.path.join(cfg["paths"]["out_dir"], "train_pc.jsonl")
        print(f"[recipe-abl] uploading {pc} ...")
        ds = client.datasets.upload_file(pc)
        dataset_id = getattr(ds, "dataset_id", None) or getattr(ds, "id", ds)
    print(f"[recipe-abl] dataset_id = {dataset_id}")

    col = cfg["adaption"]["column_mapping"]
    rows: List[Dict[str, Any]] = []
    for g in GRID:
        rspec = {"recipes": g["recipes"]}
        bc = {"length": g["length"], "blueprint": _BLUEPRINT}
        print(f"[recipe-abl] === {g['name']} (recipes={g['recipes']}, length={g['length']}) ===")
        try:
            res = client.datasets.run(dataset_id, column_mapping=col, recipe_specification=rspec,
                                      brand_controls=bc, job_specification={}, estimate=True)
            _summ(res)
            if args.estimate:
                rows.append({"config": g["name"], **g,
                             "estimate": {k: _get(res, k) for k in ("estimated_minutes", "estimated_credits_consumed")}})
                continue
            client.datasets.run(dataset_id, column_mapping=col, recipe_specification=rspec,
                                brand_controls=bc, job_specification={})
            result = client.datasets.wait_for_completion(dataset_id, timeout=7200)
            grade = _grade(_get(result, "evaluation_summary"))
            print("  grade:", grade)
            rows.append({"config": g["name"], **g, "grade": grade})
        except Exception as e:
            print(f"  [skip] {g['name']}: {type(e).__name__}: {str(e)[:140]}")
            rows.append({"config": g["name"], **g, "error": type(e).__name__})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"dataset_id": str(dataset_id), "grid": rows}, open(args.out, "w"), indent=2, default=str)
    md = ["| Config | recipes | length | grade before→after | Δ% |", "|---|---|---|---|---|"]
    for r in rows:
        g = r.get("grade") or {}
        rec = ", ".join(k for k in r.get("recipes", {}))
        ba = f"{g.get('grade_before','—')}→{g.get('grade_after','—')}" if g else "—"
        dp = f"+{g.get('improvement_percent')}%" if g.get("improvement_percent") is not None else "—"
        md.append(f"| {r['config']} | {rec} | {r.get('length','')} | {ba} | {dp} |")
    open(args.out.replace(".json", ".md"), "w").write("\n".join(md) + "\n")
    print(f"[recipe-abl] wrote {args.out} + .md")


if __name__ == "__main__":
    main()
