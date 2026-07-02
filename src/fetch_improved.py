"""Fetch the platform's IMPROVED dataset — the actual Adaptive Data deliverable.

Adaptive Data is data-centric: it rewrites each row into an `enhanced_prompt` / `enhanced_completion`
(clearer instructions, more complete answers) and attaches a `row_embedding`. This downloads a finished
dataset by id and writes a clean prompt/completion JSONL of the *enhanced* pairs (embeddings stripped),
ready to train a model on — the artifact behind the platform's quality-grade improvement.

Requires ADAPTION_API_KEY. Read-only (no write token needed).

Usage:
  python -m src.fetch_improved --dataset-id <id> --out data/adaptive_out/enhanced_train_pc.jsonl
"""
from __future__ import annotations

import argparse
import json
import os


def extract_enhanced(raw_rows, prefer_enhanced: bool = True):
    """Yield {prompt, completion} using the enhanced fields when present (fallback to originals)."""
    for r in raw_rows:
        if prefer_enhanced:
            p = r.get("enhanced_prompt") or r.get("prompt")
            c = r.get("enhanced_completion") or r.get("completion")
        else:
            p, c = r.get("prompt"), r.get("completion")
        if p and c:
            yield {"prompt": p, "completion": c}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--out", default="data/adaptive_out/enhanced_train_pc.jsonl")
    ap.add_argument("--raw-out", default=None, help="also save the full raw download (with embeddings)")
    ap.add_argument("--original", action="store_true", help="export original pairs, not enhanced")
    args = ap.parse_args()

    key = os.environ.get("ADAPTION_API_KEY")
    if not key:
        raise SystemExit("Set ADAPTION_API_KEY (pt_live_...) first.")
    from .train_adaption import _patch_httpx_timeout
    _patch_httpx_timeout(600)
    from adaption import Adaption

    client = Adaption(api_key=key)
    print(f"[fetch] downloading improved dataset {args.dataset_id} ...")
    data = client.datasets.download(args.dataset_id, file_format="jsonl")
    text = data if isinstance(data, str) else str(data)
    rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    print(f"[fetch] {len(rows)} rows")

    if args.raw_out:
        os.makedirs(os.path.dirname(args.raw_out) or ".", exist_ok=True)
        open(args.raw_out, "w", encoding="utf-8").write(text)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in extract_enhanced(rows, prefer_enhanced=not args.original):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    kind = "original" if args.original else "enhanced"
    print(f"[fetch] wrote {n} {kind} prompt/completion pairs -> {args.out}")


if __name__ == "__main__":
    main()
