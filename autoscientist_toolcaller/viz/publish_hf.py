"""Publish the chart-QA dataset to the Hugging Face Hub as an IMAGE dataset.

Two purposes: (1) the mandatory open-release artifact, and (2) it unblocks multimodal training —
Adaption's `create_from_huggingface` can ingest an HF dataset whose `image` column is a real Image
feature (inline base64 in a JSONL is rejected by the platform).

Run:  python -m autoscientist_toolcaller.viz.publish_hf --data-dir data/viz --repo <user>/autoscientist-chartqa-dataset
Token: HF_TOKEN env var or --token.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from .format_utils import answer_to_text

_SPLIT_MAP = {"train": "train", "val": "validation", "test": "test", "test_novel": "test_novel"}


def _load(data_dir: str, name: str) -> List[Dict[str, Any]]:
    p = os.path.join(data_dir, f"{name}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _records(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        img = r.get("image")
        if not (isinstance(img, str) and os.path.exists(img)):
            continue
        out.append({
            "image": img,
            "question": r["question"],
            "answer": answer_to_text(r["answer"]),
            "chart_type": r.get("chart_type", "other"),
            "qa_kind": r.get("qa_kind", "other"),
            "lang": r.get("meta", {}).get("lang", "en"),
            "split": split,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/viz")
    ap.add_argument("--repo", required=True, help="e.g. pandeyankit84/autoscientist-chartqa-dataset")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--private", action="store_true", help="default is PUBLIC (challenge requires open)")
    args = ap.parse_args()
    if not args.token:
        raise SystemExit("Set HF_TOKEN or pass --token")

    from datasets import Dataset, DatasetDict, Image

    dd = {}
    for name, hf_split in _SPLIT_MAP.items():
        recs = _records(_load(args.data_dir, name), hf_split)
        if recs:
            dd[hf_split] = Dataset.from_list(recs).cast_column("image", Image())
            print(f"[hf] {hf_split}: {len(recs)} rows")
    if not dd:
        raise SystemExit(f"no examples found under {args.data_dir} (run autoscientist_toolcaller.viz.build_dataset first)")

    ds = DatasetDict(dd)
    ds.push_to_hub(args.repo, token=args.token, private=args.private)
    print(f"[hf] pushed -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
