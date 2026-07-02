"""Assemble the Data-Viz chart-QA dataset: synth + Indic (+ optional ReachQA) -> dedup -> novel-type
holdout -> group-split (Indic en/hi twins stay together) -> write canonical JSONL, a multimodal chat
JSONL (Together format), and a tabular JSONL (Adaption column_mapping) + stats.json.

Run:  python -m src.viz.build_dataset --out data/viz --n-synth 400 --n-indic 200 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List

from . import indic_charts, synth_charts
from .format_utils import answer_to_text, image_to_data_uri, to_chat_row


def load_reachqa(limit: int, out_dir: str) -> List[Dict[str, Any]]:
    """Load ReachQA (hewei2001/ReachQA, MIT) if `datasets` is installed; else skip. Defensive parsing."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[viz] datasets not installed; skipping ReachQA (pip install datasets)")
        return []
    try:
        ds = load_dataset("hewei2001/ReachQA", split="train", streaming=True)
    except Exception as e:
        print(f"[viz] ReachQA load skipped ({e})")
        return []
    out = []
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    for i, row in enumerate(ds):
        img = row.get("image") or row.get("images")
        q = row.get("question") or row.get("query")
        a = row.get("answer") or row.get("label")
        if not (img and q and a is not None):
            continue
        out.append(
            {
                "image": img,  # HF image struct/bytes; format_utils handles it at chat/tab time
                "question": q,
                "answer": a,
                "chart_type": "other",
                "qa_kind": "reasoning",
                "meta": {"source": "reachqa", "lang": "en", "script": None,
                         "chart_type": "other", "qa_kind": "reasoning", "answer_type": "numeric"},
            }
        )
        if len(out) >= limit:
            break
    print(f"[viz] ReachQA: loaded {len(out)}")
    return out


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", str(q).strip().lower())


def dedup(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for ex in examples:
        key = (_norm_q(ex["question"]), str(ex["answer"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    print(f"[viz] dedup: {len(examples)} -> {len(out)}")
    return out


def _group_key(ex: Dict[str, Any], i: int) -> str:
    # keep all QA of one chart together: Indic twins share pair_id; synth QA share an image path.
    img = ex.get("image")
    return ex["meta"].get("pair_id") or (img if isinstance(img, str) else f"solo_{i}")


def _save_image(img: Any, path: str) -> None:
    from PIL import Image
    import io

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if isinstance(img, dict):
        if img.get("path"):
            Image.open(img["path"]).convert("RGB").save(path)
            return
        img = img.get("bytes")
    if isinstance(img, (bytes, bytearray)):
        Image.open(io.BytesIO(bytes(img))).convert("RGB").save(path)
        return
    if hasattr(img, "save"):  # PIL
        img.convert("RGB").save(path)
        return
    raise ValueError(f"cannot materialize image of type {type(img)!r}")


def materialize_images(examples: List[Dict[str, Any]], out_dir: str) -> None:
    """Ensure every example's image is a path string so the canonical JSONL is serializable (ReachQA)."""
    for i, ex in enumerate(examples):
        img = ex.get("image")
        if img is None or isinstance(img, str):
            continue
        path = os.path.join(out_dir, "img", f"src_{i:06d}.png")
        _save_image(img, path)
        ex["image"] = path


def split(examples: List[Dict[str, Any]], novel_types: List[str], ratios, seed: int):
    novel = [e for e in examples if e["chart_type"] in novel_types]
    rest = [e for e in examples if e["chart_type"] not in novel_types]
    # group by pair_id so Indic twins never straddle splits
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, e in enumerate(rest):
        groups.setdefault(_group_key(e, i), []).append(e)
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_tr, n_va = int(n * ratios[0]), int(n * ratios[1])
    parts = {"train": [], "val": [], "test": []}
    for j, k in enumerate(keys):
        bucket = "train" if j < n_tr else ("val" if j < n_tr + n_va else "test")
        parts[bucket].extend(groups[k])
    parts["test_novel"] = novel
    return parts


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _tab_row(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Tabular row for Adaption column_mapping: prompt/completion/image(data-uri)."""
    return {
        "prompt": ex["question"],
        "completion": answer_to_text(ex["answer"]),
        "image": image_to_data_uri(ex["image"]) if ex.get("image") else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/viz")
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--n-indic", type=int, default=200)
    ap.add_argument("--n-reachqa", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--novel-types", nargs="+", default=["scatter", "area"])
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    render = not args.no_render
    examples: List[Dict[str, Any]] = []
    examples += synth_charts.generate(args.n_synth, args.out, seed=args.seed, render=render)
    examples += indic_charts.generate(args.n_indic, args.out, seed=args.seed, render=render)
    if args.n_reachqa:
        examples += load_reachqa(args.n_reachqa, args.out)

    examples = dedup(examples)
    materialize_images(examples, args.out)  # ReachQA etc. -> disk paths, so canonical JSONL serializes
    parts = split(examples, args.novel_types, (0.8, 0.1), args.seed)

    for name, rows in parts.items():
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), rows)
    # Together-style multimodal chat rows for train (inline base64) + Adaption tabular rows
    if render:
        write_jsonl(os.path.join(args.out, "train_chat.jsonl"), [to_chat_row(e) for e in parts["train"]])
        write_jsonl(os.path.join(args.out, "train_tab.jsonl"), [_tab_row(e) for e in parts["train"]])

    stats = {"total": sum(len(v) for v in parts.values())}
    for name, rows in parts.items():
        stats[name] = {
            "n": len(rows),
            "by_source": dict(Counter(r["meta"]["source"] for r in rows)),
            "by_chart_type": dict(Counter(r["chart_type"] for r in rows)),
            "by_qa_kind": dict(Counter(r["qa_kind"] for r in rows)),
            "by_lang": dict(Counter(r["meta"].get("lang", "en") for r in rows)),
        }
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=2)
    print("[viz] done:", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
