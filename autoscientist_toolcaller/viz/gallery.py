"""Build a self-contained HTML gallery of the chart-QA dataset — see the self-verifying data at a glance.

Picks a diverse sample (across chart types and languages) and renders each chart next to its
question, gold answer, qa_kind and language. Great for a README screenshot or a quick demo of the
"correct by construction" moat (including the Hindi slice).

Run:  python -m autoscientist_toolcaller.viz.gallery --data-dir data/viz_sample --n 18 --out data/viz_sample/gallery.html
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from collections import defaultdict
from typing import Any, Dict, List


def _load(data_dir: str) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        if os.path.basename(p) in ("train_chat.jsonl", "train_tab.jsonl"):
            continue
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if isinstance(r.get("image"), str) and os.path.exists(
        os.path.join(data_dir, os.path.relpath(r["image"], data_dir)) if os.path.isabs(r["image"]) else
        os.path.join(data_dir, os.path.relpath(r["image"], data_dir) if r["image"].startswith(data_dir) else r["image"])
    ) or True]


def _diverse(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Round-robin across (chart_type, lang) buckets for variety."""
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r.get("chart_type"), r.get("meta", {}).get("lang", "en"))].append(r)
    out, keys = [], list(buckets.keys())
    i = 0
    while len(out) < n and any(buckets[k] for k in keys):
        b = buckets[keys[i % len(keys)]]
        if b:
            out.append(b.pop(0))
        i += 1
    return out


_LANG_BADGE = {"en": "#22C55E", "hi": "#8B5CF6", "hi-romanized": "#22D3EE"}


def _rel(data_dir: str, img: str) -> str:
    return os.path.relpath(img, data_dir) if os.path.isabs(img) else img.split(data_dir.rstrip("/") + "/")[-1] if data_dir in img else img


def build(data_dir: str, n: int, out: str) -> str:
    rows = _diverse(_load(data_dir), n)
    cards = []
    for r in rows:
        lang = r.get("meta", {}).get("lang", "en")
        color = _LANG_BADGE.get(lang, "#94A3B8")
        img_rel = _rel(data_dir, r["image"])
        cards.append(f"""
      <figure class="card">
        <img src="{html.escape(img_rel)}" alt="chart" loading="lazy"/>
        <figcaption>
          <div class="tags">
            <span class="tag" style="background:{color}22;border-color:{color}66;color:{color}">{html.escape(lang)}</span>
            <span class="tag muted">{html.escape(str(r.get('chart_type')))}</span>
            <span class="tag muted">{html.escape(str(r.get('qa_kind')))}</span>
          </div>
          <p class="q">Q: {html.escape(str(r.get('question')))}</p>
          <p class="a">A: <b>{html.escape(str(r.get('answer')))}</b></p>
        </figcaption>
      </figure>""")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>AutoScientist Chart-QA Gallery</title>
<style>
  body{{background:#0B1120;color:#F8FAFC;font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:32px}}
  h1{{font-size:24px;margin:0 0 4px}} p.sub{{color:#94A3B8;margin:0 0 24px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px}}
  .card{{background:#111c33;border:1px solid #1e293b;border-radius:14px;padding:12px;margin:0}}
  .card img{{width:100%;border-radius:8px;background:#fff}}
  figcaption{{padding:8px 2px 2px}}
  .tags{{display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap}}
  .tag{{font-size:11px;border:1px solid #334155;border-radius:999px;padding:2px 8px}}
  .tag.muted{{color:#94A3B8}}
  .q{{font-size:13px;margin:4px 0;color:#e2e8f0}} .a{{font-size:13px;margin:4px 0;color:#22C55E}}
</style></head><body>
  <h1>AutoScientist Chart-QA — self-verifying dataset</h1>
  <p class="sub">Answers computed from the underlying data (correct by construction). English + Hindi (Devanagari/romanized). {len(rows)} of {len(_load(data_dir))} examples.</p>
  <div class="grid">{''.join(cards)}</div>
</body></html>"""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w", encoding="utf-8").write(doc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/viz_sample")
    ap.add_argument("--n", type=int, default=18)
    ap.add_argument("--out", default="data/viz_sample/gallery.html")
    args = ap.parse_args()
    path = build(args.data_dir, args.n, args.out)
    print(f"[gallery] wrote {path}")


if __name__ == "__main__":
    main()
