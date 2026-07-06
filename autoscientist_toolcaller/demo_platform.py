"""Offline, reproducible DEMO of the Adaption AutoScientist **platform** loop — the challenge's demo bonus.

The live site / HF Space demo the trained MODEL (call / refuse / clarify). This instead demonstrates the
AUTOSCIENTIST PLATFORM workflow this submission actually ran — upload → grade → improve → re-grade →
train — narrated with the REAL recorded dataset-quality grades. It makes NO network call and needs NO API
key: it replays canned numbers (the same ones in MODEL_CARD.md / DATASET_CARD.md) and, if present, the
real `results/adaption_run.json`. The real SDK calls live in `autoscientist_toolcaller/train_adaption.py`;
the narrative maps to `docs/AUTOSCIENTIST_USAGE.md`.

Run:  python -m autoscientist_toolcaller.demo_platform [--transcript results/autoscientist_demo.txt]
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Optional

# Recorded Adaptive Data grades for this submission's dataset (also cited in the cards). Canned so the
# demo is self-contained and reproducible without the (gitignored) results/ or any live API call.
_CANNED_RUNS: List[Dict[str, Any]] = [
    {"dataset_id": "a99c0c96", "rows": 250, "before": 8.0, "after": 8.8,
     "improvement_percent": 10.0, "grade": "B", "status": "completed"},
    {"dataset_id": "c4923b7f", "rows": 2440, "before": 7.0, "after": 8.1,
     "improvement_percent": 15.7, "grade": "C→B", "status": "partial (1,000/2,440 under the free-tier cap)"},
    {"dataset_id": "bea4a581", "rows": 5157, "before": 7.0, "after": 8.1,
     "improvement_percent": 15.7, "grade": "C→B", "status": "completed FULL run (5,133/5,157 rows; completion quality +31.5%)"},
    {"dataset_id": "4e4178c7", "rows": 7566, "before": None, "after": None,
     "improvement_percent": None, "grade": None, "status": "final published set — run launched; then AUTOSCIENTIST → Train"},
]

_STEPS = [
    ("1. UPLOAD", "client.datasets.upload_file(train_tab.jsonl) → dataset_id",
     "The adapted, correct-by-construction dataset is uploaded to Adaptive Data."),
    ("2. GRADE", "client.datasets.get_status(dataset_id) → quality grade (before)",
     "Adaptive Data scores dataset quality on its rubric — the honest 'before'."),
    ("3. IMPROVE", "client.datasets.run(recipe=[deduplication, reasoning_traces], brand_controls=blueprint)",
     "Platform recipes + a call/refuse/clarify brand-controls blueprint produce an ENHANCED dataset."),
    ("4. RE-GRADE", "evaluation_summary → grade_after + improvement_percent",
     "The enhanced dataset is re-graded; the delta is the data-centric improvement."),
    ("5. TRAIN", "AUTOSCIENTIST tab → Launch/Train → weights + held-out % (web console)",
     "AutoScientist trains the model on the enhanced data; produces weights + the held-out gate number."),
]


def _load_real_runs(results_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Use the real recorded run if results/adaption_run.json is present; else None (fall back to canned)."""
    path = os.path.join(results_dir, "adaption_run.json")
    if not os.path.exists(path):
        return None
    try:
        data = json.load(open(path))
    except Exception:
        return None
    runs = data if isinstance(data, list) else data.get("runs") or [data]
    out = []
    for r in runs:
        s = r.get("evaluation_summary") or r
        out.append({"dataset_id": str(r.get("dataset_id", "?"))[:8], "rows": r.get("rows"),
                    "before": s.get("grade_before"), "after": s.get("grade_after"),
                    "improvement_percent": s.get("improvement_percent"),
                    "grade": r.get("grade"), "status": r.get("status", "recorded")})
    return out or None


def _fmt(v: Any) -> str:
    return "—" if v is None else (f"{v:g}" if isinstance(v, (int, float)) else str(v))


def run(emit: Callable[[str], None] = print, results_dir: str = "results") -> str:
    """Walk the AutoScientist platform loop; return the full transcript (also emitted line-by-line)."""
    lines: List[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        emit(s)

    out("=" * 72)
    out("  Adaption AutoScientist — platform workflow demo (offline replay)")
    out("  upload → grade → improve → re-grade → train   (no network, no API key)")
    out("=" * 72)
    for tag, call, desc in _STEPS:
        out("")
        out(f"▶ {tag}")
        out(f"    $ {call}")
        out(f"    {desc}")

    runs = _load_real_runs(results_dir)
    source = "results/adaption_run.json (recorded)" if runs else "canned (matches the cards)"
    runs = runs or _CANNED_RUNS
    out("")
    out(f"── Measured dataset-quality grades  [source: {source}] " + "─" * 12)
    out(f"    {'dataset':10} {'rows':>6}  {'before':>6} {'after':>6} {'Δ%':>7}  grade / status")
    for r in runs:
        d = f"{_fmt(r.get('improvement_percent'))}%" if r.get("improvement_percent") is not None else "—"
        out(f"    {str(r.get('dataset_id'))[:10]:10} {_fmt(r.get('rows')):>6}  "
            f"{_fmt(r.get('before')):>6} {_fmt(r.get('after')):>6} {d:>7}  "
            f"{_fmt(r.get('grade'))} · {r.get('status','')}")
    out("")
    out("Note: the grade above is Adaptive Data's *dataset-quality* improvement (the data-centric lever).")
    out("The held-out *model* accuracy — the challenge's eligibility gate — is produced by the TRAIN step")
    out("in the web console. Real SDK: autoscientist_toolcaller/train_adaption.py · docs/AUTOSCIENTIST_USAGE.md")
    out("=" * 72)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default=None, help="also write the transcript to this path")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()
    transcript = run(results_dir=args.results_dir)
    if args.transcript:
        os.makedirs(os.path.dirname(args.transcript) or ".", exist_ok=True)
        open(args.transcript, "w", encoding="utf-8").write(transcript + "\n")
        print(f"\n[demo] transcript -> {args.transcript}")


if __name__ == "__main__":
    main()
