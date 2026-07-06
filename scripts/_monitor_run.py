"""Poll a running Adaptive Data job to completion, then download the enhanced dataset.

Run: ADAPTION_API_KEY=pt_live_... python scripts/_monitor_run.py <dataset_id>
Writes: results/adaption_run_<short>.json  and  data/adaptive_out/ (downloaded enhanced data).
"""
import json
import os
import sys
import time

from adaption import Adaption

DID = sys.argv[1] if len(sys.argv) > 1 else "4e4178c7-adf1-4fdf-9c76-90ad29275047"
KEY = os.environ["ADAPTION_API_KEY"]
client = Adaption(api_key=KEY)

deadline = 3 * 3600  # 3h cap
start_pct = None
waited = 0
while waited < deadline:
    st = client.datasets.get_status(DID)
    status = getattr(st, "status", None)
    prog = getattr(st, "progress", None)
    pct = getattr(prog, "percent", None) if prog else None
    print(f"[monitor] t+{waited}s status={status} percent={pct}", flush=True)
    if status == "succeeded":
        break
    if status == "failed":
        err = getattr(st, "error_data", None)
        print(f"[monitor] FAILED: {err}", flush=True)
        sys.exit(2)
    time.sleep(60)
    waited += 60
else:
    print("[monitor] TIMEOUT waiting for completion", flush=True)
    sys.exit(3)

# Completed — capture evaluation
ev = client.datasets.get_evaluation(DID)
q = ev.quality
summary = {
    "grade_before": q.grade_before, "grade_after": q.grade_after,
    "improvement_percent": q.improvement_percent,
    "score_before": q.score_before, "score_after": q.score_after,
    "percentile_after": getattr(q, "percentile_after", None),
}
raw = getattr(ev, "raw_results", None)
try:
    raw = dict(raw)
except Exception:
    raw = str(raw)
out = {
    "dataset_id": DID, "status": "succeeded",
    "source": "live Adaption API (scripts/_monitor_run.py)",
    "evaluation_summary": summary,
    "raw_results_gains": raw.get("gains") if isinstance(raw, dict) else None,
}
os.makedirs("results", exist_ok=True)
short = DID.split("-")[0]
with open(f"results/adaption_run_{short}.json", "w") as f:
    json.dump(out, f, indent=2)
print("[monitor] EVAL:", json.dumps(summary), flush=True)

# Download the enhanced dataset
os.makedirs("data/adaptive_out", exist_ok=True)
try:
    dl = client.datasets.download(DID)
    # SDK returns bytes / a response object / or writes a file depending on version — handle common cases
    target = f"data/adaptive_out/{short}_enhanced"
    content = getattr(dl, "content", None)
    if isinstance(dl, (bytes, bytearray)):
        open(target + ".bin", "wb").write(dl)
        print(f"[monitor] downloaded bytes -> {target}.bin", flush=True)
    elif content is not None:
        open(target + ".bin", "wb").write(content)
        print(f"[monitor] downloaded response.content -> {target}.bin", flush=True)
    else:
        # last resort: repr so we can see what the SDK returned
        open(target + ".repr.txt", "w").write(repr(dl)[:5000])
        print(f"[monitor] download() returned {type(dl).__name__}; wrote repr -> {target}.repr.txt", flush=True)
except Exception as e:
    print(f"[monitor] download error: {type(e).__name__}: {str(e)[:200]}", flush=True)

print("[monitor] DONE", flush=True)
