"""Decontamination — prove the held-out gain isn't test-set leakage.

A big improvement on the platform's hidden test set is only believable if the training data doesn't
overlap it. We can't see the hidden set, but we CAN guard against the obvious leak: training queries
that duplicate the public BFCL / ToolACE-style prompts the hidden set is drawn from. This module drops
(and reports) any training example whose query is a near-duplicate of a probe, by:

  * lexical n-gram Jaccard (reusing dedup._shingles), and
  * semantic cosine (reusing dedup._embed / model2vec) — skipped gracefully if the model is unavailable.

Probes are a hand-authored, offline fixture (no network fetch) of representative public-style queries;
extend via --probes PATH. Everything is deterministic and offline. Wire into build_dataset via
`dedup.decontaminate: true` in config; a `contamination` block lands in stats.json for provenance.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from .dedup import _embed, _shingles

# Representative public-benchmark-style queries (BFCL simple/parallel/irrelevance + ToolACE phrasings).
# Not the real hidden set — a decoy fixture whose only purpose is to catch accidental overlap.
DEFAULT_PROBES: List[str] = [
    "What's the weather like in San Francisco in celsius?",
    "Get the current temperature for New York and Los Angeles.",
    "Book a flight from JFK to LAX on December 25th.",
    "Find the latest stock price for Apple and Microsoft.",
    "Calculate the area of a triangle with base 10 and height 5.",
    "Convert 100 US dollars to euros.",
    "What is the distance between Paris and London?",
    "Send an email to john@example.com about the quarterly report.",
    "Set a reminder for my dentist appointment tomorrow at 3pm.",
    "Search for Italian restaurants near me.",
    "What movies are playing this weekend?",
    "Play the song Bohemian Rhapsody by Queen.",
    "Translate 'hello world' into Spanish.",
    "Get the top news headlines for today.",
    "What is the exchange rate from GBP to JPY?",
    "Order a large pepperoni pizza for delivery.",
    "Add a meeting with the marketing team on Friday at 2pm.",
    "How many calories are in a banana?",
    "Find flights from Boston to Seattle next Monday.",
    "What's the population of Tokyo?",
    "Compute the factorial of 7.",
    "Get directions from the airport to downtown.",
    "Retrieve the account balance for user 12345.",
    "Schedule a video call with the engineering team.",
    "What time zone is Sydney in?",
    "Look up the definition of the word serendipity.",
    "Get the weather forecast for the next five days in Chicago.",
    "Cancel my subscription to the premium plan.",
    "Find the cheapest hotel in Las Vegas for this weekend.",
    "What is the current price of Bitcoin in USD?",
    # parallel (multi-call)
    "Get the weather in Berlin and the current time in Tokyo.",
    "Compare the stock prices of Google and Amazon.",
    "Book a table for two on Saturday and add it to my calendar.",
    # multiple / tool-selection
    "I want to either text or email Sarah that I'm running late.",
    "Find a nearby pharmacy or order the medicine online.",
    # irrelevance / chit-chat / creative (must refuse — no applicable tool)
    "Write a short story about a dragon who learns to code.",
    "What do you think about modern art?",
    "Tell me a fun fact about deep space.",
    "Give me some advice on staying motivated at work.",
    "Recommend a novel similar to Dune.",
    # multi-turn follow-ups
    "Actually, move that meeting to 4pm instead.",
    "No, use the other credit card for that order.",
    "Change the destination to Denver.",
    # smart home / IoT
    "Turn off the living room lights and set the thermostat to 68.",
    "Lock the front door and arm the security system.",
    # database / CRM
    "Update the shipping address for order 98765.",
    "Create a new contact for Jane Doe with her phone number.",
    "Delete the calendar event titled Sprint Review.",
    # media
    "Skip to the next track and lower the volume.",
    "Pause the movie and turn on subtitles.",
    # maps / travel
    "How long will it take to drive from Austin to Houston?",
    "Find gas stations along my route to the beach.",
    # finance
    "Transfer 500 dollars from checking to savings.",
    "What is my current credit card balance?",
    # productivity
    "Summarize the attached quarterly document.",
    "Add a to-do item to buy groceries after work.",
    # weather variants
    "Will it rain tomorrow in Denver?",
    "What is the humidity in Miami right now?",
    # math / units
    "What is 15 percent of 240?",
    "Convert 5 miles to kilometers.",
    "What is the square root of 144?",
    # sports / misc
    "Get the score of last night's Lakers game.",
    "What are the opening hours of the nearest post office?",
    "Check whether my package has been delivered yet.",
    "Start a 10 minute timer for the pasta.",
    "Find a highly rated sushi place open now.",
    "What's on my schedule for tomorrow morning?",
    "Mute all notifications until 9am.",
    "Show me the unread emails from my manager.",
]


def _load_probes(path: Optional[str]) -> List[str]:
    if not path:
        return list(DEFAULT_PROBES)
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            out.append(obj["query"] if isinstance(obj, dict) else str(obj))
        except json.JSONDecodeError:
            out.append(line)
    return out or list(DEFAULT_PROBES)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def decontaminate(
    examples: List[Dict[str, Any]],
    probe_texts: List[str],
    ngram_threshold: float = 0.6,
    cos_threshold: float = 0.92,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (kept, dropped_records). A row is dropped if its query exceeds EITHER threshold vs any probe.

    The semantic pass degrades gracefully to n-gram-only if embeddings are unavailable (offline).
    """
    probe_shingles = [_shingles(p) for p in probe_texts]

    # Try the semantic pass; fall back to n-gram-only on any failure (offline-safe).
    probe_vecs = train_vecs = None
    try:
        import numpy as np

        pv = np.asarray(_embed(probe_texts), dtype="float32")
        tv = np.asarray(_embed([e.get("query", "") for e in examples]), dtype="float32")

        def _norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True)
            n[n == 0] = 1.0
            return a / n

        probe_vecs, train_vecs = _norm(pv), _norm(tv)
    except Exception as e:  # model2vec/numpy missing or offline
        print(f"[decontam] semantic pass skipped ({e}); n-gram only", file=sys.stderr)

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        q = ex.get("query", "")
        sh = _shingles(q)
        ngram = max((_jaccard(sh, ps) for ps in probe_shingles), default=0.0)
        cos = 0.0
        if train_vecs is not None:
            cos = float((probe_vecs @ train_vecs[i:i + 1].T).max())
        if ngram >= ngram_threshold or cos >= cos_threshold:
            dropped.append({"query": q, "ngram": round(ngram, 3), "cos": round(cos, 3),
                            "source": ex.get("meta", {}).get("source")})
        else:
            kept.append(ex)
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/out/train.jsonl")
    ap.add_argument("--probes", default=None, help="JSONL of probe queries (defaults to built-in fixture)")
    ap.add_argument("--ngram-threshold", type=float, default=0.6)
    ap.add_argument("--cos-threshold", type=float, default=0.92)
    ap.add_argument("--out", default="results/decontamination.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    probes = _load_probes(args.probes)
    kept, dropped = decontaminate(rows, probes, args.ngram_threshold, args.cos_threshold)
    report = {
        "probes_checked": len(probes), "examples": len(rows),
        "dropped_count": len(dropped), "kept": len(kept),
        "ngram_threshold": args.ngram_threshold, "cos_threshold": args.cos_threshold,
        "dropped": dropped[:50],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[decontam] {len(rows)} examples, {len(probes)} probes -> dropped {len(dropped)} contaminated")
    print(f"[decontam] wrote {args.out}")


if __name__ == "__main__":
    main()
