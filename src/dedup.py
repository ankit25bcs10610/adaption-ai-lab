"""Deduplication + cross-split leakage detection.

Two passes:
  1. Lexical near-dup via MinHash/LSH (datasketch) on the query text.
  2. Semantic near-dup via static embeddings (model2vec) + cosine.

Cross-split leakage (a near-duplicate of a test query appearing in train) silently inflates the score,
so `drop_cross_split_leaks` removes any train/val example too close to a test/val example.

Both passes degrade gracefully: if an optional dependency is missing, that pass is skipped with a warning
rather than crashing the pipeline.
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Set, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _dedup_text(ex: Dict[str, Any]) -> str:
    """The identity used for near-dup detection.

    Keying on the query ALONE wrongly collapses distinct training signal that shares a query:
    a Hammer ``no_tool`` reuses a real positive's query (but offers different tools and the answer
    is *refuse*), and the templated ``ambiguous`` / ``miss_param`` slices share a fixed query
    across different tool sets. Two rows are true duplicates only when the request, the tools on
    offer, AND the required behavior all match — so fold all three into the signature.
    """
    tools = sorted(t.get("name", "") for t in (ex.get("tools") or []))
    atype = (ex.get("answer") or {}).get("type", "")
    return f"{atype} | {' '.join(tools)} | {ex.get('query', '')}"


def _shingles(text: str, k: int = 3) -> Set[str]:
    toks = _TOKEN_RE.findall(text.lower())
    if len(toks) < k:
        return set(toks)
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def minhash_dedup(
    examples: List[Dict[str, Any]], threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """Remove lexical near-duplicates by query text. Keeps first occurrence."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("[dedup] datasketch not installed; skipping MinHash pass", file=sys.stderr)
        return examples

    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept: List[Dict[str, Any]] = []
    for i, ex in enumerate(examples):
        m = MinHash(num_perm=128)
        for sh in _shingles(_dedup_text(ex)):
            m.update(sh.encode("utf-8"))
        if lsh.query(m):  # a near-dup already indexed
            continue
        lsh.insert(str(i), m)
        kept.append(ex)
    print(f"[dedup] MinHash: {len(examples)} -> {len(kept)}")
    return kept


def _embed(texts: List[str]):
    """Static sentence embeddings via model2vec (fast, CPU-friendly)."""
    from model2vec import StaticModel

    model = StaticModel.from_pretrained("minishlab/potion-base-8M")
    return model.encode(texts)


def semantic_dedup(
    examples: List[Dict[str, Any]], threshold: float = 0.90
) -> List[Dict[str, Any]]:
    """Remove paraphrase near-duplicates by cosine similarity of query embeddings."""
    try:
        import numpy as np
    except ImportError:
        print("[dedup] numpy missing; skipping semantic pass", file=sys.stderr)
        return examples
    try:
        embs = _embed([_dedup_text(e) for e in examples])
    except Exception as e:  # model2vec missing or offline
        print(f"[dedup] semantic pass skipped ({e})", file=sys.stderr)
        return examples

    embs = np.asarray(embs, dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms

    kept_idx: List[int] = []
    kept_vecs = None
    for i in range(len(examples)):
        v = embs[i : i + 1]
        if kept_vecs is not None:
            sims = (kept_vecs @ v.T).ravel()
            if float(sims.max()) >= threshold:
                continue
        kept_idx.append(i)
        kept_vecs = v if kept_vecs is None else np.vstack([kept_vecs, v])
    kept = [examples[i] for i in kept_idx]
    print(f"[dedup] semantic: {len(examples)} -> {len(kept)}")
    return kept


def drop_cross_split_leaks(
    train: List[Dict[str, Any]],
    holdouts: List[List[Dict[str, Any]]],
    threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    """Drop train examples whose query is a near-dup of any holdout (val/test) query."""
    try:
        import numpy as np
        hold_texts = [e["query"] for hs in holdouts for e in hs]
        if not hold_texts:
            return train
        train_embs = np.asarray(_embed([e["query"] for e in train]), dtype="float32")
        hold_embs = np.asarray(_embed(hold_texts), dtype="float32")
    except Exception as e:
        print(f"[dedup] cross-split check skipped ({e})", file=sys.stderr)
        return train

    def _norm(a):
        n = np.linalg.norm(a, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return a / n

    train_embs, hold_embs = _norm(train_embs), _norm(hold_embs)
    sims = train_embs @ hold_embs.T  # (n_train, n_hold)
    keep_mask = sims.max(axis=1) < threshold
    kept = [ex for ex, k in zip(train, keep_mask) if k]
    print(f"[dedup] cross-split leakage: dropped {len(train) - len(kept)} train examples")
    return kept


def dedup_all(
    examples: List[Dict[str, Any]],
    minhash_threshold: float = 0.85,
    semantic_threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    out = minhash_dedup(examples, minhash_threshold)
    out = semantic_dedup(out, semantic_threshold)
    return out
