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

import functools
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
    offer, AND the required behavior all match — so fold those into the signature.

    We also prefix the SLICE GROUP (the specific hard-negative / multi-turn / schema-drift kind).
    This keeps an augmentation slice like ``over_refusal`` — deliberately a hedged near-paraphrase of
    a real positive — from being deleted as a "duplicate" of that positive, while base positives
    (no kind -> group "") still deduplicate across sources as before.
    """
    tools = sorted(t.get("name", "") for t in (ex.get("tools") or []))
    atype = (ex.get("answer") or {}).get("type", "")
    return f"{_dedup_group(ex)} | {atype} | {' '.join(tools)} | {ex.get('query', '')}"


def _dedup_group(ex: Dict[str, Any]) -> str:
    """The slice a row belongs to. Base positives share group "" (so they dedup across sources);
    each augmentation slice is its own group so it never collapses against a base positive it was
    intentionally derived from (e.g. an over_refusal hedge of a real positive)."""
    meta = ex.get("meta") or {}
    kind = meta.get("hn_kind") or meta.get("mt_kind") or meta.get("sd_kind")
    if kind:
        return kind
    if meta.get("source") == "env":
        return "env"  # execution-verified env rows dedup among themselves, not vs base positives
    if meta.get("source") == "multilingual":
        # keep per-language twins distinct from each other + from base positives
        return "ml-" + (meta.get("lang") or "")
    return ""


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


@functools.lru_cache(maxsize=1)
def _embed_model():
    """Load the static embedding model once (cached) — decontam + dedup share this singleton."""
    from model2vec import StaticModel

    return StaticModel.from_pretrained("minishlab/potion-base-8M")


def _embed(texts: List[str]):
    """Static sentence embeddings via model2vec (fast, CPU-friendly)."""
    return _embed_model().encode(texts)


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

    # Compare each example only against previously-kept examples of the SAME slice group. Otherwise a
    # deliberate near-paraphrase augmentation (e.g. over_refusal, a hedged copy of a real positive)
    # is wrongly deleted as a paraphrase of the positive it was built from.
    kept_idx: List[int] = []
    kept_vecs_by_group: Dict[str, Any] = {}
    for i in range(len(examples)):
        g = _dedup_group(examples[i])
        v = embs[i : i + 1]
        store = kept_vecs_by_group.get(g)
        if store is not None and float((store @ v.T).ravel().max()) >= threshold:
            continue
        kept_idx.append(i)
        kept_vecs_by_group[g] = v if store is None else np.vstack([store, v])
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
