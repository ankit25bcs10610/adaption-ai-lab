"""Tool retrieval — rank tools by relevance so tool-calling scales past a promptful (advanced).

Real function-calling systems expose hundreds of tools; the prompt can't hold them all. This retriever
ranks tools (name + description + parameter names/descriptions) against a query — a **hybrid** of lexical
TF-IDF and model2vec dense cosine (with a pure-lexical fallback) — so only the top-k go into the prompt.
Deterministic, offline; recall@k is exactly checkable.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List

_TOK = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "with",
         "get", "set", "by", "from", "is", "it", "you", "me", "my"}


def _tokens(s: Any) -> List[str]:
    return [t for t in _TOK.findall(str(s).lower()) if len(t) > 1 and t not in _STOP]


def _tool_doc(tool: Dict[str, Any]) -> List[str]:
    props = ((tool.get("parameters") or {}).get("properties") or {})
    parts = [tool.get("name", ""), tool.get("name", "").replace("_", " "), tool.get("description", "")]
    parts += list(props.keys())
    parts += [v.get("description", "") for v in props.values() if isinstance(v, dict)]
    return _tokens(" ".join(parts))


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return [0.0 for _ in xs]
    return [(x - lo) / (hi - lo) for x in xs]


def _tool_text(tool: Dict[str, Any]) -> str:
    """Natural-language rendering of a tool for DENSE embedding (name + description + param names/descs)."""
    props = ((tool.get("parameters") or {}).get("properties") or {})
    pdesc = "; ".join(f"{k}: {v.get('description', '')}" for k, v in props.items() if isinstance(v, dict))
    return f"{tool.get('name','')}. {tool.get('description','')}. params: {', '.join(props.keys())}. {pdesc}"


class ToolRetriever:
    """Hybrid tool retriever: lexical TF-IDF **fused** with model2vec dense cosine (min-max normalized,
    weighted). Falls back to TF-IDF alone if embeddings are unavailable, so it stays offline-robust and
    deterministic. Real MCP systems expose hundreds of tools; dense+lexical fusion is the documented
    2-3x selection lever (RAG-MCP arXiv:2505.03275; Tool-to-Agent Retrieval arXiv:2511.01854)."""

    def __init__(self, tools: List[Dict[str, Any]], use_dense: bool = True, dense_weight: float = 0.5) -> None:
        self.tools = tools
        docs = [_tool_doc(t) for t in tools]
        self.tf = [Counter(d) for d in docs]
        n = max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {term: math.log((n + 1) / (c + 0.5)) + 1.0 for term, c in df.items()}
        self.dense_weight = dense_weight
        self._doc_emb = self._np = self._embed = None
        if use_dense and tools:
            try:  # dense arm is best-effort — pure-lexical fallback keeps this offline-safe
                import numpy as np
                from .dedup import _embed
                emb = np.asarray(_embed([_tool_text(t) for t in tools]), dtype=float)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self._doc_emb, self._np, self._embed = emb / norms, np, _embed
            except Exception:
                self._doc_emb = None

    def _lexical(self, query: str) -> List[float]:
        q = _tokens(query)
        return [sum(tf.get(term, 0) * self.idf.get(term, 0.0) for term in q) for tf in self.tf]

    def scores(self, query: str) -> List[float]:
        lex = self._lexical(query)
        if self._doc_emb is None:
            return lex
        np = self._np
        qe = np.asarray(self._embed([query]), dtype=float)[0]
        qe = qe / (np.linalg.norm(qe) or 1.0)
        dense = (self._doc_emb @ qe).tolist()
        L, D, w = _minmax(lex), _minmax(dense), self.dense_weight
        return [(1.0 - w) * L[i] + w * D[i] for i in range(len(self.tools))]

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        sc = self.scores(query)
        # stable: higher score first, ties broken by original index (deterministic)
        order = sorted(range(len(self.tools)), key=lambda i: (-sc[i], i))
        return [self.tools[i] for i in order[:k]]


def retrieve_tools(tools: List[Dict[str, Any]], query: str, k: int = 5,
                   use_dense: bool = True) -> List[Dict[str, Any]]:
    return ToolRetriever(tools, use_dense=use_dense).retrieve(query, k)
