"""Tool retrieval — rank tools by relevance so tool-calling scales past a promptful (advanced).

Real function-calling systems expose hundreds of tools; the prompt can't hold them all. This TF-IDF
retriever ranks tools (name + description + parameter names/descriptions) against a query, so only the
top-k go into the prompt. Pure stdlib, deterministic, offline — recall@k is exactly checkable.
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


class ToolRetriever:
    """TF-IDF retriever over a fixed set of tool schemas."""

    def __init__(self, tools: List[Dict[str, Any]]) -> None:
        self.tools = tools
        docs = [_tool_doc(t) for t in tools]
        self.tf = [Counter(d) for d in docs]
        n = max(len(docs), 1)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {term: math.log((n + 1) / (c + 0.5)) + 1.0 for term, c in df.items()}

    def scores(self, query: str) -> List[float]:
        q = _tokens(query)
        return [sum(tf.get(term, 0) * self.idf.get(term, 0.0) for term in q) for tf in self.tf]

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        sc = self.scores(query)
        # stable: higher score first, ties broken by original index (deterministic)
        order = sorted(range(len(self.tools)), key=lambda i: (-sc[i], i))
        return [self.tools[i] for i in order[:k]]


def retrieve_tools(tools: List[Dict[str, Any]], query: str, k: int = 5) -> List[Dict[str, Any]]:
    return ToolRetriever(tools).retrieve(query, k)
