"""Format-invariance + function-masking twins — anti-shortcut slices for BFCL-v4-style robustness.

Two matched-twin generators over ALREADY-BUILT examples (labels stay correct by construction):

1. **Format twins** — the same example with its tool documentation re-rendered as Python signatures,
   XML descriptors, or a compact list (`meta.doc_format`; see `format_utils.render_tools_as`). Gold is
   IDENTICAL. Basis: BFCL v4 "format sensitivity" — tool-specialized fine-tunes catastrophically
   overfit one documentation format (gorilla.cs.berkeley.edu/blogs/17_bfcl_v4_prompt_variation.html).

2. **Masked twins** — tool and parameter names replaced by neutral tokens (``func_i`` / ``arg_j``)
   while DESCRIPTIONS are kept, with the gold call renamed consistently (and re-validated). Kills
   naming-convention shortcuts: the model must select by description, not by lexical overlap.
   Basis: Hammer's "function masking" (arXiv:2410.04587).

Both twin kinds share a `meta.pair_id` WITH THEIR SOURCE example (assigned in place), so
`build_dataset.split()` keeps each twin set in one split — a twin of a train example can never leak
into test. Like the multilingual twins, these share the source's query text, so they must be appended
AFTER dedup (dedup would collapse them into the source).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

from .format_utils import DOC_FORMATS
from .schema_validator import validate_answer

_ALT_FORMATS = tuple(f for f in DOC_FORMATS if f != "json")


def _eligible(ex: Dict[str, Any]) -> bool:
    """Sources must have tools and must not already belong to a pair group (don't corrupt existing twins)."""
    return bool(ex.get("tools")) and not (ex.get("meta") or {}).get("pair_id")


def generate_format_twins(examples: List[Dict[str, Any]], n: int, seed: int = 42,
                          per_source: int = 2) -> List[Dict[str, Any]]:
    """Sample up to `n` source examples; emit `per_source` alternate-format twins each.

    Mutates each chosen source's meta to add the shared pair_id (split-group anchor). Gold, tools,
    query, and history are untouched — only meta.doc_format differs, which the prompt builders honor.
    """
    rng = random.Random(seed)
    idxs = [i for i, ex in enumerate(examples) if _eligible(ex)]
    rng.shuffle(idxs)
    out: List[Dict[str, Any]] = []
    for j, i in enumerate(idxs[:n]):
        src = examples[i]
        pair = f"fmt-{j}"
        src.setdefault("meta", {})["pair_id"] = pair
        for fmt in rng.sample(_ALT_FORMATS, k=min(per_source, len(_ALT_FORMATS))):
            twin = copy.deepcopy(src)
            twin["meta"] = dict(src.get("meta") or {})
            twin["meta"].update({"source": "format_twin", "doc_format": fmt, "pair_id": pair,
                                 "fmt_of": (src.get("meta") or {}).get("source")})
            out.append(twin)
    return out


def _mask_maps(tools: List[Dict[str, Any]]) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    tool_map = {t.get("name", f"tool{i}"): f"func_{i}" for i, t in enumerate(tools)}
    param_maps: Dict[str, Dict[str, str]] = {}
    for i, t in enumerate(tools):
        props = ((t.get("parameters") or {}).get("properties") or {})
        param_maps[t.get("name", f"tool{i}")] = {p: f"arg_{j}" for j, p in enumerate(props)}
    return tool_map, param_maps


def _mask_example(src: Dict[str, Any]) -> Dict[str, Any] | None:
    """Neutral-name twin: rename tools/params everywhere (schema + gold), keep descriptions + query."""
    tool_map, param_maps = _mask_maps(src["tools"])
    twin = copy.deepcopy(src)
    for t in twin["tools"]:
        old = t.get("name", "")
        pmap = param_maps.get(old, {})
        t["name"] = tool_map.get(old, old)
        params = t.get("parameters") or {}
        props = params.get("properties") or {}
        params["properties"] = {pmap.get(k, k): v for k, v in props.items()}
        if params.get("required"):
            params["required"] = [pmap.get(k, k) for k in params["required"]]
    ans = twin.get("answer") or {}
    if ans.get("type") == "tool_call":
        for c in ans.get("calls") or []:
            old = c.get("name", "")
            pmap = param_maps.get(old, {})
            c["name"] = tool_map.get(old, old)
            c["arguments"] = {pmap.get(k, k): v for k, v in (c.get("arguments") or {}).items()}
    ok, _ = validate_answer(twin["answer"], twin["tools"])
    return twin if ok else None


def generate_masked_twins(examples: List[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample up to `n` SINGLE-TURN sources (history may reference tool names, so multi-turn is skipped
    to stay correct-by-construction); emit one masked twin each, sharing pair_id with the source."""
    rng = random.Random(seed + 1)
    idxs = [i for i, ex in enumerate(examples) if _eligible(ex) and not ex.get("history")]
    rng.shuffle(idxs)
    out: List[Dict[str, Any]] = []
    j = 0
    for i in idxs:
        if len(out) >= n:
            break
        src = examples[i]
        twin = _mask_example(src)
        if twin is None:
            continue
        pair = f"mask-{j}"
        j += 1
        src.setdefault("meta", {})["pair_id"] = pair
        twin["meta"] = dict(twin.get("meta") or {})
        twin["meta"].update({"source": "masked_twin", "pair_id": pair,
                             "mask_of": (src.get("meta") or {}).get("source")})
        out.append(twin)
    return out
