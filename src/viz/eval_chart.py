"""ChartQA-style relaxed-accuracy evaluation — hardened per adversarial review.

The scorer is the offline-testable correctness core. It implements the hardened comparison algorithm:
Unicode/Indic digit transliteration, explicit-percent reconciliation (never magnitude-inferred),
exact-zero handling for zero gold, yes/no gated on boolean gold only (no "0"/"1" collision), range
parsing, proper (non-greedy) fuzzy list matching, South-Asian lakh grouping, and full-match numeric
parsing that rejects ranges/garbage. See src/viz — verify_eval spec for the full rationale (F1–F30).

Numeric answers: correct within ±5% relative tolerance (exact for zero gold). Categorical: NFC +
casefold + whitespace equality. Grades strict and relaxed; breaks down by chart_type, qa_kind, lang,
and a reasoning-vs-descriptive split; base-vs-fine-tuned via aggregate().
"""
from __future__ import annotations

import math
import re
import unicodedata
from itertools import permutations
from typing import Any, Dict, List, Optional, Tuple

TOL_DEFAULT = 0.05

_CURRENCY = set("$€£¥₹₩₨¢")
_RANGE_RE = re.compile(r"^\s*([+-]?[\d.,\s]*\d)\s*(?:-|–|—|~|\.\.|to|between)\s*([+-]?[\d.,\s]*\d)\s*$", re.I)
_YES = {"yes", "y", "true", "correct"}
_NO = {"no", "n", "false", "incorrect"}
_MAG_WORDS = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_MAG_LETTER = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


# --------------------------------------------------------------------------------------
# Digit / separator transliteration (F8/F9) — run before any parsing
# --------------------------------------------------------------------------------------
def translate_digits(s: str) -> str:
    out = []
    for ch in s:
        if ch.isdigit():
            try:
                out.append(str(unicodedata.digit(ch)))
                continue
            except (TypeError, ValueError):
                pass
        if ch == "٫":  # arabic decimal separator
            out.append(".")
        elif ch in ("٬", "،"):  # arabic thousands / comma -> drop
            continue
        else:
            out.append(ch)
    return "".join(out)


def _strip_group_seps(s: str) -> str:
    for sep in (" ", " ", " ", " "):
        s = s.replace(sep, "")
    return s


# --------------------------------------------------------------------------------------
# Numeric parsing with explicit percent flag (F1/F2/F10/F11/F12/F13)
# --------------------------------------------------------------------------------------
def _normalize_separators(core: str) -> str:
    """Decide decimal vs thousands. Rightmost of {. ,} is the decimal; the other is grouping.
    Accepts both Western (,\\d{3}) and South-Asian (,\\d{2}) grouping."""
    has_dot, has_comma = "." in core, "," in core
    if has_dot and has_comma:
        dec = "." if core.rfind(".") > core.rfind(",") else ","
        other = "," if dec == "." else "."
        core = core.replace(other, "")
        if dec == ",":
            core = core.replace(",", ".")
        return core
    if has_comma and not has_dot:
        # single comma followed by 1-2 digits -> decimal; else grouping (lakh or thousands)
        if core.count(",") == 1 and re.search(r",\d{1,2}$", core):
            return core.replace(",", ".")
        return core.replace(",", "")
    if has_dot and core.count(".") > 1:  # EU thousands "1.234.567"
        return core.replace(".", "")
    return core


def parse_number_with_flags(x: Any) -> Tuple[Optional[float], bool]:
    if isinstance(x, bool):
        return None, False  # bool is not a number here (F6)
    if isinstance(x, (int, float)):
        return (float(x), False) if math.isfinite(x) else (None, False)
    if x is None:
        return None, False
    s = translate_digits(str(x).strip())
    if not s:
        return None, False
    if _RANGE_RE.match(s):  # ranges route elsewhere (F1/F14)
        return None, False
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    neg = False
    if re.match(r"^\(.*\)$", s):  # accounting negative
        neg, s = True, s[1:-1].strip()
    s = s.replace("−", "-")  # unicode minus
    s = "".join(c for c in s if c not in _CURRENCY).strip()
    s = _strip_group_seps(s)
    # terminal magnitude: word form, or a single trailing uppercase letter
    mult = 1.0
    mword = re.search(r"([a-zA-Z]+)$", s)
    if mword:
        w = mword.group(1)
        if w.lower() in _MAG_WORDS:
            mult = _MAG_WORDS[w.lower()]
            s = s[: mword.start()].strip()
        elif len(w) == 1 and w in _MAG_LETTER:  # uppercase K/M/B/T only
            mult = _MAG_LETTER[w]
            s = s[: mword.start()].strip()
        else:
            return None, is_pct  # trailing non-magnitude letters (e.g. "MW", "kg") -> reject numeric
    s = _normalize_separators(s)
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", s):  # reject "1.", "1,2,3", ranges, garbage (F1/F13)
        return None, is_pct
    v = float(s) * mult
    if neg:
        v = -v
    return (v, is_pct) if math.isfinite(v) else (None, is_pct)


def numbers_match(gv: float, gp: bool, pv: float, pp: bool, tol: float) -> bool:
    if gv is None or pv is None or not math.isfinite(gv) or not math.isfinite(pv):
        return False
    if gp != pp:  # one side is a percent, the other isn't -> only a bare FRACTION (<1) can reconcile
        bare = pv if gp else gv
        if abs(bare) < 1.0:
            if gp:
                pv = pv * 100
            else:
                gv = gv * 100
        else:
            return False  # e.g. Sales 20 vs "20%": a stray % on a non-fraction is a mismatch (Bug 1)
    d = abs(gv)
    if d < 1e-9:  # zero gold -> exact (F4)
        return pv == 0.0
    return abs(pv - gv) <= tol * d


# --------------------------------------------------------------------------------------
# Ranges (F14)
# --------------------------------------------------------------------------------------
def parse_range(x: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(x, str):
        return None
    m = _RANGE_RE.match(translate_digits(x))
    if not m:
        return None
    a, _ = parse_number_with_flags(m.group(1))
    b, _ = parse_number_with_flags(m.group(2))
    if a is None or b is None:
        return None
    return (min(a, b), max(a, b))


# --------------------------------------------------------------------------------------
# String / yes-no (F18/F19/F20/F6/F7)
# --------------------------------------------------------------------------------------
def normalize_string(x: Any) -> str:
    if x is None:
        return "\x00miss"
    s = unicodedata.normalize("NFKC", str(x))
    s = translate_digits(s)
    s = s.casefold()
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1]
    return s.rstrip(".!?;:,").strip()


def yesno_of(x: Any) -> Optional[bool]:
    n = normalize_string(x)
    if n in _YES:
        return True
    if n in _NO:
        return False
    if re.fullmatch(r"(yes|no)[.!,]?", n):
        return n.startswith("yes")
    return None


# --------------------------------------------------------------------------------------
# Scalar + list dispatch
# --------------------------------------------------------------------------------------
def scalar_match(gold: Any, pred: Any, tol: float = TOL_DEFAULT) -> bool:
    if pred is None:
        return False
    if isinstance(gold, bool):  # boolean gold -> yes/no ONLY (F6)
        py = yesno_of(pred)
        return py is not None and py == gold
    gr, pr = parse_range(gold), parse_range(pred)
    if gr and pr:
        return numbers_match(gr[0], False, pr[0], False, tol) and numbers_match(gr[1], False, pr[1], False, tol)
    gv, gp = parse_number_with_flags(gold)
    pv, pp = parse_number_with_flags(pred)
    if gv is not None and pv is not None:
        return numbers_match(gv, gp, pv, pp, tol)
    if gv is None and pv is None:  # yes/no only when neither is a clean number (F6)
        gy, py = yesno_of(gold), yesno_of(pred)
        if gy is not None and py is not None:
            return gy == py
    return normalize_string(gold) == normalize_string(pred)


def _split_list(s: Any) -> List[str]:
    return [t.strip() for t in re.split(r"[,;]", str(s)) if t.strip()]  # never split on "and" (F15)


def lists_match(gold_list: List[Any], pred: Any, tol: float, ordered: bool) -> bool:
    pred_items = pred if isinstance(pred, list) else _split_list(pred)
    if len(pred_items) != len(gold_list):
        return False
    if ordered:
        return all(scalar_match(g, p, tol) for g, p in zip(gold_list, pred_items))
    # unordered: require a perfect matching (F16) — exhaustive for small k
    k = len(gold_list)
    if k <= 7:
        idxs = range(k)
        for perm in permutations(idxs):
            if all(scalar_match(gold_list[i], pred_items[perm[i]], tol) for i in idxs):
                return True
        return False
    # large k: greedy on a match matrix (approximate)
    used = [False] * k
    for g in gold_list:
        hit = next((j for j in range(k) if not used[j] and scalar_match(g, pred_items[j], tol)), None)
        if hit is None:
            return False
        used[hit] = True
    return True


# trend/correlation words across en / hi (Devanagari) / romanized -> a canonical bucket
_TREND_CANON = {
    "increasing": "up", "increase": "up", "rising": "up", "rise": "up", "upward": "up", "growing": "up",
    "grew": "up", "up": "up", "बढ़ता": "up", "बढ़ रहा है": "up", "वृद्धि": "up", "badhta": "up", "badh": "up",
    "decreasing": "down", "decrease": "down", "falling": "down", "declining": "down", "downward": "down",
    "down": "down", "घटता": "down", "घट रहा है": "down", "कमी": "down", "ghatta": "down", "ghat": "down",
    "flat": "flat", "stable": "flat", "constant": "flat", "स्थिर": "flat", "sthir": "flat",
    "positive": "pos", "सकारात्मक": "pos", "negative": "neg", "नकारात्मक": "neg", "no clear": "none",
}


def _canon_trend(x: Any) -> str:
    n = normalize_string(x)
    for k in sorted(_TREND_CANON, key=len, reverse=True):
        if k in n:
            return _TREND_CANON[k]
    return n


def relaxed_match(gold: Any, pred: Any, qa_kind: Optional[str] = None, tol: float = TOL_DEFAULT) -> bool:
    if pred is None:
        return False
    if qa_kind == "trend":  # language-agnostic trend/correlation matching (Indic + English)
        cg = _canon_trend(gold)
        return cg in {"up", "down", "flat", "pos", "neg", "none"} and cg == _canon_trend(pred)
    if qa_kind == "proportion" and isinstance(pred, str):  # accept "23.4" or "23.4%" for a percent gold
        pred = pred.strip().rstrip("%").strip()
    if isinstance(gold, (list, tuple)):
        return lists_match(list(gold), pred, tol, ordered=False)
    if isinstance(pred, str):  # multi-value pred for a scalar gold (F17)
        # Split on ';' or comma-FOLLOWED-BY-SPACE only — so a grouped number ("1,20,000") stays whole.
        parts = [p for p in re.split(r";|,\s", pred) if p.strip()]
        if len(parts) >= 2 and all(
            parse_number_with_flags(p)[0] is not None or parse_range(p) is not None for p in parts
        ):
            return False
    return scalar_match(gold, pred, tol)


def strict_match(gold: Any, pred: Any) -> bool:
    if pred is None:
        return False
    from .format_utils import answer_to_text

    return answer_to_text(gold).strip().casefold() == str(pred).strip().casefold()


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
_REASONING_KINDS = {"compare", "count", "trend", "difference", "proportion", "mean"}


def reasoning_or_descriptive(qa_kind: str) -> str:
    return "reasoning" if qa_kind in _REASONING_KINDS else "descriptive"


def _bootstrap_se(bits: List[int], n_boot: int = 1000, seed: int = 42) -> float:
    if not bits:
        return 0.0
    import random

    rng = random.Random(seed)
    n = len(bits)
    means = []
    for _ in range(n_boot):
        means.append(sum(bits[rng.randrange(n)] for _ in range(n)) / n)
    mu = sum(means) / len(means)
    return (sum((m - mu) ** 2 for m in means) / len(means)) ** 0.5


def judge_chart(example: Dict[str, Any], output_text: str, tol: float = TOL_DEFAULT) -> Dict[str, Any]:
    gold = example["answer"]
    qk = example.get("qa_kind")
    parsed_ok = output_text is not None and str(output_text).strip() != ""
    correct = relaxed_match(gold, output_text, qk, tol) if parsed_ok else False
    strict = strict_match(gold, output_text) if parsed_ok else False
    return {
        "correct": correct,
        "strict": strict,
        "parsed_ok": parsed_ok,
        "chart_type": example.get("chart_type", "other"),
        "qa_kind": qk or "other",
        "lang": example.get("meta", {}).get("lang", "en"),
        "split": reasoning_or_descriptive(qk or ""),
    }


def _bucket(verdicts: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for v in verdicts:
        b = out.setdefault(v[key], {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(v["correct"])
    for b in out.values():
        b["accuracy"] = b["correct"] / b["n"] if b["n"] else None
    return out


def evaluate(records: List[Dict[str, Any]], generate_fn, tol: float = TOL_DEFAULT) -> Dict[str, Any]:
    from .format_utils import build_eval_prompt_messages

    verdicts = []
    for ex in records:
        out = generate_fn(build_eval_prompt_messages(ex))
        verdicts.append(judge_chart(ex, out, tol))
    bits = [int(v["correct"]) for v in verdicts]
    strict_bits = [int(v["strict"]) for v in verdicts]
    return {
        "n": len(verdicts),
        "relaxed_accuracy": sum(bits) / len(bits) if bits else 0.0,
        "relaxed_stderr": _bootstrap_se(bits),
        "strict_accuracy": sum(strict_bits) / len(strict_bits) if strict_bits else 0.0,
        "parse_failure_rate": sum(not v["parsed_ok"] for v in verdicts) / len(verdicts) if verdicts else 0.0,
        "by_chart_type": _bucket(verdicts, "chart_type"),
        "by_qa_kind": _bucket(verdicts, "qa_kind"),
        "by_lang": _bucket(verdicts, "lang"),
        "by_split": _bucket(verdicts, "split"),
    }
