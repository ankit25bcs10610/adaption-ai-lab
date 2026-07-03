"""Indic (Hindi/Devanagari + romanized) chart-QA slice — unlocks the $2k HackIndia track.

A thin decorator over synth_charts: it reuses the SAME ground-truth computation, localizing only the
chart labels and the question surface. Answers stay correct by construction; numeric gold stays ASCII
(the relaxed scorer and the VLM both speak ASCII digits — converting to Devanagari would unfairly tank
measured accuracy), while categorical gold is the on-chart localized string.

Paired mode emits the same chart+question in English AND Hindi sharing a pair_id, giving a clean
matched-pair Δaccuracy(hi−en) — the exact HackIndia impact story. Lazy matplotlib/PIL imports.
"""
from __future__ import annotations

import os
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

from . import synth_charts as sc
from .synth_charts import _subseed

# --- curated label maps (lenient: unknown labels fall back to English) -------------------
HI_LABELS = {
    "North": "उत्तर", "South": "दक्षिण", "East": "पूर्व", "West": "पश्चिम", "Central": "मध्य",
    "January": "जनवरी", "February": "फ़रवरी", "March": "मार्च", "April": "अप्रैल", "May": "मई", "June": "जून",
    "Sales": "बिक्री", "Revenue": "राजस्व", "Users": "उपयोगकर्ता", "Headcount": "कर्मचारी",
    "Temperature": "तापमान", "Share": "हिस्सा", "Region": "क्षेत्र", "Month": "माह",
    "Product": "उत्पाद", "Department": "विभाग",
    "Sales by Region": "क्षेत्र अनुसार बिक्री", "Revenue by Month": "माह अनुसार राजस्व",
    "Users by Product": "उत्पाद अनुसार उपयोगकर्ता", "Headcount by Department": "विभाग अनुसार कर्मचारी",
    "Temperature by Month": "माह अनुसार तापमान", "Market Share": "बाज़ार हिस्सा",
}
HI_ROMAN_LABELS = {
    "North": "Uttar", "South": "Dakshin", "East": "Purva", "West": "Pashchim", "Central": "Madhya",
    "January": "Janvari", "February": "Farvari", "March": "March", "April": "April", "May": "Mai", "June": "Jun",
    "Sales": "Bikri", "Revenue": "Rajasva", "Users": "Upyogkarta", "Headcount": "Karmchari",
    "Temperature": "Taapmaan", "Share": "Hissa", "Region": "Kshetra", "Month": "Maah",
    "Product": "Utpaad", "Department": "Vibhag",
    "Sales by Region": "Kshetra Anusaar Bikri", "Revenue by Month": "Maah Anusaar Rajasva",
}
_MAPS = {"hi": HI_LABELS, "hi-romanized": HI_ROMAN_LABELS, "en": None}
SCRIPT_FOR_LANG = {"hi": "devanagari", "hi-romanized": None, "en": None}

# --- Hindi / romanized question templates (surface only; GT unchanged) -------------------
HI_TEMPLATES = {
    "value_lookup": "{cat} के लिए {metric} का मान क्या है?",
    "max": "सबसे अधिक {metric} किस {xlabel} में है?",
    "min": "सबसे कम {metric} किस {xlabel} में है?",
    "compare": "क्या {cat_a} का {metric} {cat_b} से अधिक है?",
    "difference": "{cat_a} और {cat_b} के {metric} में कितना अंतर है?",
    "sum": "सभी {xlabel} का कुल {metric} कितना है?",
    "mean": "सभी {xlabel} का औसत {metric} कितना है?",
    "count": "कितने {xlabel} का {metric} {threshold} से अधिक है?",
    "trend": "समय के साथ {metric} का रुझान क्या है?",
    "proportion": "{cat} कुल का कितना प्रतिशत है?",
}
HR_TEMPLATES = {
    "value_lookup": "{cat} ke liye {metric} ka maan kya hai?",
    "max": "Sabse adhik {metric} kis {xlabel} mein hai?",
    "min": "Sabse kam {metric} kis {xlabel} mein hai?",
    "compare": "Kya {cat_a} ka {metric} {cat_b} se adhik hai?",
    "difference": "{cat_a} aur {cat_b} ke {metric} mein kitna antar hai?",
    "sum": "Sabhi {xlabel} ka kul {metric} kitna hai?",
    "mean": "Sabhi {xlabel} ka ausat {metric} kitna hai?",
    "count": "Kitne {xlabel} ka {metric} {threshold} se adhik hai?",
    "trend": "Samay ke saath {metric} ka rujhaan kya hai?",
    "proportion": "{cat} kul ka kitna pratishat hai?",
}
_TEMPLATES = {"hi": HI_TEMPLATES, "hi-romanized": HR_TEMPLATES}


def _tr(s: str, lang: str) -> str:
    m = _MAPS.get(lang)
    if not m:
        return s
    return m.get(s, s)  # lenient fallback


def localize_chart(cd: "sc.ChartData", lang: str) -> "sc.ChartData":
    out = deepcopy(cd)
    out.title = _tr(cd.title, lang)
    out.x_label = _tr(cd.x_label, lang)
    out.y_label = _tr(cd.y_label, lang)
    out.categories = [_tr(c, lang) for c in cd.categories]
    out.series_names = [_tr(s, lang) for s in cd.series_names]
    return out


def _find_devanagari_font() -> Optional[str]:
    try:
        from matplotlib import font_manager as fm
    except Exception:
        return None
    keys = ("devanagari", "nirmala", "lohit", "noto sans deva", "kohinoor", "mangal")
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in keys):
            return f.fname
    return None


def make_font_applier(script: Optional[str]):
    if script is None:
        return None
    fpath = _find_devanagari_font()
    if not fpath:
        return None

    def _apply(fig, ax):
        import matplotlib as mpl
        from matplotlib import font_manager as fm

        mpl.rcParams["axes.unicode_minus"] = False
        prop = fm.FontProperties(fname=fpath)
        for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
            t.set_fontproperties(prop)
        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lbl.set_fontproperties(prop)
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                txt.set_fontproperties(prop)
        for txt in ax.texts:  # pie slice labels, annotations
            txt.set_fontproperties(prop)

    return _apply


def _fill(qa: "sc.QAItem", cd_local: "sc.ChartData", en_to_local: Dict[str, str], lang: str) -> str:
    if lang == "en":
        return qa.question
    tmpl = _TEMPLATES[lang].get(qa.qa_kind)
    if not tmpl:
        return qa.question
    sup = qa.support
    slots = {
        "metric": cd_local.y_label,
        "xlabel": cd_local.x_label,
        "cat": en_to_local.get(sup.get("category", ""), sup.get("category", "")),
        "cat_a": en_to_local.get(sup.get("cat_a", ""), sup.get("cat_a", "")),
        "cat_b": en_to_local.get(sup.get("cat_b", ""), sup.get("cat_b", "")),
        "threshold": sc.num_str(sup.get("threshold", "")) if sup.get("threshold") is not None else "",
        "x": en_to_local.get(sup.get("category", ""), sup.get("category", "")),
    }
    try:
        return tmpl.format(**slots)
    except (KeyError, IndexError):
        return qa.question


def _localize_answer(qa: "sc.QAItem", en_to_local: Dict[str, str]) -> Any:
    """Categorical answers become the on-chart localized string; numeric/bool/trend stay ASCII."""
    if qa.answer_type == "category" and isinstance(qa.answer, str):
        return en_to_local.get(qa.answer, qa.answer)
    return qa.answer


def generate(
    n: int,
    out_dir: str,
    seed: int = 42,
    lang_mix: Optional[Dict[str, float]] = None,
    paired: bool = True,
    questions_per_chart: int = 2,
    render: bool = True,
) -> List[Dict[str, Any]]:
    """Generate Indic chart-QA examples. In paired mode each chart is emitted in en + hi with a shared
    pair_id (matched-pair Δ). Numeric gold ASCII; categorical gold is the localized on-chart string."""
    lang_mix = lang_mix or {"hi": 0.55, "hi-romanized": 0.15, "en": 0.30}
    langs = list(lang_mix.keys())
    weights = [lang_mix[l] for l in langs]
    non_en = [l for l in langs if l != "en"] or ["hi"]
    non_en_w = [lang_mix.get(l, 1.0) for l in non_en]
    if any(SCRIPT_FOR_LANG.get(l) == "devanagari" for l in langs) and _find_devanagari_font() is None:
        print("[viz-indic] WARNING: no Devanagari font found — Hindi charts will render as boxes (tofu). "
              "Install a Devanagari font or matplotlib[raqm]+Noto before releasing.")
    master = random.Random(seed + 101)
    out: List[Dict[str, Any]] = []
    idx = 0
    attempts = 0
    while len(out) < n and attempts < n * 40:
        attempts += 1
        rng = random.Random(_subseed(seed + 101, idx))
        idx += 1
        # bias to fully-translatable themes/charts
        ctype = master.choices(["bar", "line", "pie", "hbar"], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
        cd = sc.sample_chart(rng, ctype)
        makers = list(sc._MAKERS.get(ctype, []))
        rng.shuffle(makers)
        made = []
        for mk in makers:
            if len(made) >= questions_per_chart:
                break
            item = mk(cd, rng)
            if item is not None:
                made.append(item)
        if not made:
            continue
        pair_id = f"c{idx:06d}"
        # paired mode: always an English control + a NON-English twin (never en+en, which would dedup away)
        emit_langs = (
            ["en", master.choices(non_en, weights=non_en_w, k=1)[0]]
            if paired
            else [master.choices(langs, weights=weights, k=1)[0]]
        )
        for lang in emit_langs:
            cd_local = localize_chart(cd, lang)
            en_to_local = dict(zip(cd.categories, cd_local.categories))
            img_path = None
            if render:
                script = SCRIPT_FOR_LANG.get(lang)
                img_path = os.path.join(out_dir, "img", f"indic_{idx:06d}_{lang}.png")
                sc.render_chart(cd_local, img_path, apply_fonts=make_font_applier(script))
            for q in made:
                if len(out) >= n:
                    break
                out.append(
                    {
                        "image": img_path,
                        "question": _fill(q, cd_local, en_to_local, lang),
                        "answer": _localize_answer(q, en_to_local) if lang != "en" else q.answer,
                        "chart_type": cd.chart_type,
                        "qa_kind": q.qa_kind,
                        "meta": {
                            "source": "indic_synth",
                            "lang": lang,
                            "script": SCRIPT_FOR_LANG.get(lang),
                            "chart_type": cd.chart_type,
                            "qa_kind": q.qa_kind,
                            "answer_type": q.answer_type,
                            "pair_id": pair_id,
                            "seed": _subseed(seed + 101, idx),
                        },
                    }
                )
    return out
