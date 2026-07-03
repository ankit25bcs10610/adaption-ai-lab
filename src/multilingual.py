"""Multilingual tool-calling slice — matched en / hi / hi-romanized / es / fr twins.

Adaption's headline feature is 242-language support, yet almost no entry uses it. We add a
correct-by-construction multilingual slice: the SAME tools + gold envelope, with the user request phrased
in several languages. Because the twins share a gold answer and only the query language differs, they
enable a matched-pair Δaccuracy(lang − en) — a clean cross-language robustness signal and a direct fit
for the HackIndia track.

Behavior is unchanged from the English moat: call / refuse / clarify. Labels are correct by construction
(the answer envelope is fixed; only the query text is localized). Seeded + offline: phrasings are
hand-authored (accurate, not machine-mangled). Beyond these seed languages, the platform's
`language_expansion` recipe scales to the full 242 (see train_adaption / config).
"""
from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

_WEATHER = {"name": "get_weather", "description": "Get the current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
_FLIGHT = {"name": "book_flight", "description": "Book a flight between two cities on a date",
           "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}}
_EMAIL = {"name": "send_email", "description": "Send an email to a recipient",
          "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"]}}
_CONVERT = {"name": "convert_currency", "description": "Convert an amount from one currency to another",
            "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_ccy": {"type": "string"}, "to_ccy": {"type": "string"}}, "required": ["amount", "from_ccy", "to_ccy"]}}
_STOCK = {"name": "get_stock_price", "description": "Get the latest price for a stock ticker",
          "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}}
_REMINDER = {"name": "set_reminder", "description": "Set a reminder at a given time",
             "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "time": {"type": "string"}}, "required": ["text", "time"]}}
_TRANSLATE = {"name": "translate_text", "description": "Translate text into a target language",
              "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string", "enum": ["en", "hi", "fr", "es"]}}, "required": ["text", "target_lang"]}}

_CITIES = ["Mumbai", "Delhi", "Pune", "Chennai", "Kolkata", "Jaipur", "Hyderabad", "Bengaluru", "Goa", "Surat"]
_LANGS = ("en", "hi", "hi-rom", "es", "fr")


def _call_specs():
    return [
        {"tool": _WEATHER, "distractors": [_STOCK, _EMAIL],
         "value": lambda rng: rng.choice(_CITIES), "args": lambda v: {"city": v},
         "q": {"en": lambda v: f"What's the weather in {v}?",
               "hi": lambda v: f"{v} में मौसम कैसा है?",
               "hi-rom": lambda v: f"{v} mein mausam kaisa hai?",
               "es": lambda v: f"¿Qué tiempo hace en {v}?",
               "fr": lambda v: f"Quel temps fait-il à {v} ?"}},
        {"tool": _WEATHER, "distractors": [_CONVERT, _REMINDER],
         "value": lambda rng: rng.choice(_CITIES), "args": lambda v: {"city": v},
         "q": {"en": lambda v: f"Tell me today's temperature in {v}.",
               "hi": lambda v: f"आज {v} का तापमान बताइए।",
               "hi-rom": lambda v: f"Aaj {v} ka taapmaan bataiye.",
               "es": lambda v: f"Dime la temperatura de hoy en {v}.",
               "fr": lambda v: f"Donne-moi la température aujourd'hui à {v}."}},
        {"tool": _STOCK, "distractors": [_WEATHER, _CONVERT],
         "value": lambda rng: rng.choice(["AAPL", "TSLA", "INFY", "TCS", "RELIANCE", "WIPRO"]), "args": lambda v: {"ticker": v},
         "q": {"en": lambda v: f"Get the latest price of {v}.",
               "hi": lambda v: f"{v} का ताज़ा शेयर भाव बताओ।",
               "hi-rom": lambda v: f"{v} ka taaza share bhaav batao.",
               "es": lambda v: f"Dame el último precio de {v}.",
               "fr": lambda v: f"Donne-moi le dernier cours de {v}."}},
        {"tool": _CONVERT, "distractors": [_STOCK, _WEATHER],
         "value": lambda rng: rng.choice([50, 100, 250, 500, 1000, 2000]), "args": lambda v: {"amount": v, "from_ccy": "USD", "to_ccy": "INR"},
         "q": {"en": lambda v: f"Convert {v} USD to INR.",
               "hi": lambda v: f"{v} अमेरिकी डॉलर को भारतीय रुपये में बदलो।",
               "hi-rom": lambda v: f"{v} US dollar ko Indian rupee mein badlo.",
               "es": lambda v: f"Convierte {v} USD a INR.",
               "fr": lambda v: f"Convertis {v} USD en INR."}},
        {"tool": _REMINDER, "distractors": [_EMAIL, _WEATHER],
         "value": lambda rng: rng.choice(["9am tomorrow", "6pm today", "Monday morning", "tonight"]), "args": lambda v: {"text": "call the bank", "time": v},
         "q": {"en": lambda v: f"Remind me to call the bank at {v}.",
               "hi": lambda v: f"मुझे {v} बैंक को कॉल करने की याद दिलाओ।",
               "hi-rom": lambda v: f"Mujhe {v} bank ko call karne ki yaad dilao.",
               "es": lambda v: f"Recuérdame llamar al banco a las {v}.",
               "fr": lambda v: f"Rappelle-moi d'appeler la banque à {v}."}},
        {"tool": _TRANSLATE, "distractors": [_EMAIL, _STOCK],
         "value": lambda rng: rng.choice(["hi", "fr", "es"]), "args": lambda v: {"text": "good morning", "target_lang": v},
         "q": {"en": lambda v: f"Translate 'good morning' to {v}.",
               "hi": lambda v: f"'good morning' का {v} में अनुवाद करो।",
               "hi-rom": lambda v: f"'good morning' ka {v} mein anuvaad karo.",
               "es": lambda v: f"Traduce 'good morning' al {v}.",
               "fr": lambda v: f"Traduis 'good morning' en {v}."}},
    ]


_REFUSE_Q = {  # off-domain: no listed tool applies -> refuse
    "en": ["Write me a short poem about the monsoon.", "Tell me a joke about cricket.", "What's the meaning of life?", "Sing me a lullaby.", "Recommend a good biryani recipe."],
    "hi": ["मानसून पर एक छोटी कविता लिखो।", "क्रिकेट पर एक चुटकुला सुनाओ।", "जीवन का अर्थ क्या है?", "मुझे एक लोरी सुनाओ।", "एक अच्छी बिरयानी की विधि बताओ।"],
    "hi-rom": ["Monsoon par ek chhoti kavita likho.", "Cricket par ek chutkula sunao.", "Jeevan ka arth kya hai?", "Mujhe ek lori sunao.", "Ek achhi biryani ki vidhi batao."],
    "es": ["Escríbeme un poema corto sobre el monzón.", "Cuéntame un chiste sobre críquet.", "¿Cuál es el sentido de la vida?", "Cántame una nana.", "Recomiéndame una buena receta de biryani."],
    "fr": ["Écris-moi un court poème sur la mousson.", "Raconte-moi une blague sur le cricket.", "Quel est le sens de la vie ?", "Chante-moi une berceuse.", "Recommande-moi une bonne recette de biryani."],
}
_CLARIFY_Q = {  # references a tool but omits a required arg -> clarify
    "en": ["Book me a flight to Goa.", "Send an email saying I'll be late.", "Set a reminder for the meeting.", "Book a flight for next Friday."],
    "hi": ["मेरे लिए गोवा की फ्लाइट बुक करो।", "एक ईमेल भेजो कि मुझे देर हो जाएगी।", "मीटिंग के लिए एक रिमाइंडर सेट करो।", "अगले शुक्रवार की फ्लाइट बुक करो।"],
    "hi-rom": ["Mere liye Goa ki flight book karo.", "Ek email bhejo ki mujhe der ho jaayegi.", "Meeting ke liye ek reminder set karo.", "Agle Friday ki flight book karo."],
    "es": ["Resérvame un vuelo a Goa.", "Envía un correo diciendo que llegaré tarde.", "Pon un recordatorio para la reunión.", "Reserva un vuelo para el próximo viernes."],
    "fr": ["Réserve-moi un vol pour Goa.", "Envoie un e-mail disant que je serai en retard.", "Mets un rappel pour la réunion.", "Réserve un vol pour vendredi prochain."],
}
# per clarify template (aligned index): tools on offer + the missing required arg
_CLARIFY_SPECS = [
    ([_FLIGHT, _EMAIL], "origin and date"),
    ([_EMAIL, _REMINDER], "recipient"),
    ([_REMINDER, _EMAIL], "time"),
    ([_FLIGHT, _WEATHER], "origin and destination"),
]


def generate(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Matched twins across languages for call/refuse/clarify. `n` ~ total examples."""
    rng = random.Random(seed + 909)
    specs = _call_specs()
    out: List[Dict[str, Any]] = []
    pair = 0
    kinds = ["call", "call", "refuse", "clarify"]
    while len(out) < n:
        pair += 1
        kind = rng.choice(kinds)
        if kind == "call":
            spec = rng.choice(specs)
            v = spec["value"](rng)
            tools = [spec["tool"]] + spec["distractors"]
            ans = {"type": "tool_call", "calls": [{"name": spec["tool"]["name"], "arguments": spec["args"](v)}]}
            qbuild = {lang: spec["q"][lang](v) for lang in _LANGS}
        elif kind == "refuse":
            tools = [_WEATHER, _STOCK, _CONVERT]
            ans = {"type": "refuse", "content": "None of the available tools can handle this request."}
            i = rng.randrange(len(_REFUSE_Q["en"]))
            qbuild = {lang: _REFUSE_Q[lang][i] for lang in _LANGS}
        else:  # clarify
            i = rng.randrange(len(_CLARIFY_Q["en"]))
            tools, need = _CLARIFY_SPECS[i]
            ans = {"type": "clarify", "content": f"I can help, but I need the required {need} before I can proceed."}
            qbuild = {lang: _CLARIFY_Q[lang][i] for lang in _LANGS}
        for lang in _LANGS:
            out.append({
                "tools": copy.deepcopy(tools),
                "query": qbuild[lang],
                "answer": copy.deepcopy(ans),
                "meta": {"source": "multilingual", "hn_kind": None, "lang": lang, "pair_id": pair, "ml_kind": kind},
            })
    trim = (n // len(_LANGS)) * len(_LANGS) if n % len(_LANGS) == 0 else (n // len(_LANGS) + 1) * len(_LANGS)
    return out[:trim]
