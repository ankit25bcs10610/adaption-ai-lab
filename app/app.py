"""Live demo — AutoScientist Tool-Caller (CPU, no weights required).

This runs a FAITHFUL, DETERMINISTIC simulation of the fine-tuned model's decision logic
(call / refuse / clarify) — the same engine as the web playground. It needs no GPU and no model
download, so the demo is instant and the "toggle a tool off → CALL becomes REFUSE" cause/effect is real.
It is a simulation of the behavior, not the weights; once the AutoScientist-trained weights are
published, swap in `app_gpu.py` to run the real model.
"""
import json
import re

import gradio as gr

STOP = {
    "the", "a", "an", "to", "for", "of", "in", "on", "me", "my", "please", "can", "you", "i", "is",
    "it", "and", "with", "get", "what", "whats", "s", "do", "now", "right", "give", "tell",
}


def tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 1 and t not in STOP]


def score_tool(tool, qtok):
    ttok = set(tokens((tool["name"] + " " + tool.get("description", "")).replace("_", " ")))
    return sum(1 for t in ttok if t in qtok)


def extract_arg(name, spec, query, qtok):
    q = query.lower()
    if spec.get("enum"):
        for e in spec["enum"]:
            if str(e).lower() in q:
                return e
        return None
    t = spec.get("type", "string")
    if t in ("integer", "number"):
        m = re.search(r"-?\d+(\.\d+)?", query)
        return (float(m.group()) if "." in m.group() else int(m.group())) if m else None
    if t == "boolean":
        if re.search(r"\b(true|yes|enable|on)\b", query, re.I):
            return True
        if re.search(r"\b(false|no|disable|off)\b", query, re.I):
            return False
        return None
    quoted = re.search(r'"([^"]+)"|\'([^\']+)\'', query)
    if quoted:
        return quoted.group(1) or quoted.group(2)
    proper = re.search(r"\b([A-Z][a-zA-Z]{2,})\b", query)
    if proper and not re.match(r"^(please|book|delete|write|what)$", proper.group(1), re.I):
        return proper.group(1)
    if name.lower() in qtok:
        return name
    return None


def simulate(tools, query):
    qtok = set(tokens(query))
    if not query.strip():
        return {"action": "clarify", "message": "What would you like to do?", "rationale": "Empty request."}
    if not tools:
        return {"action": "refuse", "message": "No tools are available, so I can't act on this request.",
                "rationale": "No enabled tools."}
    scored = sorted(((t, score_tool(t, qtok)) for t in tools), key=lambda x: -x[1])
    best, best_s = scored[0]
    if best_s == 0:
        return {"action": "refuse", "message": "None of the available tools can handle this request.",
                "rationale": "No tool's name/description overlaps with the request."}
    if len(scored) > 1 and scored[1][1] == best_s and best_s > 0:
        return {"action": "clarify",
                "message": f"Did you mean `{best['name']}` or `{scored[1][0]['name']}`? Let me know which.",
                "rationale": "Two tools match the request equally well."}
    props = (best.get("parameters") or {}).get("properties", {})
    required = (best.get("parameters") or {}).get("required", list(props))
    args, missing = {}, []
    for n in required:
        v = extract_arg(n, props.get(n, {}), query, qtok)
        if v is None:
            missing.append(n)
        else:
            args[n] = v
    if missing:
        need = ", ".join(f"`{m}`" for m in missing)
        return {"action": "clarify",
                "message": f"I can use `{best['name']}`, but I need the required {need} before I can proceed.",
                "rationale": f"Missing required argument(s): {', '.join(missing)}."}
    return {"action": "call", "calls": [{"name": best["name"], "arguments": args}],
            "rationale": f"`{best['name']}` matches and all required arguments are present."}


PRESET_TOOLS = [
    {"name": "get_weather", "description": "Get the current weather for a city",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "book_flight", "description": "Book a flight between two cities on a date",
     "parameters": {"type": "object", "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}, "date": {"type": "string"}}, "required": ["origin", "destination", "date"]}},
    {"name": "convert_currency", "description": "Convert an amount from one currency to another",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from": {"type": "string"}, "to": {"type": "string"}}, "required": ["amount", "from", "to"]}},
]
TOOLS_BY_NAME = {t["name"]: t for t in PRESET_TOOLS}
BADGE = {"call": "🟢 TOOL CALL", "clarify": "🟣 NEEDS INFO", "refuse": "🔵 REFUSED"}


def run(enabled_names, query):
    tools = [TOOLS_BY_NAME[n] for n in enabled_names]
    r = simulate(tools, query)
    envelope = ({"action": "call", "calls": r["calls"]} if r["action"] == "call"
                else {"action": r["action"], "message": r["message"]})
    md = f"### {BADGE[r['action']]}\n\n**why:** {r['rationale']}"
    return json.dumps(envelope, indent=2), md


with gr.Blocks(title="AutoScientist Tool-Caller", theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown(
        "# AutoScientist Tool-Caller — live demo\n"
        "The tool-caller that knows **when *not* to call**. Toggle a tool off and re-run — a valid "
        "**call** becomes a **refusal**. This is a **faithful deterministic simulator** of the decision "
        "logic (no GPU) — *not the trained model; weights pending the AutoScientist run.* "
        "[Model card](https://huggingface.co/pandeyankit84/autoscientist-toolcaller) · "
        "[Dataset](https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset)"
    )
    with gr.Row():
        with gr.Column():
            enabled = gr.CheckboxGroup(
                choices=[t["name"] for t in PRESET_TOOLS],
                value=[t["name"] for t in PRESET_TOOLS],
                label="Available tools (toggle to change the outcome)",
            )
            query = gr.Textbox(label="User request", value='What is the weather in "Mumbai"?', lines=2)
            gr.Examples(
                examples=[['What is the weather in "Mumbai"?'], ["Book me a flight to Delhi."],
                          ["Write me a poem about the monsoon."], ["Convert 100 USD to INR."]],
                inputs=query, label="Try a scenario (call · clarify · refuse · call)",
            )
            btn = gr.Button("Run", variant="primary")
        with gr.Column():
            out_json = gr.Code(label="Model output (JSON envelope)", language="json")
            out_md = gr.Markdown()
    btn.click(run, inputs=[enabled, query], outputs=[out_json, out_md])
    query.submit(run, inputs=[enabled, query], outputs=[out_json, out_md])

if __name__ == "__main__":
    demo.launch()
