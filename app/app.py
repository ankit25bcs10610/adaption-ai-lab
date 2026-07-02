"""Gradio ZeroGPU demo for the fine-tuned function-calling model.

The demo's whole point is the narrative: give it a request no tool can satisfy, and watch it REFUSE
instead of hallucinating a call. Users pick a scenario (or write their own tools + request) and see the
single JSON envelope the model emits.

ZeroGPU rules honored: `import spaces` first; model loaded to cuda at module scope; gradio/spaces/
huggingface_hub NOT pinned in requirements.txt.
"""
import json

import spaces  # must precede torch / transformers
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "YOUR_USERNAME/autoscientist-toolcall"  # <- set to your published model

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

SYSTEM = """You are a precise function-calling assistant.

You are given a list of available tools as JSON Schemas. For the user's request, respond with a SINGLE \
JSON object and nothing else, using exactly one of these shapes:
- {"action": "call", "calls": [{"name": "<tool_name>", "arguments": {...}}]}
- {"action": "refuse", "message": "<why no tool applies>"}
- {"action": "clarify", "message": "<what you need>"}
Only use listed tools. Include every required argument; never guess a missing one. Output JSON only.

Available tools:
"""

EXAMPLES = {
    "Valid call": (
        json.dumps([{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }], indent=2),
        "What's the weather in Mumbai right now?",
    ),
    "No applicable tool (should REFUSE)": (
        json.dumps([{
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]},
        }], indent=2),
        "Write me a poem about the monsoon.",
    ),
    "Missing required arg (should CLARIFY)": (
        json.dumps([{
            "name": "book_flight",
            "description": "Book a flight",
            "parameters": {"type": "object",
                           "properties": {"origin": {"type": "string"},
                                          "destination": {"type": "string"},
                                          "date": {"type": "string"}},
                           "required": ["origin", "destination", "date"]},
        }], indent=2),
        "Book me a flight to Delhi.",
    ),
}


@spaces.GPU(duration=90)
def run(tools_json: str, query: str, temperature: float):
    try:
        tools = json.loads(tools_json)
    except json.JSONDecodeError as e:
        return f"⚠️ Tools JSON is invalid: {e}"
    prompt = SYSTEM + json.dumps(tools, indent=2) + f"\n\nUser request:\n{query}"
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs,
            max_new_tokens=512,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    # pretty-print if it's valid JSON
    try:
        return json.dumps(json.loads(text), indent=2)
    except json.JSONDecodeError:
        return text


def load_example(name):
    return EXAMPLES[name]


with gr.Blocks(title="AutoScientist Tool-Caller") as demo:
    gr.Markdown(
        "# 🛠️ AutoScientist Tool-Caller\n"
        "Fine-tuned to **refuse or clarify instead of hallucinating** tool calls. "
        "Try the *No applicable tool* example and watch it decline."
    )
    with gr.Row():
        with gr.Column():
            picker = gr.Dropdown(list(EXAMPLES), label="Scenario", value="No applicable tool (should REFUSE)")
            tools_box = gr.Code(label="Available tools (JSON Schema)", language="json")
            query_box = gr.Textbox(label="User request", lines=2)
            temp = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="Temperature")
            go = gr.Button("Run", variant="primary")
        with gr.Column():
            out_box = gr.Code(label="Model output (JSON envelope)", language="json")

    picker.change(load_example, picker, [tools_box, query_box])
    go.click(run, [tools_box, query_box, temp], out_box)
    demo.load(load_example, picker, [tools_box, query_box])

if __name__ == "__main__":
    demo.launch()
