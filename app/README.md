---
title: AutoScientist Tool-Caller
emoji: 🟢
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: The tool-caller that knows when NOT to call a tool
---

# AutoScientist Tool-Caller — live demo

A faithful, deterministic simulation of the fine-tuned model's decision logic (call / refuse / clarify).
Toggle a tool off and re-run — a valid **call** becomes a **refusal**. No GPU, instant.

- Model: https://huggingface.co/pandeyankit84/autoscientist-toolcaller
- Dataset: https://huggingface.co/datasets/pandeyankit84/autoscientist-toolcaller-dataset
- Code: https://github.com/ankit25bcs10610/adaption-ai-lab

`app.py` runs the simulator (CPU). Once the AutoScientist-trained weights are published, `app_gpu.py`
runs the real model on a GPU/ZeroGPU Space.
