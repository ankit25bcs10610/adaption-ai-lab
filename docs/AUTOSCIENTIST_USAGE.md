# Depth of AutoScientist / Adaptive Data usage

How this submission uses Adaption's platform end-to-end (judging criterion #4). Everything here is real
and reproducible from `src/train_adaption.py` + `config.yaml`.

## 1. Adaptive Data — data adaptation + evaluation (SDK)

The dataset is uploaded and run through Adaptive Data with an explicit recipe and brand-controls, then
the platform grades it on its held-out quality evaluation.

**Exact call** (`src/train_adaption.py`):
```python
client.datasets.run(
    dataset_id,
    column_mapping={"prompt": "prompt", "completion": "completion"},
    recipe_specification={"recipes": {"deduplication": True, "reasoning_traces": True}},
    brand_controls={
        "length": "concise",
        "blueprint": ("You are a reliable function-calling assistant. Emit schema-correct JSON tool "
                      "calls with all required arguments; REFUSE when no available tool applies; ASK "
                      "for clarification when a required argument is missing or the tool choice is "
                      "ambiguous. Never hallucinate a call or guess an argument."),
    },
)
```

- **`deduplication`** recipe — platform-side near-dup removal on top of our own MinHash+semantic pass.
- **`reasoning_traces`** recipe — the platform augments completions with reasoning (see §3).
- **`brand_controls.blueprint`** — encodes the whole thesis (call / refuse / clarify discipline) as a
  system blueprint applied to every generated completion, so the augmentation reinforces the moat.

## 2. Measured result (real, held-out quality grade)

| Run (dataset_id) | rows | score before → after | Δ | grade |
|---|---|---|---|---|
| `c4923b7f-3ee7-4691-bb1f-b47a85cf5097` | 2,440 | **7.0 → 8.1** | **+15.7%** | C → B |
| `a99c0c96-ff5b-490a-9aa9-372ea62d79d4` | 250 | 8.0 → 8.8 | +10.0% | B |

`improvement_percent` comes from `datasets.get_evaluation` (`score_before` / `score_after`, grade A–E).
This is Adaptive Data's **dataset-quality** grade — the data-centric platform's measurable improvement.

## 3. Platform value-add is real, not nominal

We downloaded the platform's processed output (`src/fetch_improved.py`) and inspected it. Adaptive Data
rewrites each row into a richer `enhanced_prompt` / `enhanced_completion` and attaches a `row_embedding`:
- **100% of completions were revised.** Example: where our original clarify flagged only an invalid enum,
  the enhanced completion flagged **both** the invalid enum **and** a missing required argument.
- The cleaned enhanced pairs are published (`enhanced_train_pc.jsonl`) as the artifact behind the grade.

## 4. AutoScientist — model training loop (console)

The Python SDK is dataset-centric; the model-training loop that co-optimizes data + recipe and emits
**weights** runs in the AutoScientist console (adaptionlabs.ai/app) on the adapted dataset. That run
produces the trained weights and the per-category held-out model number, which are then published to the
HF model repo + a Kaggle model. Our local harness (`scripts/run_all.sh`, `src/eval_*`) provides the
matching offline baseline→eval→decomposition→report so the platform number can be corroborated.

## 5. Reproduce

```bash
export ADAPTION_API_KEY=pt_live_...      # your key
python -m src.build_dataset --config config.yaml     # build + adapt-ready prompt/completion
python -m src.train_adaption --config config.yaml    # upload -> estimate -> run -> print evaluation_summary
python -m src.fetch_improved --dataset-id <id>       # download the platform's enhanced dataset
```
Uploads use a widened httpx timeout (multi-MB uploads) and `wait_for_completion(dataset_id)`.
