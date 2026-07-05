# AutoScientist platform demo (the challenge's demo bonus)

The official challenge awards **bonus points for building and releasing a demo of AutoScientist**. This
repo ships two *kinds* of demo:

1. **The trained model** — call / refuse / clarify, interactively:
   - Live site: <https://autoscientist-toolcaller.vercel.app> · HF Space: `app/` (Gradio)
2. **The AutoScientist *platform* workflow** — the data-centric loop this submission actually ran
   (**upload → grade → improve → re-grade → train**), which is what *this* page is about.

## One-command replay (offline, no API key)

```bash
python -m autoscientist_toolcaller.demo_platform            # prints the walkthrough
python -m autoscientist_toolcaller.demo_platform --transcript results/autoscientist_demo.txt
```

It narrates each platform step (mapped to the real SDK calls in
[`autoscientist_toolcaller/train_adaption.py`](../autoscientist_toolcaller/train_adaption.py)) and prints
the **real recorded dataset-quality grades**. It makes no network call: it replays canned numbers (the
same ones in `MODEL_CARD.md` / `DATASET_CARD.md`) and, if `results/adaption_run.json` is present, the
actual recorded run. Full platform write-up: [`AUTOSCIENTIST_USAGE.md`](AUTOSCIENTIST_USAGE.md).

## Transcript (fresh-checkout replay)

```text
========================================================================
  Adaption AutoScientist — platform workflow demo (offline replay)
  upload → grade → improve → re-grade → train   (no network, no API key)
========================================================================

▶ 1. UPLOAD
    $ client.datasets.upload_file(train_tab.jsonl) → dataset_id
    The adapted, correct-by-construction dataset is uploaded to Adaptive Data.

▶ 2. GRADE
    $ client.datasets.get_status(dataset_id) → quality grade (before)
    Adaptive Data scores dataset quality on its rubric — the honest 'before'.

▶ 3. IMPROVE
    $ client.datasets.run(recipe=[deduplication, reasoning_traces], brand_controls=blueprint)
    Platform recipes + a call/refuse/clarify brand-controls blueprint produce an ENHANCED dataset.

▶ 4. RE-GRADE
    $ evaluation_summary → grade_after + improvement_percent
    The enhanced dataset is re-graded; the delta is the data-centric improvement.

▶ 5. TRAIN
    $ AUTOSCIENTIST tab → Launch/Train → weights + held-out % (web console)
    AutoScientist trains the model on the enhanced data; produces weights + the held-out gate number.

── Measured dataset-quality grades  [source: canned (matches the cards)] ────────────
    dataset      rows  before  after      Δ%  grade / status
    a99c0c96      250       8    8.8     10%  B · completed
    c4923b7f     2440       7    8.1   15.7%  C→B · partial (1,000/2,440 under the free-tier cap)
    d92279d3     2557       —      —       —  — · pending — full grade completes on the console run

Note: the grade above is Adaptive Data's *dataset-quality* improvement (the data-centric lever).
The held-out *model* accuracy — the challenge's eligibility gate — is produced by the TRAIN step
in the web console. Real SDK: autoscientist_toolcaller/train_adaption.py · docs/AUTOSCIENTIST_USAGE.md
========================================================================
```

> To also capture a screen recording (a nice-to-have on top of this reproducible transcript), run the
> command above in a terminal and record with asciinema/Loom, then link it here and in the submission.
