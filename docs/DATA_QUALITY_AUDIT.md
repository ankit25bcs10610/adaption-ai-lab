# Data-quality audit — how the moat almost shipped empty

This is a data-centric submission, so the dataset itself is the product. Before the final release we
ran an adversarial audit of the build pipeline (a fan-out of independent review agents, each grounded
in the actual code and the emitted `stats.json`). It surfaced a class of bug that is invisible in a
green test suite but silently guts a data-centric result: **the slices the whole thesis rests on were
being generated and then thrown away.**

We document it here because *finding and fixing it* is the data-centric methodology, not an aside.

## The finding

The submission's headline claim is a **refuse / clarify moat**: unlike ordinary function-calling data,
this set teaches the model *when not to call a tool*. The audit read `stats.json` and found the moat
was almost entirely absent from the shipped data:

| Slice (the moat)                     | Intended | **Before** | **After** |
|--------------------------------------|:--------:|:----------:|:---------:|
| `no_tool` (refuse — no tool applies) | ~10% of total | **8 rows** | **644 rows** |
| `miss_param` (clarify — arg missing across turns) | material | **1 row** | **70 rows** |
| `ambiguous` (clarify — which tool?)  | material | **0 rows** | **167 rows** |

Eight training rows cannot teach abstention. The claim and the data had diverged.

## Three root causes (all confirmed in-code, all now fixed + regression-tested)

1. **A stop-word overlap guard rejected almost every `no_tool` candidate.**
   `make_no_tool` keyed its "does a tool already cover this?" guard on the request's *first word*
   (`req.lower().split()[0]`) — almost always a stop-word like *what / can / write* that appears in
   some tool description, so nearly every candidate was discarded.
   *Fix:* a grounded **Hammer construction** — take a real positive and remove the tool it needs from
   the offered set, so the request is genuinely unsatisfiable → refuse. The guard now compares
   *content words*, not the first token. (`autoscientist_toolcaller/hard_negatives.py`)

2. **Dedup collapsed templated slices because it keyed on the query text alone.**
   The Hammer `no_tool` reuses a real positive's query (with different tools + a *refuse* answer), and
   the templated `ambiguous` / `miss_param` slices shared a fixed query across different tool sets.
   Query-only dedup treated all of these as duplicates and deleted them.
   *Fix:* dedup on the **full training signature** — `answer-type + tool-set + query`. Two rows are
   duplicates only when the request, the tools on offer, *and* the required behavior all match.
   (`autoscientist_toolcaller/dedup.py`)

3. **`miss_param` selected tools uniformly, but only tools with a required arg can build one.**
   Most tools have no required arg to withhold, so `make_miss_param` returned `None` ~99% of the time
   and the slice starved to a single row.
   *Fix:* pre-filter the pool to tools with ≥1 required arg before sampling. (`autoscientist_toolcaller/multiturn.py`)

## Two more defects the same audit caught

4. **Mixing math undershot every slice.** Each slice was sized as if it were the only additive one
   (`n = base · r/(1−r)`); with four additive slices, every realized share fell below target
   (positives drifted to ~62%, hard negatives to ~17% instead of 22%). Replaced with a **shared
   denominator** so realized ≈ intended, and `stats.json` now records intended-vs-realized shares
   plus a `mix_ok` guard that fails loudly if the moat drifts out of band. (`autoscientist_toolcaller/build_dataset.py`)

5. **DPO could emit poison pairs.** A "corrupted" negative that swapped to a synonym tool, or dropped
   a merely-*optional* argument, could accidentally still be correct — training the model *toward* the
   rejected sample. Added a `_confirmed_wrong` gate + rejection sampling: every `rejected` is now
   provably not the correct answer (different tool, missing *required* arg, or changed required
   value), and `drop_arg` only ever drops required args. (`autoscientist_toolcaller/build_preference.py`)

(Two correctness bugs outside the data path were fixed in the same pass: a `<think>` reasoning block
could be mis-parsed as the answer JSON during eval — now the parser reads only after `</think>`; and
the Kaggle release handle mislabelled the model *variation* as a version.)

## Result

After the fixes, on the same config and seed:

- `no_tool` **8 → 644**, `miss_param` **1 → 70**, `ambiguous` **0 → 167** (current realized counts).
- Realized source shares — positives (ToolACE) **43.2%**, hard-negative **17.1%**, multi-turn **12.9%**,
  schema-drift **7.5%**, multilingual **3.2%**, format-twin **9.8%**, masked-twin **3.9%** — now track the
  intended mix instead of collapsing toward positives.
- `no_tool` sits at **8.8% of the total set**, inside the ~10% research optimum band; `mix_ok` passes.
- Total examples: **7,566** (the published HF/Kaggle set) — 7,323 across `train / val / test` + 243 in the
  `test_novel` novel-tools holdout, spanning **7,315 unique tools**.
- **0 schema-invalid `tool_call` golds shipped** — a build-time drop-guard validates every gold against
  its tool's Draft-7 schema (`stats.json:schema_invalid_dropped`), so "correct by construction" is enforced,
  not just asserted.

Regression tests were added for each fix (`tests/smoke_test.py`: Hammer construction excludes the
gold tool, the `<think>` firewall, the DPO poison guard), so the moat can't silently empty out again.

Reproduce: `python -m autoscientist_toolcaller.build_dataset --config config.yaml` then inspect `data/out/stats.json`
(`mix` block).

---

## Round 2 — a second adversarial pass (25 agents)

A second research + adversarial-verification pass audited the *fixed* pipeline and found more, most
notably another confirmed poison bug:

- **Schema-drift poison (36% of rename golds were schema-invalid).** `make_rename` hardcoded the string
  `"example"` / `"ACME-42"` for every required argument, ignoring declared type/enum — so integer and
  enum fields received invalid values. Those gold calls are graded through the validator, so they were
  literally teaching the model to emit invalid calls. Fixed by centralizing a type/enum-aware
  `sample_value()` (`autoscientist_toolcaller/format_utils.py`, shared with multi-turn) and adding a `validate_answer`
  drop-guard in `schema_drift.generate()`. Now **0** invalid rename golds.

Four new correct-by-construction slices/defenses were added on top:

- **Over-refusal traps** (`over_refusal`, 79 rows) — a hedged but fully-satisfiable request whose gold
  is to **call**. Direct counterweight to the enlarged refuse/clarify slice, so the model doesn't learn
  to over-abstain. (A model that refuses here scores 0.)
- **Partial-parallel** (`partial_parallel`, 77 rows) — two intents in one request; the gold is **two**
  calls. The only slice that stresses call *completeness*.
- **Execution-verified multi-call trajectories** (`autoscientist_toolcaller/envs.py`) — 2–3 order-independent calls verified
  by replaying them against a deterministic environment.
- **Decontamination** (`autoscientist_toolcaller/decontaminate.py`) — every training query is checked (n-gram + embedding)
  against public BFCL/ToolACE-style probes and dropped on overlap; a `contamination` block lands in
  `stats.json`. This defends the held-out claim against leakage.

Two dedup subtleties fixed along the way: dedup is now **slice-group aware** (an over-refusal example,
being a deliberate hedged near-paraphrase of a positive, is no longer deleted as a "duplicate" of it),
and the execution-labeled env DPO pairs — previously generated but **never consumed** — are now merged
into `pref.jsonl` with the hardest confirmed-wrong negative selected per pair.

All of the above is regression-tested in `tests/smoke_test.py` and gated for release by a blocking
`autoscientist_toolcaller/release.py` preflight (missing artifacts / placeholders / missing LICENSE).
