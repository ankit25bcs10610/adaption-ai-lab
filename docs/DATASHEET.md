# Datasheet — AutoScientist Tool-Caller dataset

Following *Datasheets for Datasets* (Gebru et al., 2021). Companion to `DATASET_CARD.md`.

## Motivation
- **Why created?** Most function-calling datasets contain only examples where a tool *should* be called.
  Real agents fail on the opposite decisions — inventing a call when none applies, or guessing a missing
  argument. This dataset teaches **when *not* to call a tool** (refuse / clarify / disambiguate), plus
  call-completeness and over-refusal resistance. Built for the Adaption AutoScientist Challenge
  (category: All Other Domains).
- **Created by:** the repository author (solo). Released open under Apache-2.0.

## Composition
- **Instances:** one tool-use decision each — `{tools (JSON Schema list), query, answer envelope, meta}`.
  Answer is `{"action":"call"|"refuse"|"clarify", ...}`.
- **Slices:** positives (real tool-calls, ToolACE-derived) · hard negatives (`no_tool`→refuse,
  `missing_arg`/`ambiguous`→clarify, `over_refusal`→call, `partial_parallel`→two calls) · multi-turn
  (`miss_param`/`miss_func`/`long_context`) · schema-drift (`add_required`/`retype_enum`/`rename`) ·
  execution-verified environment trajectories · a multilingual slice (en/hi/hi-rom/es/fr matched twins).
- **Size:** ~3k train + val/test + a novel-tools holdout; exact counts + realized-vs-intended mix in
  `stats.json` (`mix` block).
- **Labels:** correct **by construction** (synthesized from real schemas / verified by execution), not
  human-annotated → no annotation noise ceiling.
- **Splits:** train / val / test + `test_novel` (tools never seen in training — a generalization probe).
- **Confidentiality / PII:** none. Tool schemas + synthetic requests; no personal data.

## Collection & processing
- **Sources:** positives curated from `Team-ACE/ToolACE` (Apache-2.0); every other slice is original,
  synthesized from the real tool schemas. (Optional opt-in: `Agent-Ark/Toucan-1.5M`, Apache-2.0.)
- **Cleaning:** MinHash + slice-aware semantic dedup; cross-split leakage removed; **decontamination**
  against public BFCL/ToolACE-style probes; ToolACE conversation boilerplate stripped into real turns.
- **Quality control:** a two-pass adversarial audit (`docs/DATA_QUALITY_AUDIT.md`) that caught the moat
  nearly shipping empty (no_tool 8→239) and 36%→0% schema-invalid gold calls; a build-time `mix_ok`
  guard; every tool-call gold validated against its schema. Seed 42; fully reproducible.

## Uses
- **Intended:** SFT + preference (DPO) tuning of function-calling models that must decide *whether* and
  *how* to call tools, including safe abstention and multi-turn/multi-call.
- **Out of scope:** general chat; domains outside tool-use; strict-format consumers should post-validate
  the JSON (eval uses relaxed argument matching).

## Distribution & maintenance
- **Where:** Hugging Face + Kaggle (both public), plus the reproducible builder in the repo.
- **License:** Apache-2.0. **Provenance:** ToolACE attribution preserved.
- **Reproduce/verify:** `python -m autoscientist_toolcaller.build_dataset` → `stats.json`; `python -m autoscientist_toolcaller.manifest --verify`
  checks SHA-256 of every artifact against the committed manifest.

## Known limitations
- Synthetic phrasing is template-derived (labels correct; wording less varied than fully natural queries).
- Argument scoring is relaxed in eval.
- Multilingual seed covers 5 languages; the platform's `language_expansion` recipe scales further.
