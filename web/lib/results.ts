/**
 * Landing-page data.
 *
 * Two kinds of numbers, kept strictly separate for credibility:
 *   1. MEASURED — real outputs from the pipeline (Adaptive Data quality grade + the data-quality
 *      audit before/after). These are facts.
 *   2. PROJECTED — the base-vs-fine-tuned model table. Adaptive Data is data-centric (it grades the
 *      dataset, it doesn't hand back weights), so these become real only after training a model on the
 *      improved data (a GPU step). They are labeled "target / illustrative" in the UI, never as facts.
 *
 * `npm run sync:results` overwrites `benchmarks` from ../results/*.json and flips `projected` to false
 * once a real eval has run.
 */
export interface BenchRow {
  label: string;
  base: number; // 0-100
  ft: number; // 0-100
  lowerIsBetter?: boolean;
}

export interface AuditRow {
  label: string;
  before: string;
  after: string;
  note: string;
}

/** The base-vs-fine-tuned model table is illustrative until a GPU eval runs. */
export const projected = true;

/** MEASURED — Adaptive Data's dataset-quality grade on the fixed dataset (real platform output). */
export const dataQuality = {
  scoreBefore: 7.0,
  scoreAfter: 8.1,
  improvementPercent: 15.7,
  gradeBefore: "C",
  gradeAfter: "B",
  rows: 2440,
};

/** Hero stats — all MEASURED and honest. */
export const headlineStats = [
  { value: "+15.7%", label: "dataset quality · Adaptive Data", accent: "run" as const },
  { value: "239", label: "abstain cases · recovered from 8", accent: "cyan" as const },
  { value: "3B", label: "base · Qwen2.5-Coder", accent: "fg" as const },
];

/** MEASURED — the data-quality audit: what two adversarial passes found and fixed (all real). */
export const audit: AuditRow[] = [
  { label: "Refuse cases (no_tool)", before: "8", after: "239", note: "the moat was starved by a dedup bug" },
  { label: "Clarify cases (miss_param)", before: "1", after: "36", note: "generator selected argless tools" },
  { label: "Disambiguate cases", before: "0", after: "133", note: "templated queries collapsed under dedup" },
  { label: "Schema-invalid gold calls", before: "36%", after: "0%", note: "type/enum-aware value synthesis" },
];

/** PROJECTED — target behavior once a model is trained on the improved dataset (illustrative). */
export const benchmarks: BenchRow[] = [
  { label: "Overall accuracy", base: 41, ft: 83 },
  { label: "Positive (tool-call) accuracy", base: 55, ft: 86 },
  { label: "Refusal accuracy", base: 30, ft: 92 },
  { label: "Clarify accuracy", base: 25, ft: 80 },
  { label: "Hallucination rate", base: 62, ft: 8, lowerIsBetter: true },
];
