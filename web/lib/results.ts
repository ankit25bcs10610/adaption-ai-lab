/**
 * Benchmark data for the landing page.
 *
 * These are REPRESENTATIVE placeholders. To wire real numbers, run the eval pipeline and then
 * `npm run sync:results` (scripts/sync-results.mjs) which reads ../results/baseline.json and
 * ../results/eval.json and overwrites `benchmarks` + flips `representative` to false.
 */
export interface BenchRow {
  label: string;
  base: number; // 0-100
  ft: number; // 0-100
  lowerIsBetter?: boolean;
}

export const representative = true;

export const headlineStats = [
  { value: "−54pp", label: "hallucination rate", accent: "run" as const },
  { value: "+42pp", label: "overall accuracy", accent: "cyan" as const },
  { value: "3B", label: "params · QLoRA", accent: "fg" as const },
];

export const benchmarks: BenchRow[] = [
  { label: "Overall accuracy", base: 41, ft: 83 },
  { label: "Positive (tool-call) accuracy", base: 55, ft: 86 },
  { label: "Refusal accuracy", base: 30, ft: 92 },
  { label: "Clarify accuracy", base: 25, ft: 80 },
  { label: "Hallucination rate", base: 62, ft: 8, lowerIsBetter: true },
];
