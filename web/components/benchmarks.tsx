"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Reveal } from "@/components/reveal";
import { Section, SectionHeader } from "@/components/section";
import { NumberTicker } from "@/components/ui/number-ticker";
import { audit, benchmarks, dataQuality, projected } from "@/lib/results";

function Bar({ base, ft, lowerIsBetter }: { base: number; ft: number; lowerIsBetter?: boolean }) {
  const reduce = useReducedMotion();
  return (
    <div className="relative h-3 overflow-hidden rounded-full bg-foreground/5">
      <div
        className={`absolute inset-y-0 rounded-full ${lowerIsBetter ? "bg-red-500/40" : "bg-foreground/15"}`}
        style={{ width: `${base}%` }}
      />
      <motion.div
        className="absolute inset-y-0 rounded-full bg-gradient-to-r from-run to-cyan"
        initial={reduce ? { width: `${ft}%` } : { width: 0 }}
        whileInView={{ width: `${ft}%` }}
        viewport={{ once: true, amount: 0.5 }}
        transition={{ duration: 1.4, ease: [0.2, 0.7, 0.2, 1] }}
      />
    </div>
  );
}

export function Benchmarks() {
  return (
    <Section id="bench">
      <SectionHeader eyebrow="results" title="Measured wins, honest targets." tone="violet" className="mb-12">
        The dataset is the product, so we lead with what the platform actually measured — and clearly mark
        what still needs a training run.
      </SectionHeader>

      <div className="grid gap-6 lg:grid-cols-2">
          {/* MEASURED — Adaptive Data quality grade */}
          <Reveal>
            <div className="flex h-full flex-col rounded-2xl border-glow glass p-6 sm:p-8">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-run/15 px-2.5 py-1 text-xs font-medium text-run">
                  <span className="h-1.5 w-1.5 rounded-full bg-run" /> Measured
                </span>
                <span className="text-xs text-muted-foreground">Adaptive Data · {dataQuality.rows.toLocaleString()} rows</span>
              </div>
              <h3 className="mt-4 font-display text-xl font-semibold">Dataset-quality grade</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                A data-centric platform grades the <em>dataset</em>. This is the improvement it reported.
              </p>
              <div className="mt-6 flex items-end gap-4">
                <div className="font-display text-6xl font-bold text-grad">
                  <NumberTicker value={dataQuality.improvementPercent} prefix="+" suffix="%" decimals={1} />
                </div>
                <div className="pb-2 font-mono text-sm text-muted-foreground">
                  score {dataQuality.scoreBefore.toFixed(1)} → <span className="text-run">{dataQuality.scoreAfter.toFixed(1)}</span>
                </div>
              </div>
              <div className="mt-5 flex items-center gap-3 font-mono text-sm">
                <span className="rounded-md border border-border/60 px-2.5 py-1 text-muted-foreground">grade {dataQuality.gradeBefore}</span>
                <span className="text-muted-foreground">→</span>
                <span className="rounded-md border border-run/40 bg-run/10 px-2.5 py-1 font-semibold text-run">grade {dataQuality.gradeAfter}</span>
              </div>
            </div>
          </Reveal>

          {/* MEASURED — the data-quality audit before/after */}
          <Reveal>
            <div className="flex h-full flex-col rounded-2xl glass p-6 sm:p-8">
              <div className="flex items-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full bg-cyan/15 px-2.5 py-1 text-xs font-medium text-cyan">
                  <span className="h-1.5 w-1.5 rounded-full bg-cyan" /> Measured
                </span>
                <span className="text-xs text-muted-foreground">two adversarial audit passes</span>
              </div>
              <h3 className="mt-4 font-display text-xl font-semibold">The data-quality audit</h3>
              <p className="mt-1 text-sm text-muted-foreground">
                What a bug hunt on the build pipeline found — and fixed. The moat had nearly shipped empty.
              </p>
              <ul className="mt-5 space-y-3">
                {audit.map((r) => (
                  <li key={r.label} className="grid grid-cols-[1fr_auto] items-baseline gap-x-3 border-b border-border/40 pb-3 last:border-0">
                    <span className="text-sm font-medium">{r.label}</span>
                    <span className="font-mono text-sm tabular-nums">
                      <span className="text-muted-foreground/80">{r.before}</span>
                      <span className="mx-1 text-muted-foreground">→</span>
                      <span className="font-semibold text-run">{r.after}</span>
                    </span>
                    <span className="col-span-2 text-xs text-muted-foreground/80">{r.note}</span>
                  </li>
                ))}
              </ul>
            </div>
          </Reveal>
        </div>

        {/* PROJECTED — base vs fine-tuned model behavior (clearly labeled) */}
        <Reveal className="mt-6">
          <div className="rounded-2xl glass p-6 sm:p-8">
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="flex items-center gap-2">
                  <span className="inline-flex items-center gap-1.5 rounded-full bg-violet/15 px-2.5 py-1 text-xs font-medium text-violet">
                    Target
                  </span>
                  <h3 className="font-display text-xl font-semibold">Base vs. fine-tuned model</h3>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  Illustrative targets for the behavior the moat trains. Same greedy decoding, bootstrapped SE.
                </p>
              </div>
            </div>
            <div className="space-y-6">
              {benchmarks.map((row) => (
                <div key={row.label}>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">
                      {row.label} {row.lowerIsBetter && <span className="text-cyan">↓</span>}
                    </span>
                    <span className="font-mono text-muted-foreground">
                      <span className="text-muted-foreground/80">{row.base}%</span> →{" "}
                      <span className="text-run">{row.ft}%</span>
                    </span>
                  </div>
                  <Bar base={row.base} ft={row.ft} lowerIsBetter={row.lowerIsBetter} />
                </div>
              ))}
            </div>
            {projected && (
              <p className="mt-6 text-xs text-muted-foreground/80">
                Illustrative targets — the model table becomes real after training on the improved dataset
                (a GPU step). The one-command harness (<code className="text-muted-foreground">scripts/run_all.sh</code>){" "}
                runs baseline → multi-seed eval → paired significance → gap decomposition → report.
              </p>
            )}
          </div>
        </Reveal>
    </Section>
  );
}
