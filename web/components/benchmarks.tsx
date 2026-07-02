"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Reveal } from "@/components/reveal";
import { benchmarks, representative } from "@/lib/results";

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
    <section id="bench" className="relative z-10 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <Reveal className="mb-12 max-w-2xl">
          <p className="mb-3 font-mono text-sm text-violet">{"// results"}</p>
          <h2 className="font-display text-4xl font-bold tracking-tight sm:text-5xl">Base vs. fine-tuned.</h2>
          <p className="mt-4 text-lg text-muted-foreground">
            Same greedy decoding, bootstrapped standard error. Lower is better for hallucination.
          </p>
        </Reveal>

        <Reveal>
          <div className="rounded-2xl border-glow glass p-6 sm:p-8">
            <div className="space-y-6">
              {benchmarks.map((row) => (
                <div key={row.label}>
                  <div className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">
                      {row.label} {row.lowerIsBetter && <span className="text-cyan">↓</span>}
                    </span>
                    <span className="font-mono text-muted-foreground">
                      <span className="text-muted-foreground/60">{row.base}%</span> →{" "}
                      <span className="text-run">{row.ft}%</span>
                    </span>
                  </div>
                  <Bar base={row.base} ft={row.ft} lowerIsBetter={row.lowerIsBetter} />
                </div>
              ))}
            </div>
            {representative && (
              <p className="mt-6 text-xs text-muted-foreground/70">
                Representative figures shown for illustration. <code className="text-muted-foreground">src/eval_bfcl.py</code>{" "}
                emits the final numbers; run <code className="text-muted-foreground">npm run sync:results</code> to wire them in.
              </p>
            )}
          </div>
        </Reveal>
      </div>
    </section>
  );
}
