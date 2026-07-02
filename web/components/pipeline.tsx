"use client";

import { Reveal } from "@/components/reveal";

const steps = [
  { n: "01", c: "run", title: "Build the dataset", desc: "Merge xLAM + ToolACE, curate to schema-valid positives, mix in hard negatives." },
  { n: "02", c: "run", title: "Dedup + novel split", desc: "MinHash + semantic dedup, cross-split leakage removed, tools held out for generalization." },
  { n: "03", c: "run", title: "Honest baseline", desc: "Eval the raw base model first — identical decoding, so the comparison is fair." },
  { n: "04", c: "cyan", title: "Train on AutoScientist", desc: "Upload, co-optimize data + recipe, get grade_before / grade_after / improvement." },
  { n: "05", c: "cyan", title: "DPO on the moat", desc: "Preference pairs: chosen = correct refuse/clarify, rejected = a hallucinated call." },
  { n: "06", c: "violet", title: "Release + demo", desc: "Publish to HF + Kaggle with an auto-filled model card, ship a live Space." },
] as const;

const badge = {
  run: "bg-run text-slate-950",
  cyan: "bg-cyan text-slate-950",
  violet: "bg-violet text-slate-950",
} as const;

export function Pipeline() {
  return (
    <section id="how" className="relative z-10 px-6 py-28">
      <div className="mx-auto max-w-6xl">
        <Reveal className="mb-14 max-w-2xl">
          <p className="mb-3 font-mono text-sm text-cyan">{"// the pipeline"}</p>
          <h2 className="font-display text-4xl font-bold tracking-tight sm:text-5xl">
            A data-centric loop, end to end.
          </h2>
          <p className="mt-4 text-lg text-muted-foreground">
            AutoScientist automates training — so the whole game is the data. Every stage is reproducible and seeded.
          </p>
        </Reveal>

        <ol className="grid gap-5 md:grid-cols-3">
          {steps.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.05}>
              <li className="relative h-full rounded-2xl glass p-6">
                <span className={`absolute -top-3 left-6 rounded px-2 py-0.5 font-mono text-xs ${badge[s.c]}`}>
                  {s.n}
                </span>
                <h3 className="mt-3 font-display text-lg font-semibold">{s.title}</h3>
                <p className="mt-2 text-sm text-muted-foreground">{s.desc}</p>
              </li>
            </Reveal>
          ))}
        </ol>
      </div>
    </section>
  );
}
