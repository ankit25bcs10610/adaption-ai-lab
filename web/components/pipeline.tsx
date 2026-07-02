"use client";

import { Reveal } from "@/components/reveal";
import { Section, SectionHeader } from "@/components/section";

const steps = [
  { n: "01", c: "run", title: "Build the dataset", desc: "Curate ToolACE to schema-valid positives, synthesize hard negatives + execution-verified env examples." },
  { n: "02", c: "run", title: "Dedup + decontaminate", desc: "Slice-aware MinHash + semantic dedup, cross-split leakage removed, and a decontamination pass vs public probes." },
  { n: "03", c: "run", title: "Honest baseline", desc: "Eval the raw base model first — identical greedy decoding, so the comparison is fair." },
  { n: "04", c: "cyan", title: "Train on AutoScientist", desc: "Upload, co-optimize data + recipe, get grade_before / grade_after / improvement (real: +15.7%, C→B)." },
  { n: "05", c: "cyan", title: "DPO on the moat", desc: "Preference pairs: chosen = verified refuse/clarify/call, rejected = the hardest confirmed-wrong near-miss." },
  { n: "06", c: "violet", title: "Release + demo", desc: "Manifest + preflight, then publish to HF + Kaggle with an auto-filled model card." },
] as const;

const badge = {
  run: "bg-run text-slate-950",
  cyan: "bg-cyan text-slate-950",
  violet: "bg-violet text-slate-950",
} as const;

export function Pipeline() {
  return (
    <Section id="how">
      <SectionHeader eyebrow="the pipeline" title="A data-centric loop, end to end." tone="cyan" className="mb-14">
        AutoScientist automates training — so the whole game is the data. Every stage is reproducible,
        seeded, and one command (<code className="font-mono text-sm text-muted-foreground/90">scripts/run_all.sh</code>).
      </SectionHeader>

      <ol className="grid gap-5 md:grid-cols-3">
        {steps.map((s, i) => (
          <Reveal key={s.n} delay={i * 0.05}>
            <li className="relative h-full rounded-2xl glass p-6 transition-colors hover:border-border">
              <span className={`absolute -top-3 left-6 rounded px-2 py-0.5 font-mono text-xs ${badge[s.c]}`}>
                {s.n}
              </span>
              <h3 className="mt-3 font-display text-lg font-semibold">{s.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{s.desc}</p>
            </li>
          </Reveal>
        ))}
      </ol>
    </Section>
  );
}
